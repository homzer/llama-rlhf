import json
import os
import random
from pathlib import Path

import fire
import numpy as np

from policy_train_policy_gradient_logits_convex import train_policy_gradient_logits_convex
from policy_train_policy_gradient_off import collect_actor_forward_buffer_with_label
from policy_train_ppo_with_evaluate import evaluate_actor
from policy_train_ppo_with_rule_rm import collect_actor_buffer_with_label, collect_rule_based_verifier_buffer
from src.dataset import JsonDataset
from src.entities import IterationHandler
from src.evaluator import get_evaluator
from src.parallel.data_parallel.datawriter import ParallelDataWriter
from src.parallel.data_parallel.utils import scatter_object_to_data_parallel_region
from src.parallel.initialize import setup_model_parallel, get_data_parallel_world_size, set_barrier, get_rank
from src.ppo.buffer import RolloutBuffer, CriticRolloutBuffer
from src.ppo.parallel_buffer import ParallelRolloutBuffer
from src.utils import json_load, print_current_func_args, json_dump, masked_mean


class ParallelRolloutSelector:  # TODO: timeout
    def __init__(self, task: str, log_dir: str, num_samples_per_prompt: int):
        self.log_dir = log_dir
        self.filename = "rollout.jsonl"
        self.evaluator = get_evaluator(task)
        self.num_samples_per_prompt = num_samples_per_prompt

    def write(self, policy_rollout_buffer):
        # save to pool
        datawriter = ParallelDataWriter(os.path.join(self.log_dir, self.filename), mode='a', final_gather=False)
        for data in policy_rollout_buffer.get(1):
            datawriter.write(json.dumps(dict(
                instruction=data.instructions.tolist()[0],
                response=data.responses.tolist()[0],
                label=data.labels.tolist()[0]
            )) + '\n')
        del datawriter
        set_barrier()

    def read(self, policy_rollout_buffer) -> list:  # ensure at least one chosen response as much as possible
        policy_rollout_buffer = ParallelRolloutBuffer(**policy_rollout_buffer)
        policy_rollout_buffer.gather_from_data_parallel_region()
        instructions = set(policy_rollout_buffer["instructions"].tolist())  # deduplicate
        responses = set(policy_rollout_buffer["responses"].tolist())  # deduplicate

        files = sorted(Path(self.log_dir).glob("rollout.*.jsonl"))
        assert len(files) > 0
        datadict = dict()
        for file in files:
            for data in json_load(file):
                if data["instruction"] not in datadict:
                    datadict[data["instruction"]] = dict(chosen=[], rejected=[], label=data["label"])
                if self.evaluator.eval(data["response"], data["label"]) is True:
                    datadict[data["instruction"]]["chosen"].append(data["response"])
                else:
                    datadict[data["instruction"]]["rejected"].append(data["response"])

        results = []
        for instruction in instructions:
            if len(datadict[instruction]["chosen"]) > 0:
                response = random.choice(datadict[instruction]["chosen"])
                if response not in responses:
                    label = datadict[instruction]["label"]
                    results.append(dict(instruction=instruction, output=response, label=label))

        random.shuffle(results)
        # check remainder in data parallel region
        while len(results) % get_data_parallel_world_size() > 0:
            results.append(results[len(results) % get_data_parallel_world_size()])
        results = scatter_object_to_data_parallel_region(results)
        print(f"Number of additional selected samples: {len(results)}")

        return results


class RolloutSelector:
    def __init__(self, task: str, log_dir: str, num_samples_per_prompt: int):
        self.log_dir = log_dir
        self.filename = "rollout.jsonl"
        self.evaluator = get_evaluator(task)
        self.num_samples_per_prompt = num_samples_per_prompt

    def write(self, policy_rollout_buffer):
        # save to pool
        results = []
        for data in policy_rollout_buffer.get(1):
            results.append(dict(
                instruction=data.instructions.tolist()[0],
                response=data.responses.tolist()[0],
                label=data.labels.tolist()[0]
            ))
        if get_rank() == 0:
            json_dump(results, os.path.join(self.log_dir, self.filename))
        set_barrier()

    def read(self, policy_rollout_buffer) -> list:  # ensure at least one chosen response as much as possible
        instructions = set(policy_rollout_buffer["instructions"].tolist())  # deduplicate
        responses = set(policy_rollout_buffer["responses"].tolist())  # deduplicate

        datadict = dict()
        for data in json_load(os.path.join(self.log_dir, self.filename)):
            if data["instruction"] not in datadict:
                datadict[data["instruction"]] = dict(chosen=[], rejected=[], label=data["label"])
            if self.evaluator.eval(data["response"], data["label"]) is True:
                datadict[data["instruction"]]["chosen"].append(data["response"])
            else:
                datadict[data["instruction"]]["rejected"].append(data["response"])

        results = []
        for instruction in instructions:
            if len(datadict[instruction]["chosen"]) > 0:
                response = random.choice(datadict[instruction]["chosen"])
                if response not in responses:
                    label = datadict[instruction]["label"]
                    results.append(dict(instruction=instruction, output=response, label=label))

        random.shuffle(results)
        print(f"Number of additional selected samples: {len(results)}")

        return results


class RolloutBalancePool:
    def __init__(self, task):
        self.chosen = []
        self.rejected = []
        self.evaluator = get_evaluator(task)

    def load(self, file):
        print(f"Loading from {file}")
        for data in json_load(file):
            instruction = data["instruction"]
            label = data["label"]
            if isinstance(data["output"], list):
                for response in data["output"]:
                    if self.evaluator.eval(response, label) is True:
                        self.chosen.append(dict(instruction=instruction, output=response, label=label))
                    else:
                        self.rejected.append(dict(instruction=instruction, output=response, label=label))
            else:
                assert isinstance(data["output"], str)
                response = data["output"]
                if self.evaluator.eval(response, label) is True:
                    self.chosen.append(dict(instruction=instruction, output=response, label=label))
                else:
                    self.rejected.append(dict(instruction=instruction, output=response, label=label))
        print(f"Loading done! Number of chosen: {len(self.chosen)}. Number of rejected: {len(self.rejected)}")

    def put(self, policy_rollout_buffer):
        for data in policy_rollout_buffer.get(1):
            instruction = data.instructions.tolist()[0]
            response = data.responses.tolist()[0]
            label = data.labels.tolist()[0]
            if self.evaluator.eval(response, label) is True:
                self.chosen.append(dict(instruction=instruction, output=response, label=label))
            else:
                self.rejected.append(dict(instruction=instruction, output=response, label=label))

    @staticmethod
    def sample(population: list, k: int) -> list:
        if len(population) > k:
            return random.sample(population, k)
        else:
            return population.copy()

    def get(self, policy_rollout_buffer):
        num_chosen, num_rejected = 0, 0
        for data in policy_rollout_buffer.get(1):
            response = data.responses.tolist()[0]
            label = data.labels.tolist()[0]
            if self.evaluator.eval(response, label) is True:
                num_chosen += 1
            else:
                num_rejected += 1

        if num_chosen > num_rejected:
            return self.sample(self.rejected, num_chosen - num_rejected)
        else:
            return self.sample(self.chosen, num_rejected - num_chosen)


class RolloutPool:
    def __init__(self, task):
        self.datadict = dict()
        self.evaluator = get_evaluator(task)

    def put(self, policy_rollout_buffer):
        for data in policy_rollout_buffer.get(1):
            instruction = data.instructions.tolist()[0]
            response = data.responses.tolist()[0]
            label = data.labels.tolist()[0]
            if instruction not in self.datadict:
                self.datadict[instruction] = dict(chosen=[], rejected=[], label=data.labels.tolist()[0])
            if self.evaluator.eval(response, label) is True:
                self.datadict[instruction]["chosen"].append(response)
            else:
                self.datadict[instruction]["rejected"].append(response)

    def get(self, policy_rollout_buffer):
        instructions = set(policy_rollout_buffer["instructions"].tolist())  # deduplicate
        responses = set(policy_rollout_buffer["responses"].tolist())  # deduplicate

        results = []
        for instruction in instructions:
            if len(self.datadict[instruction]["chosen"]) > 0:
                response = random.choice(self.datadict[instruction]["chosen"])
                if response not in responses:
                    label = self.datadict[instruction]["label"]
                    results.append(dict(instruction=instruction, output=response, label=label))

        random.shuffle(results)
        print(f"Number of additional selected samples: {len(results)}")

        return results

    def save(self, log_dir):
        if get_rank() == 0:
            datalist = []
            for key in self.datadict.keys():
                datalist.append(dict(instruction=key, **self.datadict[key]))
            json_dump(datalist, os.path.join(log_dir, "rollout.jsonl"))


def rearrange_buffer(policy_rollout_buffer: RolloutBuffer, verifier_rollout_buffer: CriticRolloutBuffer):
    scores = []
    for data in verifier_rollout_buffer.get(1):
        scores.append(masked_mean(data.scores[0], data.action_masks[0]))
    scores = np.array(scores)
    positive_indices = np.where(scores > 0)[0]
    negative_indices = np.where(scores <= 0)[0]
    np.random.shuffle(positive_indices)
    np.random.shuffle(negative_indices)

    n_positive = len(positive_indices)
    n_negative = len(negative_indices)
    total_length = n_positive + n_negative
    total_indices = np.empty(total_length, dtype=int)

    if n_positive >= n_negative:
        larger = positive_indices
        smaller = negative_indices
        n_larger = n_positive
        n_smaller = n_negative
    else:
        larger = negative_indices
        smaller = positive_indices
        n_larger = n_negative
        n_smaller = n_positive

    step = n_larger / (n_smaller + 1)

    larger_idx = 0
    smaller_idx = 0
    for i in range(total_length):
        if i % (step + 1) < step:
            total_indices[i] = larger[larger_idx]
            larger_idx += 1
        else:
            total_indices[i] = smaller[smaller_idx]
            smaller_idx += 1

    policy_rollout_buffer.rearrange(total_indices)
    verifier_rollout_buffer.rearrange(total_indices)
    return policy_rollout_buffer, verifier_rollout_buffer


def run(
        task: str,
        train_file: str,
        log_dir: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        label_file: str = None,
        pool_file: str = None,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 24,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
        rho_pos: float = 1.2,
        rho_neg: float = 0.8,
        save_optim: bool = False,
        accumulation_steps: int = 1,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
):
    parallel_infos = setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        log_mode="w" if begin_epoch == 0 else "a",
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir
    pool = RolloutBalancePool(task=task)
    if pool_file:
        pool.load(pool_file)

    for epoch, datalist in IterationHandler(json_load(train_file), epochs, chunk_size, begin_epoch):
        dataset = JsonDataset(datalist)
        if len(dataset) == 0:
            continue

        # Collecting policy buffer
        policy_rollout_buffer = collect_actor_buffer_with_label(
            actor_model_type=policy_model_type,
            actor_config_file=policy_config_file,
            max_seq_len=max_seq_len,
            actor_tokenizer_file=policy_tokenizer_file,
            dtype=dtype,
            actor_ckpt_dir=policy_ckpt_dir,
            epoch=epoch,
            actor_save_dir=save_dir,
            use_chat_template=use_chat_template,
            dataset=dataset,
            max_generate_batch_size=max_generate_batch_size,
            temperature=temperature,
            top_p=top_p,
            num_samples_per_prompt=num_samples_per_prompt
        )

        verifier_rollout_buffer = collect_rule_based_verifier_buffer(policy_rollout_buffer, task)

        # Logging
        print(f"Original Average Rewards: {verifier_rollout_buffer.mean()}")

        # Read from pool
        datalist = pool.get(policy_rollout_buffer=policy_rollout_buffer)
        # Write to pool
        # pool.put(policy_rollout_buffer=policy_rollout_buffer)

        policy_rollout_buffer.extend(collect_actor_forward_buffer_with_label(
            actor_model_type=policy_model_type,
            actor_config_file=policy_config_file,
            max_seq_len=max_seq_len,
            actor_tokenizer_file=policy_tokenizer_file,
            dtype=dtype,
            actor_ckpt_dir=policy_ckpt_dir,
            epoch=epoch,
            actor_save_dir=save_dir,
            use_chat_template=False,
            dataset=JsonDataset(datalist),
            max_forward_batch_size=max_forward_batch_size
        ))

        verifier_rollout_buffer = collect_rule_based_verifier_buffer(
            actor_rollout_buffer=policy_rollout_buffer, task=task
        )
        print(f"Selected Average Rewards: {verifier_rollout_buffer.mean()}")
        # rearrange rollout buffer
        policy_rollout_buffer, verifier_rollout_buffer = rearrange_buffer(policy_rollout_buffer, verifier_rollout_buffer)

        rollout_buffer = RolloutBuffer(
            obs=policy_rollout_buffer["obs"],
            actions=policy_rollout_buffer["actions"],
            rewards=verifier_rollout_buffer["scores"],
            action_masks=policy_rollout_buffer["action_masks"],
            action_logprobs=policy_rollout_buffer["action_logprobs"]
        )

        train_policy_gradient_logits_convex(
            rollout_buffer=rollout_buffer,
            policy_ckpt_dir=policy_ckpt_dir,
            policy_model_type=policy_model_type,
            policy_config_file=policy_config_file,
            policy_tokenizer_file=policy_tokenizer_file,
            max_seq_len=max_seq_len,
            lora_rank=lora_rank,
            dtype=dtype,
            lora_dtype=lora_dtype,
            lr=lr,
            epoch=epoch,
            inner_epochs=inner_epochs,
            save_dir=save_dir,
            max_batch_size=max_batch_size,
            rho_pos=rho_pos,
            rho_neg=rho_neg,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps,
            shuffle=False
        )

        if parallel_infos.global_rank == 0:
            rollout_buffer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))

        if label_file is not None:
            evaluate_actor(
                task=task,
                label_file=label_file,
                log_dir=log_dir,
                actor_model_type=policy_model_type,
                actor_config_file=policy_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=policy_tokenizer_file,
                dtype=dtype,
                epoch=epoch,
                actor_save_dir=save_dir,
                max_generate_batch_size=max_generate_batch_size,
                use_chat_template=use_chat_template,
                # dataset=JsonDataset(json_load(label_file)[:max_generate_batch_size])
            )


if __name__ == '__main__':
    fire.Fire(run)
