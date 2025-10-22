import gc
import os
import re

import fire
import numpy as np
import torch
from tqdm import tqdm

from policy_train_policy_gradient_norm_visualize import (
    ParallelPolicyGradientTrainerForCausalLM,
    ParallelGRPOTrainerForCausalLM,
    ParallelPolicyGradientConvexTrainerForCausalLM,
    ParallelPolicyGradientConvexBoundedTrainerForCausalLM
)
from policy_train_ppo_with_rule_rm import collect_actor_buffer_with_label, collect_rule_based_verifier_buffer
from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer, IterationHandler
from src.evaluator import DataParallelPolicyEvaluator
from src.modeling import get_parallel_model
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.parallel.optimizer import ParallelOptimizer
from src.ppo.buffer import PPORolloutBuffer, RolloutBuffer, CriticRolloutBuffer
from src.ppo.collector import ActorForwardBufferCollector
from src.utils import json_load, print_current_func_args, json_dump, masked_mean


def evaluate_policy(task, policy, policy_tokenizer, label_file, use_chat_template, max_generate_batch_size,
                    max_seq_len) -> float:
    print("Actor Evaluating ...")
    label_dataset = JsonDataset(label_file)
    if use_chat_template:
        label_dataset = ChatTemplateDataset(label_dataset, policy_tokenizer)
    evaluator = DataParallelPolicyEvaluator(
        model=policy,
        tokenizer=policy_tokenizer,
        batch_size=max_generate_batch_size,
        max_seq_len=max_seq_len
    )
    evaluator_outputs = evaluator.forward(task=task, dataset=label_dataset)
    print(f"{task.upper()} Evaluate Accuracy: {evaluator_outputs.acc}")

    del evaluator

    return evaluator_outputs.acc


def run(
        task: str,
        train_file: str,
        log_dir: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        label_file: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 5,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
        train_strategy: str = "vanilla",
        reward_sub_mean: bool = False,
        clip_range: float = 0.2,
        delta: float = 0.1,
        rho_pos: float = 1.8,
        rho_neg: float = 0.9,
        eval_steps: int = 100,
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

    results = []
    for epoch, datalist in IterationHandler(json_load(train_file), epochs, chunk_size, begin_epoch):
        if epoch >= 3:  # TODO
            break

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

        verifier_rollout_buffer = collect_rule_based_verifier_buffer(
            actor_rollout_buffer=policy_rollout_buffer, task=task
        )

        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")

        rollout_buffer = PPORolloutBuffer(
            obs=policy_rollout_buffer["obs"],
            actions=policy_rollout_buffer["actions"],
            rewards=verifier_rollout_buffer["scores"],
            values=verifier_rollout_buffer["scores"],  # pseudo
            action_logits=policy_rollout_buffer["action_logits"],
            action_masks=policy_rollout_buffer["action_masks"],
            action_logprobs=policy_rollout_buffer["action_logprobs"],
            use_last_token_reward=True,
            reward_sub_mean=reward_sub_mean
        )

        policy, policy_tokenizer = get_parallel_model(
            model_type=policy_model_type,
            config_file=policy_config_file,
            max_seq_len=max_seq_len,
            tokenizer_file=policy_tokenizer_file,
            dtype=dtype
        )
        policy.load(policy_ckpt_dir if epoch == 0 else os.path.join(save_dir, "epoch-%03d" % epoch))
        policy.train()
        optimizer = ParallelOptimizer(torch.optim.Adam(policy.parameters(), lr=lr))
        if train_strategy == "vanilla":
            print("Using ParallelPolicyGradientTrainerForCausalLM")
            trainer = ParallelPolicyGradientTrainerForCausalLM(policy, optimizer)
        elif train_strategy == "ratio":
            print("Using ParallelGRPOTrainerForCausalLM")
            trainer = ParallelGRPOTrainerForCausalLM(policy, optimizer, clip_range=clip_range)
        elif train_strategy == "convex":
            print("Using ParallelPolicyGradientConvexTrainerForCausalLM")
            trainer = ParallelPolicyGradientConvexTrainerForCausalLM(policy, optimizer, delta=delta)
        elif train_strategy == "convex-bounded":
            print("Using ParallelPolicyGradientConvexBoundedTrainerForCausalLM")
            trainer = ParallelPolicyGradientConvexBoundedTrainerForCausalLM(
                policy, optimizer, rho_pos=rho_pos, rho_neg=rho_neg)
        else:
            raise ValueError(train_strategy)

        print('Policy training ...')
        timer = Timer(total=(len(rollout_buffer) // max_batch_size) * inner_epochs, episode=10)
        for inner_epoch in range(inner_epochs):
            for data in rollout_buffer.get(max_batch_size):
                timer.step()
                outputs = trainer.forward(data)
                results.append(dict(
                    gradient=outputs.gradient,
                    pos_gradient=outputs.pos_gradient,
                    neg_gradient=outputs.neg_gradient,
                    action_probs=outputs.action_probs,
                    entropy=outputs.entropy,
                ))
                if trainer.step % 10 == 0:
                    print(f'--------- STEP {trainer.step} OF {timer.total} ---------')
                    print(f'Gradient: {outputs.gradient}')
                    print(f'Positive Gradient: {outputs.pos_gradient}')
                    print(f'Negative Gradient: {outputs.neg_gradient}')
                    print(f'Action Probs: {outputs.action_probs}')
                    print(f'Entropy: {outputs.entropy}')
                if trainer.step % eval_steps == 0:
                    accuracy = evaluate_policy(
                        task=task,
                        policy=policy,
                        policy_tokenizer=policy_tokenizer,
                        label_file=label_file,
                        use_chat_template=use_chat_template,
                        max_seq_len=max_seq_len,
                        max_generate_batch_size=max_generate_batch_size
                    )
                    results[-1]["accuracy"] = accuracy
                else:
                    results[-1]["accuracy"] = -1

        if parallel_infos.global_rank == 0:
            os.makedirs(log_dir, exist_ok=True)
            json_dump(results, os.path.join(log_dir, "grad.jsonl"))

        trainer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))

        policy.cpu()
        del policy
        del optimizer
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()


# statistic for average token probabilities on sft data
def run_for_probs(
        train_file: str,
        log_dir: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        max_forward_batch_size: int = 48,
        max_seq_len: int = 1024,
        save_step: int = 10000,
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
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

    policy, policy_tokenizer = get_parallel_model(
        model_type=policy_model_type,
        config_file=policy_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=policy_tokenizer_file
    )
    policy.load(policy_ckpt_dir)
    policy_buffer_collector = ActorForwardBufferCollector(
        actor=policy,
        tokenizer=policy_tokenizer,
        max_seq_len=max_seq_len
    )
    policy_rollout_buffer = RolloutBuffer()

    dataset = JsonDataset(json_load(train_file))
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, policy_tokenizer)
    dataloader = ParallelDataLoader(dataset, batch_size=max_forward_batch_size)
    timer = Timer(len(dataloader), episode=100)
    if parallel_infos.global_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    for data in dataloader:
        timer.step()
        policy_rollout_buffer.extend(
            policy_buffer_collector.forward(data["instruction"], data['output'])
        )
        if timer.ticktock % save_step == 0 and parallel_infos.global_rank == 0:
            policy_rollout_buffer.save(save_dir)

    if parallel_infos.global_rank == 0:
        policy_rollout_buffer.save(save_dir)


def filter_probs(file: str, threshold: float = 0.7):
    def extract_instruction(s: str) -> str:
        match = re.search(r"<\|im_start\|>system\nYou are a helpful assistant.<\|im_end\|>\n<\|im_start\|>user\n(.*)<|im_end|>\n<\|im_start\|>assistant\n", s, re.DOTALL)
        if match:
            return match.group(1)
        return s

    datalist = json_load(file)
    results = []
    results_action_probs = []
    for data in tqdm(datalist):
        action_logprobs = np.array(data["action_logprobs"])[np.array(data["action_masks"])]
        action_probs = np.exp(action_logprobs).mean().item()
        if action_probs < threshold:
            results.append(dict(instruction=extract_instruction(data["instructions"]), output=data["responses"]))
        results_action_probs.append(action_probs)
    print(len(results))
    json_dump(results, "../../data/results/prm800k-aime-gsm8k/train/sft/results-low-probs.jsonl")

    # import matplotlib.pyplot as plt
    #
    # plt.hist(results_action_probs, bins=100)
    # plt.savefig("probs_distribution.pdf")


def collect_actor_forward_buffer(
        actor_model_type: str,
        actor_config_file: str,
        max_seq_len: int,
        actor_tokenizer_file: str,
        dtype: str,
        actor_ckpt_dir: str,
        epoch: int,
        actor_save_dir: str,
        use_chat_template: bool,
        dataset: JsonDataset,
        max_forward_batch_size: int
) -> RolloutBuffer:
    if len(dataset) == 0:
        return RolloutBuffer()
    actor, actor_tokenizer = get_parallel_model(
        model_type=actor_model_type,
        config_file=actor_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=actor_tokenizer_file,
        lora_rank=-1,
        dtype=dtype
    )
    actor.load(actor_ckpt_dir if epoch == 0 else os.path.join(actor_save_dir, "epoch-%03d" % epoch))
    actor_buffer_collector = ActorForwardBufferCollector(
        actor=actor,
        tokenizer=actor_tokenizer,
        max_seq_len=max_seq_len
    )
    actor_rollout_buffer = RolloutBuffer()
    print("Actor anchor buffer collecting ...")
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, actor_tokenizer)
    dataloader = ParallelDataLoader(dataset, batch_size=max_forward_batch_size)
    timer = Timer(len(dataloader), episode=10)
    for data in dataloader:
        timer.step()
        actor_rollout_buffer.extend(actor_buffer_collector.forward(data["instruction"], data["output"]))

    actor.cpu()
    del actor
    del actor_buffer_collector
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    return actor_rollout_buffer


def run_forward(
        task: str,
        train_file: str,
        log_dir: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        label_file: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 12,
        max_seq_len: int = 1024,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 5,
        lr: float = 1e-6,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
        reward_sub_mean: bool = False,
        clip_range: float = 0.2,
        eval_steps: int = 100,
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

    results = []
    for epoch, datalist in IterationHandler(json_load(train_file), epochs, chunk_size, begin_epoch):
        dataset = JsonDataset(datalist)
        if len(dataset) == 0:
            continue

        # Collecting policy buffer
        policy_rollout_buffer = collect_actor_forward_buffer(
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
            max_forward_batch_size=max_forward_batch_size
        )

        verifier_rollout_buffer = CriticRolloutBuffer(
            scores=[1.] * policy_rollout_buffer.size(),
            action_masks=policy_rollout_buffer["action_masks"],
            use_last_token_reward=True,
        )

        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")

        rollout_buffer = PPORolloutBuffer(
            obs=policy_rollout_buffer["obs"],
            actions=policy_rollout_buffer["actions"],
            rewards=verifier_rollout_buffer["scores"],
            values=verifier_rollout_buffer["scores"],  # pseudo
            action_logits=policy_rollout_buffer["action_logits"],
            action_masks=policy_rollout_buffer["action_masks"],
            action_logprobs=policy_rollout_buffer["action_logprobs"],
            use_last_token_reward=True,
            reward_sub_mean=reward_sub_mean,
            reward_normalize=False
        )

        policy, policy_tokenizer = get_parallel_model(
            model_type=policy_model_type,
            config_file=policy_config_file,
            max_seq_len=max_seq_len,
            tokenizer_file=policy_tokenizer_file,
            dtype=dtype
        )
        policy.load(policy_ckpt_dir if epoch == 0 else os.path.join(save_dir, "epoch-%03d" % epoch))

        policy.train()
        optimizer = ParallelOptimizer(torch.optim.Adam(policy.parameters(), lr=lr))
        trainer = ParallelGRPOTrainerForCausalLM(policy, optimizer, clip_range=clip_range)

        print('Policy training ...')
        timer = Timer(total=(len(rollout_buffer) // max_batch_size) * inner_epochs, episode=10)
        for inner_epoch in range(inner_epochs):
            for data in rollout_buffer.get(max_batch_size):
                timer.step()
                outputs = trainer.forward(data)
                results.append(dict(
                    gradient=outputs.gradient,
                    pos_gradient=outputs.pos_gradient,
                    neg_gradient=outputs.neg_gradient,
                    action_probs=outputs.action_probs,
                    entropy=outputs.entropy,
                ))
                if trainer.step % 10 == 0:
                    print(f'--------- STEP {trainer.step} OF {timer.total} ---------')
                    print(f'Gradient: {outputs.gradient}')
                    print(f'Positive Gradient: {outputs.pos_gradient}')
                    print(f'Negative Gradient: {outputs.neg_gradient}')
                    print(f'Action Probs: {outputs.action_probs}')
                    print(f'Entropy: {outputs.entropy}')
                if trainer.step % eval_steps == 0:
                    accuracy = evaluate_policy(
                        task=task,
                        policy=policy,
                        policy_tokenizer=policy_tokenizer,
                        label_file=label_file,
                        use_chat_template=use_chat_template,
                        max_seq_len=max_seq_len,
                        max_generate_batch_size=max_generate_batch_size
                    )
                    results[-1]["accuracy"] = accuracy
                else:
                    results[-1]["accuracy"] = -1

        if parallel_infos.global_rank == 0:
            os.makedirs(log_dir, exist_ok=True)
            json_dump(results, os.path.join(log_dir, "grad.jsonl"))

        break


def filter_for_low_probs_pos_buffer(
        policy_rollout_buffer: RolloutBuffer,
        verifier_rollout_buffer: CriticRolloutBuffer,
        num_of_samples: int = 3072
) -> (RolloutBuffer, CriticRolloutBuffer):
    sequence_scores = masked_mean(verifier_rollout_buffer["scores"], policy_rollout_buffer["action_masks"], dim=-1)
    positive_indices = np.arange(len(sequence_scores))[sequence_scores > 0]
    print(f"Number of Positive Samples: {len(positive_indices)}")
    verifier_rollout_buffer.rearrange(positive_indices)
    policy_rollout_buffer.rearrange(positive_indices)

    action_probs = np.exp(policy_rollout_buffer["action_logprobs"])
    sequence_probs = masked_mean(action_probs, policy_rollout_buffer["action_masks"], dim=-1)
    print(f"Average Sequence Probs: {np.mean(sequence_probs)}")
    sort_indices = np.argsort(sequence_probs)
    sort_indices = sort_indices[:num_of_samples]
    print(f"Number of Final Selecting Samples: {len(sort_indices)}")
    verifier_rollout_buffer.rearrange(sort_indices)
    policy_rollout_buffer.rearrange(sort_indices)
    print(f"Average Sequence Probs: {np.mean(sequence_probs[sort_indices])}")

    return policy_rollout_buffer, verifier_rollout_buffer


def run_for_low_probs(
        task: str,
        train_file: str,
        log_dir: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        label_file: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
        num_of_samples: int = 3072,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 5,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
        clip_range: float = 0.2,
        eval_steps: int = 100,
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

    results = []
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

        verifier_rollout_buffer = collect_rule_based_verifier_buffer(
            actor_rollout_buffer=policy_rollout_buffer, task=task
        )

        policy_rollout_buffer, verifier_rollout_buffer = filter_for_low_probs_pos_buffer(
            policy_rollout_buffer=policy_rollout_buffer,
            verifier_rollout_buffer=verifier_rollout_buffer,
            num_of_samples=num_of_samples
        )

        if parallel_infos.global_rank == 0:
            policy_rollout_buffer.save(log_dir)

        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")

        rollout_buffer = PPORolloutBuffer(
            obs=policy_rollout_buffer["obs"],
            actions=policy_rollout_buffer["actions"],
            rewards=verifier_rollout_buffer["scores"],
            values=verifier_rollout_buffer["scores"],  # pseudo
            action_logits=policy_rollout_buffer["action_logits"],
            action_masks=policy_rollout_buffer["action_masks"],
            action_logprobs=policy_rollout_buffer["action_logprobs"],
            use_last_token_reward=True,
            reward_normalize=False
        )

        policy, policy_tokenizer = get_parallel_model(
            model_type=policy_model_type,
            config_file=policy_config_file,
            max_seq_len=max_seq_len,
            tokenizer_file=policy_tokenizer_file,
            dtype=dtype
        )
        policy.load(policy_ckpt_dir if epoch == 0 else os.path.join(save_dir, "epoch-%03d" % epoch))
        policy.train()
        optimizer = ParallelOptimizer(torch.optim.Adam(policy.parameters(), lr=lr))
        trainer = ParallelGRPOTrainerForCausalLM(policy, optimizer, clip_range=clip_range)

        print('Policy training ...')
        timer = Timer(total=(len(rollout_buffer) // max_batch_size) * inner_epochs, episode=10)
        for inner_epoch in range(inner_epochs):
            for data in rollout_buffer.get(max_batch_size):
                timer.step()
                outputs = trainer.forward(data)
                results.append(dict(
                    gradient=outputs.gradient,
                    pos_gradient=outputs.pos_gradient,
                    neg_gradient=outputs.neg_gradient,
                    action_probs=outputs.action_probs,
                    entropy=outputs.entropy,
                ))
                if trainer.step % 10 == 0:
                    print(f'--------- STEP {trainer.step} OF {timer.total} ---------')
                    print(f'Gradient: {outputs.gradient}')
                    print(f'Positive Gradient: {outputs.pos_gradient}')
                    print(f'Negative Gradient: {outputs.neg_gradient}')
                    print(f'Action Probs: {outputs.action_probs}')
                    print(f'Entropy: {outputs.entropy}')
                if trainer.step % eval_steps == 0:
                    accuracy = evaluate_policy(
                        task=task,
                        policy=policy,
                        policy_tokenizer=policy_tokenizer,
                        label_file=label_file,
                        use_chat_template=use_chat_template,
                        max_seq_len=max_seq_len,
                        max_generate_batch_size=max_generate_batch_size
                    )
                    results[-1]["accuracy"] = accuracy
                else:
                    results[-1]["accuracy"] = -1

        if parallel_infos.global_rank == 0:
            os.makedirs(log_dir, exist_ok=True)
            json_dump(results, os.path.join(log_dir, "grad.jsonl"))

        trainer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))

        policy.cpu()
        del policy
        del optimizer
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        break


if __name__ == '__main__':
    fire.Fire()
