import gc
import os
import random

import fire
import torch

from policy_train_policy_gradient_logits_convex import train_policy_gradient_logits_convex
from policy_train_ppo_with_evaluate import evaluate_actor
from policy_train_ppo_with_rule_rm import collect_rule_based_verifier_buffer
from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import IterationHandler, Timer
from src.modeling import get_parallel_model
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.ppo.buffer import RolloutBuffer
from src.ppo.collector import ActorPrefixBufferCollector
from src.utils import json_load, print_current_func_args


# class PrefixDataset(JsonDataset):
#     def __init__(self, f, ratio: float = 0.5):
#         super().__init__(f)
#         self.ratio = ratio
#         datalist = []
#         for data in self.datalist:
#             if len(data["output"]) == 0:
#                 continue
#             data = deepcopy(data)
#             data["original_output"] = data.pop("output")
#             datalist.append(data)
#         self.datalist = datalist
#
#     def __getitem__(self, i):
#         data = super().__getitem__(i)
#         if isinstance(data["original_output"], list):
#             data["original_output"] = random.choice(data["original_output"])
#         response_tokens = data["original_output"].split(" ")
#         data["prefix"] = " ".join(response_tokens[: int(self.ratio * len(response_tokens))])
#         return data


class PrefixDataset(JsonDataset):
    def __init__(self, f):
        super().__init__(f)
        self.first_words = [
            "Since", "Because", "To", "The", "We", "Given", "First", "Let's", "For", "Based", "According", "Now", "As", ""
        ]

    def __getitem__(self, i):
        data = super().__getitem__(i)
        data["prefix"] = random.choice(self.first_words)
        return data


def collect_actor_buffer_with_prefix(
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
        max_generate_batch_size: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
) -> RolloutBuffer:
    dataset.repeat(num_samples_per_prompt).shuffle()

    actor, actor_tokenizer = get_parallel_model(
        model_type=actor_model_type,
        config_file=actor_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=actor_tokenizer_file,
        lora_rank=-1,
        dtype=dtype
    )
    actor.load(actor_ckpt_dir if epoch == 0 else os.path.join(actor_save_dir, "epoch-%03d" % epoch))
    actor_buffer_collector = ActorPrefixBufferCollector(
        actor=actor,
        tokenizer=actor_tokenizer,
        max_seq_len=max_seq_len,
        temperature=temperature,
        top_p=top_p
    )
    actor_rollout_buffer_with_prefix = RolloutBuffer()
    print("Actor prefix buffer collecting ...")
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, actor_tokenizer)
    dataloader = ParallelDataLoader(dataset, batch_size=max_generate_batch_size)
    timer = Timer(len(dataloader))
    for data in dataloader:
        timer.step()
        actor_rollout_buffer = actor_buffer_collector.forward(data['instruction'], data['prefix'])
        actor_rollout_buffer_with_prefix.extend(RolloutBuffer(
            **actor_rollout_buffer, labels=data['label']
        ))
        print(data["instruction"][-1] + "\n" + data["prefix"][-1] + "\n---- Suffix ----\n" + actor_rollout_buffer["responses"][-1])

    actor.cpu()
    del actor
    del actor_buffer_collector
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    return actor_rollout_buffer_with_prefix


def run(
        task: str,
        train_file: str,
        log_dir: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        label_file: str = None,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        max_num_ckpts: int = None,
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

    for epoch, datalist in IterationHandler(json_load(train_file), epochs, chunk_size, begin_epoch):
        dataset = PrefixDataset(datalist)
        if len(dataset) == 0:
            continue

        policy_rollout_buffer = collect_actor_buffer_with_prefix(
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

        # also train on prefix
        policy_rollout_buffer["action_masks"] = policy_rollout_buffer["prefix_masks"] | policy_rollout_buffer["action_masks"]

        verifier_rollout_buffer = collect_rule_based_verifier_buffer(policy_rollout_buffer, task)
        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")

        rollout_buffer = RolloutBuffer(
            obs=policy_rollout_buffer["obs"],
            actions=policy_rollout_buffer["actions"],
            rewards=verifier_rollout_buffer["scores"],
            action_masks=policy_rollout_buffer["action_masks"],
            action_logprobs=policy_rollout_buffer["action_logprobs"],
            prefix_masks=policy_rollout_buffer["prefix_masks"]
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
            max_num_ckpts=max_num_ckpts
        )

        if parallel_infos.global_rank == 0:
            rollout_buffer.save(os.path.join(log_dir, "epoch-%03d" % (epoch + 1)))

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
            )


if __name__ == '__main__':
    fire.Fire(run)
