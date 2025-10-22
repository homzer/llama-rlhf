import gc
import os

import fire
import torch

from policy_train_ppo import (
    collect_reference_buffer,
    collect_critic_buffer,
    collect_verifier_buffer,
    train_actor,
    train_critic,
    collect_actor_buffer
)
from src.dataset import JsonDataset, ChatTemplateDataset
from src.evaluator import DataParallelPolicyEvaluator
from src.modeling import get_parallel_model
from src.parallel.initialize import setup_model_parallel, set_barrier, get_rank
from src.ppo.buffer import PPORolloutBuffer
from src.utils import json_load, json_dump


def evaluate_actor(
        label_file: str,
        actor_model_type: str,
        actor_config_file: str,
        max_seq_len: int,
        actor_tokenizer_file: str,
        dtype: str,
        epoch: int,
        actor_save_dir: str,
        max_generate_batch_size: int,
        use_chat_template: bool,
        task: str = None,
        log_dir: str = None,
        dataset: JsonDataset = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
):
    if task is None:
        return
    actor, actor_tokenizer = get_parallel_model(
        model_type=actor_model_type,
        config_file=actor_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=actor_tokenizer_file,
        dtype=dtype,
    )
    actor.load(os.path.join(actor_save_dir, "epoch-%03d" % (epoch + 1)))
    print("Actor Evaluating ...")
    if dataset is None:
        dataset = JsonDataset(label_file)
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, actor_tokenizer)
    evaluator = DataParallelPolicyEvaluator(
        model=actor,
        tokenizer=actor_tokenizer,
        batch_size=max_generate_batch_size,
        max_seq_len=max_seq_len,
        temperature=temperature,
        top_p=top_p
    )
    evaluator_outputs = evaluator.forward(task=task, dataset=dataset)
    print(f"{task.upper()} Evaluate Accuracy: {evaluator_outputs.acc} | Missing: {evaluator_outputs.missing}")
    if log_dir is not None and get_rank() == 0:
        os.makedirs(os.path.join(log_dir, "epoch-%03d" % (epoch + 1), task), exist_ok=True)
        json_dump(evaluator_outputs.datalist, os.path.join(
            log_dir, "epoch-%03d" % (epoch + 1), task, f'results-{round(evaluator_outputs.acc, 4)}.jsonl'))

    actor.cpu()
    del actor
    del evaluator
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()


def run(
        task: str,
        label_file: str,
        train_file: str,
        log_dir: str,
        actor_ckpt_dir: str,
        actor_model_type: str,
        actor_save_dir: str,
        critic_ckpt_dir: str,
        critic_model_type: str,
        critic_save_dir: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,
        actor_config_file: str = None,
        actor_tokenizer_file: str = None,
        critic_config_file: str = None,
        critic_tokenizer_file: str = None,
        verifier_config_file: str = None,
        verifier_tokenizer_file: str = None,
        reference_ckpt_dir: str = None,
        actor_lora_rank: int = -1,
        actor_lora_dtype: str = "bfloat16",
        critic_lora_rank: int = -1,
        critic_lora_dtype: str = "bfloat16",
        actor_max_batch_size: int = 1,
        critic_max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 24,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-6,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        kl_coef: float = 0.1,
        clip_range: float = 0.2,
        gamma: float = 0.9,
        gae_lambda: float = 0.8,
        use_chat_template: bool = False,
        use_last_token_reward: bool = False,
        last_token_reward_only: bool = False,
        reward_is_q: bool = False
):
    parallel_infos = setup_model_parallel(log_dir=log_dir)
    actor_config_file = actor_config_file or actor_ckpt_dir
    actor_tokenizer_file = actor_tokenizer_file or actor_ckpt_dir
    critic_config_file = critic_config_file or critic_ckpt_dir
    critic_tokenizer_file = critic_tokenizer_file or critic_ckpt_dir
    verifier_config_file = verifier_config_file or verifier_ckpt_dir
    verifier_tokenizer_file = verifier_tokenizer_file or verifier_ckpt_dir

    datalist = json_load(train_file)
    chunk_size = chunk_size or len(datalist)
    local_epochs = len(datalist) // chunk_size
    begin_global_epoch = begin_epoch // local_epochs
    begin_local_epoch = begin_epoch % local_epochs
    for global_epoch in range(begin_global_epoch, epochs):
        for local_epoch in range(begin_local_epoch, local_epochs):
            epoch = local_epoch + global_epoch * local_epochs
            print(f"Epoch - {epoch} of {local_epochs * epochs}")
            dataset = JsonDataset(f=datalist[local_epoch * chunk_size: (local_epoch + 1) * chunk_size])
            if len(dataset) == 0:
                continue

            # Collecting actor buffer
            actor_rollout_buffer = collect_actor_buffer(
                actor_model_type=actor_model_type,
                actor_config_file=actor_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=actor_tokenizer_file,
                dtype=dtype,
                actor_ckpt_dir=actor_ckpt_dir,
                epoch=epoch,
                actor_save_dir=actor_save_dir,
                use_chat_template=use_chat_template,
                dataset=dataset,
                max_generate_batch_size=max_generate_batch_size,
                temperature=temperature,
                top_p=top_p,
                num_samples_per_prompt=num_samples_per_prompt
            )

            reference_rollout_buffer = None
            if reference_ckpt_dir is not None and kl_coef != 0:
                # Collecting reference logprobs
                reference_rollout_buffer = collect_reference_buffer(
                    actor_model_type=actor_model_type,
                    actor_config_file=actor_config_file,
                    max_seq_len=max_seq_len,
                    actor_tokenizer_file=actor_tokenizer_file,
                    dtype=dtype,
                    reference_ckpt_dir=reference_ckpt_dir,
                    actor_rollout_buffer=actor_rollout_buffer,
                    max_forward_batch_size=max_forward_batch_size
                )

            # Collecting critic buffer
            critic_rollout_buffer = collect_critic_buffer(
                critic_model_type=critic_model_type,
                critic_config_file=critic_config_file,
                max_seq_len=max_seq_len,
                critic_tokenizer_file=critic_tokenizer_file,
                dtype=dtype,
                critic_ckpt_dir=critic_ckpt_dir,
                epoch=epoch,
                critic_save_dir=critic_save_dir,
                actor_rollout_buffer=actor_rollout_buffer,
                max_forward_batch_size=max_forward_batch_size
            )

            # Collecting verifier buffer
            verifier_rollout_buffer = collect_verifier_buffer(
                verifier_model_type=verifier_model_type,
                verifier_config_file=verifier_config_file,
                max_seq_len=max_seq_len,
                verifier_tokenizer_file=verifier_tokenizer_file,
                dtype=dtype,
                verifier_ckpt_dir=verifier_ckpt_dir,
                actor_rollout_buffer=actor_rollout_buffer,
                max_forward_batch_size=max_forward_batch_size
            )

            print(f"Average Rewards: {verifier_rollout_buffer.mean()}")

            rollout_buffer = PPORolloutBuffer(
                obs=actor_rollout_buffer["obs"],
                actions=actor_rollout_buffer["actions"],
                rewards=verifier_rollout_buffer["scores"],
                values=critic_rollout_buffer["scores"],
                action_logits=actor_rollout_buffer["action_logits"],
                action_masks=actor_rollout_buffer["action_masks"],
                action_logprobs=actor_rollout_buffer["action_logprobs"],
                ref_action_logprobs=reference_rollout_buffer.output_tokens_logps if (
                        reference_rollout_buffer is not None
                ) else None,
                use_last_token_reward=use_last_token_reward,
                last_token_reward_only=last_token_reward_only,
                reward_is_q=reward_is_q,  # Reward is Q-Value
                kl_coef=kl_coef,
                gamma=gamma,
                gae_lambda=gae_lambda
            )

            # Actor training
            train_actor(
                actor_model_type=actor_model_type,
                actor_config_file=actor_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=actor_tokenizer_file,
                actor_lora_rank=actor_lora_rank,
                dtype=dtype,
                actor_lora_dtype=actor_lora_dtype,
                lr=lr,
                epoch=epoch,
                actor_ckpt_dir=actor_ckpt_dir,
                actor_save_dir=actor_save_dir,
                rollout_buffer=rollout_buffer,
                actor_max_batch_size=actor_max_batch_size,
                inner_epochs=inner_epochs,
                clip_range=clip_range
            )

            if parallel_infos.local_rank == 0:
                torch.save({
                    'obs': rollout_buffer.obs,
                    'actions': rollout_buffer.actions,
                    'values': rollout_buffer.origin_values,
                    'rewards': rollout_buffer.origin_rewards,
                    'action_masks': rollout_buffer.action_masks,
                    'advantages': rollout_buffer.advantages,
                    'returns': rollout_buffer.returns
                }, os.path.join(actor_save_dir, "epoch-%03d" % (epoch + 1), "buffer.bin"))

            evaluate_actor(
                task=task,
                label_file=label_file,
                log_dir=log_dir,
                actor_model_type=actor_model_type,
                actor_config_file=actor_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=actor_tokenizer_file,
                dtype=dtype,
                epoch=epoch,
                actor_save_dir=actor_save_dir,
                max_generate_batch_size=max_generate_batch_size,
                use_chat_template=use_chat_template
            )

            train_critic(
                critic_model_type=critic_model_type,
                critic_config_file=critic_config_file,
                max_seq_len=max_seq_len,
                critic_tokenizer_file=critic_tokenizer_file,
                critic_lora_rank=critic_lora_rank,
                dtype=dtype,
                lr=lr,
                critic_lora_dtype=critic_lora_dtype,
                critic_ckpt_dir=critic_ckpt_dir,
                epoch=epoch,
                critic_save_dir=critic_save_dir,
                rollout_buffer=rollout_buffer,
                critic_max_batch_size=critic_max_batch_size,
                inner_epochs=inner_epochs
            )


if __name__ == '__main__':
    fire.Fire(run)
