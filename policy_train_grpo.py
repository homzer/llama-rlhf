import gc
import os

import fire
import numpy as np
import torch

from policy_train_ppo import collect_actor_buffer, collect_reference_buffer, collect_verifier_buffer
from src.dataset import JsonDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.ppo.buffer import CriticRolloutBuffer, PolicyRolloutBuffer, RolloutBuffer
from src.ppo.parallel_buffer import ParallelRolloutBuffer
from src.ppo.trainer import ParallelGRPOTrainerForCausalLM
from src.utils import json_load


def compute_grpo_rewards(
        policy_rollout_buffer: RolloutBuffer,
        verifier_rollout_buffer: CriticRolloutBuffer,
        num_samples_per_prompt: int,
) -> (RolloutBuffer, CriticRolloutBuffer):
    policy_rollout_buffer = ParallelRolloutBuffer(**policy_rollout_buffer)
    policy_rollout_buffer.gather_from_data_parallel_region()
    verifier_rollout_buffer = ParallelRolloutBuffer(**verifier_rollout_buffer)
    verifier_rollout_buffer.gather_from_data_parallel_region()

    sorted_indices = sorted(range(policy_rollout_buffer.size()), key=lambda x: policy_rollout_buffer["instructions"][x])
    # sort policy rollout buffer
    policy_rollout_buffer.rearrange(sorted_indices)
    # sort verifier rollout buffer
    verifier_rollout_buffer.rearrange(sorted_indices)

    # check for arranged instructions
    for i in range(0, policy_rollout_buffer.size(), num_samples_per_prompt):
        assert len(set(policy_rollout_buffer["instructions"][i: i + num_samples_per_prompt].tolist())) == 1

    scores = []
    for i in range(verifier_rollout_buffer.size()):
        nonzero_indices = np.nonzero(verifier_rollout_buffer["action_masks"][i])[0]
        if len(nonzero_indices) > 0:
            scores.append(verifier_rollout_buffer["scores"][i][nonzero_indices[-1]])
        else:
            scores.append(0.0)
    scores = np.stack(scores, axis=0).reshape((-1, num_samples_per_prompt))
    scores_std = scores.std(axis=-1, keepdims=True)
    scores_std[scores_std == 0] = 1e-12
    scores = (scores - scores.mean(axis=-1, keepdims=True)) / scores_std
    scores = scores.reshape(-1).tolist()
    # update verifier rollout buffer
    verifier_rollout_buffer = ParallelRolloutBuffer(**CriticRolloutBuffer(
        scores=scores, action_masks=verifier_rollout_buffer["action_masks"]
    ))

    # shuffle
    randon_indices = np.random.permutation(policy_rollout_buffer.size())
    policy_rollout_buffer.rearrange(randon_indices)
    verifier_rollout_buffer.rearrange(randon_indices)

    policy_rollout_buffer.scatter_to_data_parallel_region()
    verifier_rollout_buffer.scatter_to_data_parallel_region()

    return policy_rollout_buffer, verifier_rollout_buffer


def train_grpo(
        policy_model_type: str,
        policy_config_file: str,
        max_seq_len: int,
        policy_tokenizer_file: str,
        lora_rank: int,
        dtype: str,
        lora_dtype: str,
        lr: float,
        epoch: int,
        policy_ckpt_dir: str,
        save_dir: str,
        rollout_buffer: PolicyRolloutBuffer,
        max_batch_size: int,
        inner_epochs: int,
        clip_range: float = 0.2,
        kl_coef: float = 0.04
):
    policy, policy_tokenizer = get_parallel_model(
        model_type=policy_model_type,
        config_file=policy_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=policy_tokenizer_file,
        lora_rank=lora_rank,
        dtype=dtype,
        lora_dtype=lora_dtype
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    trainer = ParallelGRPOTrainerForCausalLM(policy, optimizer, clip_range=clip_range, kl_coef=kl_coef)
    trainer.load_model(policy_ckpt_dir) if (
            epoch == 0
    ) else trainer.load(os.path.join(save_dir, "epoch-%03d" % epoch))
    print('Policy training ...')
    timer = Timer(total=(len(rollout_buffer) // max_batch_size) * inner_epochs, episode=100)
    for inner_epoch in range(inner_epochs):
        for data in rollout_buffer.get(max_batch_size):
            timer.step()
            trainer_outputs = trainer.forward(data)
            if trainer.step % 100 == 0:
                print(f'--------- STEP {trainer.step} OF {timer.total} ---------')
                print(f'Loss: {trainer_outputs.loss}')
                print(f'Policy Loss: {trainer_outputs.policy_loss}')
                print(f'Rewards: {trainer_outputs.rewards}')
                print(f'KL Loss: {trainer_outputs.kl_loss}')
    trainer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))

    policy.cpu()
    del policy
    del optimizer
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()


def run(
        train_file: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,
        reference_ckpt_dir: str = None,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        verifier_config_file: str = None,
        verifier_tokenizer_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 24,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 4,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-6,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        kl_coef: float = 0.04,
        clip_range: float = 0.2,
        use_chat_template: bool = False,
):
    setup_model_parallel()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir
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

            policy_rollout_buffer = collect_actor_buffer(
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

            reference_rollout_buffer = None
            if reference_ckpt_dir is not None:
                reference_rollout_buffer = collect_reference_buffer(
                    actor_model_type=policy_model_type,
                    actor_config_file=policy_config_file,
                    max_seq_len=max_seq_len,
                    actor_tokenizer_file=policy_tokenizer_file,
                    dtype=dtype,
                    reference_ckpt_dir=reference_ckpt_dir,
                    actor_rollout_buffer=policy_rollout_buffer,
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
                actor_rollout_buffer=policy_rollout_buffer,
                max_forward_batch_size=max_forward_batch_size
            )

            print(f"Average Rewards: {verifier_rollout_buffer.mean(use_last_token_reward=True)}")

            # Compute GRPO rewards
            policy_rollout_buffer, verifier_rollout_buffer = compute_grpo_rewards(
                verifier_rollout_buffer=verifier_rollout_buffer,
                policy_rollout_buffer=policy_rollout_buffer,
                num_samples_per_prompt=num_samples_per_prompt
            )

            rollout_buffer = PolicyRolloutBuffer(
                obs=policy_rollout_buffer["obs"],
                actions=policy_rollout_buffer["actions"],
                rewards=verifier_rollout_buffer["scores"],
                values=verifier_rollout_buffer["scores"],
                action_logits=policy_rollout_buffer["action_logits"],
                action_masks=policy_rollout_buffer["action_masks"],
                action_logprobs=policy_rollout_buffer["action_logprobs"],
                ref_action_logprobs=reference_rollout_buffer.output_tokens_logps if (
                    reference_rollout_buffer is not None
                ) else None,
                use_last_token_reward=True,
                last_token_reward_only=False,
                kl_coef=0.0,
                reward_normalize=False
            )

            train_grpo(
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
                kl_coef=kl_coef,
                clip_range=clip_range
            )


if __name__ == '__main__':
    fire.Fire(run)
