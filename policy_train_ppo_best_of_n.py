import os

import fire
import numpy as np
import torch

from policy_train_ppo import (
    train_critic,
    train_actor,
    collect_actor_buffer,
    collect_verifier_buffer,
    collect_critic_buffer,
    collect_reference_buffer,
)
from src.dataset import JsonDataset
from src.parallel.utils import setup_model_parallel
from src.ppo.buffer import RolloutBuffer, ActorRolloutBuffer, CriticRolloutBuffer
from src.utils import masked_mean, json_load


def select_best_of_n_buffer(
        actor_rollout_buffer: ActorRolloutBuffer,
        verifier_rollout_buffer: CriticRolloutBuffer,
        num_samples_per_prompt: int,
        num_samples_keep_per_prompt: int,
        use_last_token_reward: bool
) -> (ActorRolloutBuffer, CriticRolloutBuffer):
    num_samples_keep_per_prompt = min(num_samples_per_prompt, num_samples_keep_per_prompt)
    if use_last_token_reward:
        scores = []
        for i in range(len(verifier_rollout_buffer)):
            nonzero_indices = np.nonzero(actor_rollout_buffer.action_masks[i])[0]
            if len(nonzero_indices) > 0:
                scores.append(verifier_rollout_buffer.scores[i][nonzero_indices[-1]])
            else:
                scores.append(0.0)
        scores = np.stack(scores, axis=0)
    else:
        scores = masked_mean(verifier_rollout_buffer.scores, actor_rollout_buffer.action_masks, dim=-1)
    scores_values, scores_indices = torch.topk(
        torch.from_numpy(scores).reshape(-1, num_samples_per_prompt),  # numpy does not support topk()
        k=num_samples_keep_per_prompt,
    )
    scores_indices += (torch.arange(0, len(scores_indices)) * num_samples_per_prompt).unsqueeze(-1)
    scores_indices = scores_indices.reshape(-1).tolist()

    # update actor rollout buffer
    actor_rollout_buffer.instructions = actor_rollout_buffer.instructions[scores_indices]
    actor_rollout_buffer.obs = actor_rollout_buffer.obs[scores_indices]
    actor_rollout_buffer.actions = actor_rollout_buffer.actions[scores_indices]
    actor_rollout_buffer.action_logits = actor_rollout_buffer.action_logits[scores_indices]
    actor_rollout_buffer.action_masks = actor_rollout_buffer.action_masks[scores_indices]
    actor_rollout_buffer.action_logprobs = actor_rollout_buffer.action_logprobs[scores_indices]
    actor_rollout_buffer.responses = actor_rollout_buffer.responses[scores_indices]
    # update verifier rollout buffer
    verifier_rollout_buffer.scores = verifier_rollout_buffer.scores[scores_indices]
    return actor_rollout_buffer, verifier_rollout_buffer


def run(
        train_file: str,
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
        num_samples_per_prompt: int = 5,
        num_samples_keep_per_prompt: int = 1,
        actor_lora_rank: int = -1,
        actor_lora_dtype: str = "bfloat16",
        critic_lora_rank: int = -1,
        critic_lora_dtype: str = "bfloat16",
        actor_max_batch_size: int = 1,
        critic_max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 24,
        max_seq_len: int = 1024,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-6,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        kl_coef: float = 0.1,
        clip_range: float = 0.2,
        use_chat_template: bool = False,
        use_last_token_reward: bool = False
):
    setup_model_parallel()
    actor_config_file = actor_config_file or actor_ckpt_dir
    actor_tokenizer_file = actor_tokenizer_file or actor_ckpt_dir
    critic_config_file = critic_config_file or critic_ckpt_dir
    critic_tokenizer_file = critic_tokenizer_file or critic_ckpt_dir
    verifier_config_file = verifier_config_file or verifier_ckpt_dir
    verifier_tokenizer_file = verifier_tokenizer_file or verifier_ckpt_dir

    datalist = json_load(train_file)
    chunk_size = chunk_size or len(datalist)
    epochs = len(datalist) // chunk_size
    for epoch in range(begin_epoch, epochs):
        print(f"Epoch - {epoch} of {epochs}")
        dataset = JsonDataset(f=datalist[epoch * chunk_size: (epoch + 1) * chunk_size])

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

        actor_rollout_buffer, verifier_rollout_buffer = select_best_of_n_buffer(
            actor_rollout_buffer=actor_rollout_buffer,
            verifier_rollout_buffer=verifier_rollout_buffer,
            num_samples_per_prompt=num_samples_per_prompt,
            num_samples_keep_per_prompt=num_samples_keep_per_prompt,
            use_last_token_reward=use_last_token_reward
        )

        reference_rollout_buffer = None
        if reference_ckpt_dir is not None:
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

        if use_last_token_reward:
            rewards = []
            for i in range(len(verifier_rollout_buffer)):
                last_idx = np.nonzero(actor_rollout_buffer.action_masks[i])[0][-1].item()
                rewards.append(verifier_rollout_buffer.scores[i][last_idx])
            print("Average Rewards: ", np.mean(rewards))
        else:
            print("Average Rewards: ", masked_mean(verifier_rollout_buffer.scores, actor_rollout_buffer.action_masks))

        rollout_buffer = RolloutBuffer(
            obs=actor_rollout_buffer.obs,
            actions=actor_rollout_buffer.actions,
            rewards=verifier_rollout_buffer.scores,
            values=critic_rollout_buffer.scores,
            action_logits=actor_rollout_buffer.action_logits,
            action_masks=actor_rollout_buffer.action_masks,
            action_logprobs=actor_rollout_buffer.action_logprobs,
            ref_action_logprobs=reference_rollout_buffer.output_tokens_logps if (
                    reference_rollout_buffer is not None
            ) else None,
            use_last_token_reward=use_last_token_reward,
            kl_coef=kl_coef
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

        torch.save({
            'obs': rollout_buffer.obs[: max_forward_batch_size],
            'actions': rollout_buffer.actions[: max_forward_batch_size],
            'values': rollout_buffer.values[: max_forward_batch_size],
            'rewards': rollout_buffer.rewards[: max_forward_batch_size],
            'action_masks': rollout_buffer.action_masks[: max_forward_batch_size],
            'advantages': rollout_buffer.advantages[: max_forward_batch_size],
            'returns': rollout_buffer.returns[: max_forward_batch_size]
        }, os.path.join(actor_save_dir, f"epoch-{epoch + 1}", "buffer.bin"))

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
