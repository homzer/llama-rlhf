import fire
import numpy as np

from policy_train_policy_gradient import train_policy_gradient
from policy_train_ppo import collect_actor_buffer, collect_verifier_buffer
from src.dataset import JsonDataset
from src.parallel.initialize import setup_model_parallel
from src.ppo.buffer import CriticRolloutBuffer, PolicyRolloutBuffer, RolloutBuffer
from src.utils import json_load


def compute_rloo_rewards(
        verifier_rollout_buffer: CriticRolloutBuffer,
        policy_rollout_buffer: RolloutBuffer,
        num_samples_per_prompt: int,
) -> CriticRolloutBuffer:  # TODO: rollout buffer
    """ Normalize EOS token reward across batch. And Set rewards of the output tokens to the normalized rewards. """
    assert len(verifier_rollout_buffer) % num_samples_per_prompt == 0
    for i in range(0, len(verifier_rollout_buffer), num_samples_per_prompt):
        assert all(
            instruction == policy_rollout_buffer.instructions[i] for
            instruction in policy_rollout_buffer.instructions[i + 1: i + num_samples_per_prompt]
        )

        # compute baseline reward
        baseline = []
        for j in range(i, i + num_samples_per_prompt):
            action_masks = policy_rollout_buffer.action_masks[j]
            nonzero_indices = np.nonzero(action_masks)[0]
            if len(nonzero_indices) == 0:
                continue
            baseline.append(verifier_rollout_buffer.scores[j][nonzero_indices[-1]])
        if len(baseline) == 0:
            continue
        assert len(baseline) == num_samples_per_prompt, "please check for the samples bug"
        baseline = sum(baseline) / num_samples_per_prompt

        for j in range(i, i + num_samples_per_prompt):
            last_token_idx = np.nonzero(policy_rollout_buffer.action_masks[j])[0][-1]
            score = verifier_rollout_buffer.scores[j][last_token_idx]
            # leave-one-out
            score = score - (baseline * num_samples_per_prompt - score) / (num_samples_per_prompt - 1)
            verifier_rollout_buffer.scores[j][last_token_idx] = score
    return verifier_rollout_buffer


def run(
        train_file: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        verifier_config_file: str = None,
        verifier_tokenizer_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 1,
        max_forward_batch_size: int = 1,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        chunk_size: int = None,
        inner_epochs: int = 1,
        epochs: int = 1,
        lr: float = 1e-6,
        num_samples_per_prompt: int = 4,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
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

            # Collecting policy buffer
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
                num_samples_per_prompt=num_samples_per_prompt,
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

            # Using last token reward
            rewards = []
            for i in range(len(verifier_rollout_buffer)):
                nonzero = np.nonzero(policy_rollout_buffer.action_masks[i])[0]
                if len(nonzero) > 0:
                    rewards.append(verifier_rollout_buffer.scores[i][nonzero[-1]])
            print("Average Rewards: ", np.mean(rewards))

            # Reinforce Leave-One-Out
            verifier_rollout_buffer = compute_rloo_rewards(
                verifier_rollout_buffer=verifier_rollout_buffer,
                policy_rollout_buffer=policy_rollout_buffer,
                num_samples_per_prompt=num_samples_per_prompt
            )

            rollout_buffer = PolicyRolloutBuffer(
                obs=policy_rollout_buffer.obs,
                actions=policy_rollout_buffer.actions,
                rewards=verifier_rollout_buffer.scores,
                values=verifier_rollout_buffer.scores,  # pseudo
                action_logits=policy_rollout_buffer.action_logits,
                action_masks=policy_rollout_buffer.action_masks,
                action_logprobs=policy_rollout_buffer.action_logprobs,
                use_last_token_reward=True,
                reward_normalize=False,
            )

            train_policy_gradient(
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
                max_batch_size=max_batch_size
            )


if __name__ == '__main__':
    fire.Fire(run)
