import fire
import numpy as np

from policy_train_policy_gradient import train_policy_gradient
from policy_train_ppo import collect_actor_buffer, collect_verifier_buffer
from src.dataset import JsonDataset
from src.parallel.utils import setup_model_parallel
from src.ppo.buffer import CriticRolloutBuffer, RolloutBuffer, ActorRolloutBuffer
from src.utils import json_load


def compute_rloo_rewards(
        verifier_rollout_buffer: CriticRolloutBuffer,
        policy_rollout_buffer: ActorRolloutBuffer,
        num_samples_per_prompt: int,
) -> CriticRolloutBuffer:
    """ Normalize EOS token reward across batch. And distribute normalized reward across output tokens """
    assert len(verifier_rollout_buffer) % num_samples_per_prompt == 0
    for i in range(0, len(verifier_rollout_buffer), num_samples_per_prompt):
        assert all(
            instruction == policy_rollout_buffer.instructions[i] for
            instruction in policy_rollout_buffer.instructions[i + 1: i + num_samples_per_prompt]
        )

        # compute baseline reward
        baseline = []
        for j in range(i, i + num_samples_per_prompt):
            action_mask = policy_rollout_buffer.action_masks[j]
            last_idx = np.nonzero(action_mask)[0]
            if len(last_idx) == 0:
                continue
            baseline.append(verifier_rollout_buffer.scores[j][last_idx])
        if len(baseline) == 0:
            continue
        assert len(baseline) == num_samples_per_prompt, "please check for the samples bug"
        baseline_score = sum(baseline) / len(baseline)

        for j in range(i, i + num_samples_per_prompt):
            action_mask = policy_rollout_buffer.action_masks[j]
            score = verifier_rollout_buffer.scores[j][np.nonzero(action_mask)[0][-1].item()]
            # leave-one-out
            score = (baseline_score * len(baseline) - score) / (len(baseline) - 1)
            verifier_rollout_buffer.scores[j][action_mask] = score
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
        chunk_size: int = None,
        inner_epochs: int = 1,
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
    epochs = len(datalist) // chunk_size
    for epoch in range(begin_epoch, epochs):
        print(f"Epoch - {epoch} of {epochs}")
        dataset = JsonDataset(f=datalist[epoch * chunk_size: (epoch + 1) * chunk_size])

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
            last_idx = np.nonzero(policy_rollout_buffer.action_masks[i])[0][-1].item()
            rewards.append(verifier_rollout_buffer.scores[i][last_idx])
        print("Average Rewards: ", np.mean(rewards))

        # Reinforce Leave-One-Out
        verifier_rollout_buffer = compute_rloo_rewards(verifier_rollout_buffer)

        rollout_buffer = RolloutBuffer(
            obs=policy_rollout_buffer.obs,
            actions=policy_rollout_buffer.actions,
            rewards=verifier_rollout_buffer.scores,
            values=verifier_rollout_buffer.scores,  # pseudo
            action_logits=policy_rollout_buffer.action_logits,
            action_masks=policy_rollout_buffer.action_masks,
            action_logprobs=policy_rollout_buffer.action_logprobs,
            ref_action_logprobs=None,
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
