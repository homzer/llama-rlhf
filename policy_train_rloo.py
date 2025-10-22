import fire
import numpy as np

from policy_train_grpo import rearrange_buffers_and_group_scores
from policy_train_policy_gradient import train_policy_gradient
from policy_train_ppo import collect_actor_buffer, collect_verifier_buffer
from src.dataset import JsonDataset
from src.entities import IterationHandler
from src.parallel.initialize import setup_model_parallel
from src.ppo.buffer import CriticRolloutBuffer, PPORolloutBuffer, RolloutBuffer
from src.ppo.parallel_buffer import ParallelRolloutBuffer
from src.utils import json_load


def compute_rloo_rewards(
        policy_rollout_buffer: RolloutBuffer,
        verifier_rollout_buffer: CriticRolloutBuffer,
        num_samples_per_prompt: int,
) -> (RolloutBuffer, CriticRolloutBuffer):
    """ Compute Leave-One-Out reward in RLOO """
    policy_rollout_buffer = ParallelRolloutBuffer(**policy_rollout_buffer)
    policy_rollout_buffer.gather_from_data_parallel_region()
    verifier_rollout_buffer = ParallelRolloutBuffer(**verifier_rollout_buffer)
    verifier_rollout_buffer.gather_from_data_parallel_region()

    policy_rollout_buffer, verifier_rollout_buffer, scores = rearrange_buffers_and_group_scores(
        policy_rollout_buffer=policy_rollout_buffer,
        verifier_rollout_buffer=verifier_rollout_buffer,
        num_samples_per_prompt=num_samples_per_prompt
    )

    # leave-one-out
    baselines = (scores.mean(axis=-1, keepdims=True) * num_samples_per_prompt - scores) / (num_samples_per_prompt - 1)
    scores = (scores - baselines).reshape(-1).tolist()
    # update verifier rollout buffer
    verifier_rollout_buffer = ParallelRolloutBuffer(**CriticRolloutBuffer(
        scores=scores, action_masks=verifier_rollout_buffer["action_masks"]
    ))

    # shuffle
    random_indices = np.random.permutation(policy_rollout_buffer.size())
    policy_rollout_buffer.rearrange(random_indices)
    verifier_rollout_buffer.rearrange(random_indices)

    policy_rollout_buffer.scatter_to_data_parallel_region()
    verifier_rollout_buffer.scatter_to_data_parallel_region()

    return policy_rollout_buffer, verifier_rollout_buffer


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
        save_optim: bool = False,
        accumulation_steps: int = 1,
        use_chat_template: bool = False,
        use_last_token_reward: bool = False,
):
    setup_model_parallel()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir
    verifier_config_file = verifier_config_file or verifier_ckpt_dir
    verifier_tokenizer_file = verifier_tokenizer_file or verifier_ckpt_dir

    for epoch, datalist in IterationHandler(json_load(train_file), epochs, chunk_size, begin_epoch):
        dataset = JsonDataset(datalist)
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
            max_forward_batch_size=max_forward_batch_size,
            use_last_token_reward=use_last_token_reward
        )

        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")

        # Reinforce Leave-One-Out
        policy_rollout_buffer, verifier_rollout_buffer = compute_rloo_rewards(
            verifier_rollout_buffer=verifier_rollout_buffer,
            policy_rollout_buffer=policy_rollout_buffer,
            num_samples_per_prompt=num_samples_per_prompt
        )

        rollout_buffer = RolloutBuffer(
            obs=policy_rollout_buffer["obs"],
            actions=policy_rollout_buffer["actions"],
            rewards=verifier_rollout_buffer["scores"],
            action_masks=policy_rollout_buffer["action_masks"],
            action_logprobs=policy_rollout_buffer["action_logprobs"],
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
            max_batch_size=max_batch_size,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )


if __name__ == '__main__':
    fire.Fire(run)
