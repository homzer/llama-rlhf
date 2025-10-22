import os

import fire
import numpy as np
import torch.optim

from policy_train_ppo_with_rule_rm import collect_actor_buffer_with_label, collect_rule_based_verifier_buffer
from policy_train_ppo_best_of_n import select_best_of_n_buffer
from policy_train_ppo_with_evaluate import evaluate_actor
from policy_train_rft import train_rft_policy
from src.dataset import JsonDataset
from src.entities import IterationHandler
from src.parallel.initialize import setup_model_parallel
from src.ppo.buffer import CriticRolloutBuffer, RolloutBuffer
from src.ppo.parallel_buffer import ParallelRolloutBuffer
from src.utils import json_load, print_current_func_args, masked_mean


def select_lowest_confidence_of_n_buffer(
        actor_rollout_buffer: RolloutBuffer,
        verifier_rollout_buffer: CriticRolloutBuffer,
        num_samples_per_prompt: int,
        num_samples_keep_per_prompt: int,
        use_last_token_reward: bool
) -> (RolloutBuffer, CriticRolloutBuffer):
    actor_rollout_buffer = ParallelRolloutBuffer(**actor_rollout_buffer)
    actor_rollout_buffer.gather_from_data_parallel_region()
    verifier_rollout_buffer = ParallelRolloutBuffer(**verifier_rollout_buffer)
    verifier_rollout_buffer.gather_from_data_parallel_region()

    sorted_indices = sorted(range(actor_rollout_buffer.size()), key=lambda x: actor_rollout_buffer["instructions"][x])
    # Sort actor rollout buffer
    actor_rollout_buffer.rearrange(sorted_indices)
    # Sort critic rollout buffer
    verifier_rollout_buffer.rearrange(sorted_indices)
    # check for arranged instructions
    for i in range(0, actor_rollout_buffer.size(), num_samples_per_prompt):
        assert len(set(actor_rollout_buffer["instructions"][i: i + num_samples_per_prompt].tolist())) == 1

    num_samples_keep_per_prompt = min(num_samples_per_prompt, num_samples_keep_per_prompt)

    # Sorted by the confidence of responses
    action_logprobs = masked_mean(actor_rollout_buffer["action_logprobs"], actor_rollout_buffer["action_masks"], dim=-1)
    action_logprobs = action_logprobs.reshape(-1, num_samples_per_prompt)
    action_logprobs_indices = np.argsort(action_logprobs, axis=-1)
    action_logprobs_indices += (np.arange(0, len(action_logprobs_indices)) * num_samples_per_prompt)[..., None]
    action_logprobs_indices = action_logprobs_indices.reshape(-1).tolist()
    # update actor rollout buffer
    actor_rollout_buffer.rearrange(action_logprobs_indices)
    # update verifier rollout buffer
    verifier_rollout_buffer.rearrange(action_logprobs_indices)

    if use_last_token_reward:
        scores = []
        for i in range(verifier_rollout_buffer.size()):
            nonzero_indices = np.nonzero(verifier_rollout_buffer["action_masks"][i])[0]
            if len(nonzero_indices) > 0:
                scores.append(verifier_rollout_buffer["scores"][i][nonzero_indices[-1]])
            else:
                scores.append(-1.0)
        scores = np.stack(scores, axis=0)
    else:
        scores = masked_mean(verifier_rollout_buffer["scores"], verifier_rollout_buffer["action_masks"], dim=-1)

    scores_values, scores_indices = torch.topk(
        torch.from_numpy(scores).reshape(-1, num_samples_per_prompt), k=num_samples_keep_per_prompt
    )
    scores_indices += (torch.arange(0, len(scores_indices)) * num_samples_per_prompt).unsqueeze(-1)
    # filter out error samples
    scores_indices = scores_indices.reshape(-1)[scores_values.reshape(-1) > 0].tolist()
    print(f"Pass@{num_samples_per_prompt}: {(scores_values.reshape(-1) > 0).sum().item() / len(scores_values.reshape(-1))}")

    # update actor rollout buffer
    actor_rollout_buffer.rearrange(scores_indices)
    # update verifier rollout buffer
    verifier_rollout_buffer.rearrange(scores_indices)

    # Shuffle
    randon_indices = np.random.permutation(actor_rollout_buffer.size())
    actor_rollout_buffer.rearrange(randon_indices)
    verifier_rollout_buffer.rearrange(randon_indices)
    actor_rollout_buffer.scatter_to_data_parallel_region()
    verifier_rollout_buffer.scatter_to_data_parallel_region()

    return actor_rollout_buffer, verifier_rollout_buffer


def run(
        task: str,
        log_dir: str,
        train_file: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        label_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 1,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        chunk_size: int = None,
        inner_epochs: int = 1,
        epochs: int = 1,
        lr: float = 1e-6,
        num_samples_per_prompt: int = 4,
        num_samples_keep_per_prompt: int = 1,
        select_low_confidence_sample: bool = False,
        accumulation_steps: int = 1,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        score_threshold: float = None,
        use_chat_template: bool = False,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
        reuse_buffer: bool = False
):
    setup_model_parallel(
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size,
        log_dir=log_dir,
        log_mode='w' if begin_epoch == 0 else 'a'
    )
    print_current_func_args()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir

    for epoch, datalist in IterationHandler(json_load(train_file), epochs, chunk_size, begin_epoch):
        dataset = JsonDataset(datalist)
        if len(dataset) == 0:
            continue

        if reuse_buffer and os.path.exists(os.path.join(save_dir, "epoch-%03d" % epoch)):
            policy_rollout_buffer = ParallelRolloutBuffer.load(os.path.join(save_dir, "epoch-%03d" % epoch))
        else:
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
            ParallelRolloutBuffer(**policy_rollout_buffer).save(os.path.join(save_dir, "epoch-%03d" % epoch))

        verifier_rollout_buffer = collect_rule_based_verifier_buffer(
            actor_rollout_buffer=policy_rollout_buffer, task=task
        )

        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")

        if select_low_confidence_sample:
            policy_rollout_buffer_filtered, _ = select_lowest_confidence_of_n_buffer(
                actor_rollout_buffer=policy_rollout_buffer,
                verifier_rollout_buffer=verifier_rollout_buffer,
                num_samples_per_prompt=num_samples_per_prompt,
                num_samples_keep_per_prompt=num_samples_keep_per_prompt,
                use_last_token_reward=True
            )
        else:
            policy_rollout_buffer_filtered, _ = select_best_of_n_buffer(
                actor_rollout_buffer=policy_rollout_buffer,
                verifier_rollout_buffer=verifier_rollout_buffer,
                num_samples_per_prompt=num_samples_per_prompt,
                num_samples_keep_per_prompt=num_samples_keep_per_prompt,
                use_last_token_reward=True,
                score_threshold=score_threshold
            )

        print("Training policy ......")
        train_rft_policy(
            policy_rollout_buffer=policy_rollout_buffer_filtered,
            policy_model_type=policy_model_type,
            policy_config_file=policy_config_file,
            max_seq_len=max_seq_len,
            policy_tokenizer_file=policy_tokenizer_file,
            lora_dtype=lora_dtype,
            lora_rank=lora_rank,
            dtype=dtype,
            lr=lr,
            policy_ckpt_dir=policy_ckpt_dir,
            epoch=epoch,
            save_dir=save_dir,
            max_batch_size=max_batch_size,
            inner_epochs=inner_epochs,
            accumulation_steps=accumulation_steps
        )

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
                use_chat_template=use_chat_template
            )


if __name__ == '__main__':
    fire.Fire(run)
