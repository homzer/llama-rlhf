""" Trained on PRM800K dataset, and evaluated on PRM800K, AIME2024, AIME2025, and AMC23 """
import fire
import numpy as np

from policy_train_policy_gradient import train_policy_gradient
from policy_train_ppo_with_rule_rm import collect_actor_buffer_with_label, reward_fn
from policy_train_ppo_best_of_n import select_best_of_n_buffer
from policy_train_ppo_with_evaluate import evaluate_actor
from src.dataset import JsonDataset
from src.parallel.initialize import setup_model_parallel
from src.ppo.buffer import PPORolloutBuffer, RolloutBuffer, CriticRolloutBuffer
from src.utils import json_load, print_current_func_args


def collect_rule_based_verifier_buffer(
        policy_rollout_buffer: RolloutBuffer, task: str
) -> CriticRolloutBuffer:
    outputs = reward_fn(policy_rollout_buffer, task=task)
    print(f"Accuracy Rewards: {np.mean(outputs.acc_rewards)}")
    print(f"Think Length Rewards: {np.mean(outputs.think_len_rewards)}")
    scores = []
    for acc_reward, think_len_reward in zip(outputs.acc_rewards, outputs.think_len_rewards):
        scores.append(acc_reward + 2.0 * think_len_reward)
    verifier_rollout_buffer = CriticRolloutBuffer(scores=scores, action_masks=policy_rollout_buffer["action_masks"])
    return verifier_rollout_buffer


def run(
        train_file: str,
        log_dir: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
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
        num_samples_keep_per_prompt: int = None,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
        train_strategy: str = "vanilla",
        delta: float = 0.01,
        reward_sub_mean: bool = False,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
):
    setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir

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
                policy_rollout_buffer=policy_rollout_buffer, task='math'
            )

            print(f"Average Rewards: {verifier_rollout_buffer.mean(use_last_token_reward=True)}")

            policy_rollout_buffer, verifier_rollout_buffer = select_best_of_n_buffer(
                actor_rollout_buffer=policy_rollout_buffer,
                verifier_rollout_buffer=verifier_rollout_buffer,
                num_samples_per_prompt=num_samples_per_prompt,
                num_samples_keep_per_prompt=num_samples_keep_per_prompt or num_samples_per_prompt,
                use_last_token_reward=True
            )

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
                train_strategy=train_strategy,
                delta=delta
            )
            for task in ['prm800k', 'aime2024', 'aime2025', 'amc23', 'gsm8k']:
                evaluate_actor(
                    task=task,
                    label_file=f"../../data/{task}_test_with_zero_shot.jsonl",
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
