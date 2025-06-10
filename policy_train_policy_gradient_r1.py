""" Policy Gradient with Rule-Based Rewards """
import collections
import os

import fire
import numpy as np
import torch

from policy_train_policy_gradient import re_scoring_eos_rewards, train_policy_gradient
from policy_train_ppo import collect_actor_buffer, collect_verifier_buffer
from policy_train_ppo_with_evaluate import evaluate_actor
from src.dataset import JsonDataset
from src.evaluator import MATHEvaluator
from src.parallel.initialize import setup_model_parallel
from src.ppo.buffer import PolicyRolloutBuffer, RolloutBuffer, CriticRolloutBuffer
from src.utils import json_load, masked_std, print_current_func_args


def reward_fn(
        policy_rollout_buffer: RolloutBuffer,
        dataset: JsonDataset
):
    """
    :return:
    acc_rewards: 1 for correct answer, -1 for incorrect answer.
    think_len_rewards: 1 for the longest correct think response, 0 for the shortest.
    """
    evaluator = MATHEvaluator()
    acc_rewards = []
    think_len_rewards = []
    max_think_len = None
    min_think_len = None
    for i, data in enumerate(policy_rollout_buffer.get(1)):
        assert dataset[i]["instruction"] in data.instructions[0]
        if evaluator.eval(data.responses[0], dataset[i]["label"]) is True:
            acc_rewards.append(1.0)  # answer correct
            think_len = len(data.responses[0])
            max_think_len = think_len if max_think_len is None else max(max_think_len, think_len)
            min_think_len = think_len if min_think_len is None else min(min_think_len, think_len)
        else:
            acc_rewards.append(-1.0)  # answer incorrect

    for i, data in enumerate(policy_rollout_buffer.get(1)):
        if acc_rewards[i] == 1.0:  # answer correct
            # think length rewards
            if max_think_len is not None and min_think_len is not None and max_think_len > min_think_len:
                think_len = len(data.responses[0])
                think_len_rewards.append((think_len - min_think_len) / (max_think_len - min_think_len))
            else:
                think_len_rewards.append(0)
        else:
            think_len_rewards.append(0)

    assert len(acc_rewards) == len(think_len_rewards)
    Output = collections.namedtuple("Output", ["acc_rewards", "think_len_rewards"])
    return Output(acc_rewards=acc_rewards, think_len_rewards=think_len_rewards)


def collect_rule_based_verifier_buffer(
        policy_rollout_buffer: RolloutBuffer,
        dataset: JsonDataset,
        num_samples_per_prompt: int,
) -> CriticRolloutBuffer:
    scores = []
    datalist = []
    for data in dataset.datalist:
        for _ in range(num_samples_per_prompt):  # copy for multi-sampling
            datalist.append(data)
    dataset = JsonDataset(datalist)
    outputs = reward_fn(policy_rollout_buffer, dataset)
    print(f"Accuracy Rewards: {np.mean(outputs.acc_rewards)}")
    print(f"Think Length Rewards: {np.mean(outputs.think_len_rewards)}")
    for acc_reward, think_len_reward in zip(outputs.acc_rewards, outputs.think_len_rewards):
        scores.append(acc_reward + 1.5 * think_len_reward)
    rule_verifier_rollout_buffer = CriticRolloutBuffer(scores, action_masks=policy_rollout_buffer["action_masks"])

    return rule_verifier_rollout_buffer


def run(
        label_file_gsm: str,
        label_file_prm: str,
        label_file_aime: str,
        log_dir: str,
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
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 24,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
        use_last_token_reward: bool = False,
        add_rule_based_reward: bool = False
):
    parallel_infos = setup_model_parallel(seed=seed, log_dir=log_dir)
    print_current_func_args()
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
                num_samples_per_prompt=num_samples_per_prompt
            )

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

            if add_rule_based_reward:
                rule_verifier_rollout_buffer = collect_rule_based_verifier_buffer(
                    policy_rollout_buffer=policy_rollout_buffer,
                    dataset=dataset,
                    num_samples_per_prompt=num_samples_per_prompt,
                )

                # Add rule-based reward
                if use_last_token_reward:
                    rewards = []
                    for i in range(len(verifier_rollout_buffer)):
                        nonzero_indices = np.nonzero(policy_rollout_buffer["action_masks"][i])[0]
                        if len(nonzero_indices) > 0:
                            rewards.append(verifier_rollout_buffer["scores"][i][nonzero_indices][-1].item())
                    reward_std = np.std(rewards)
                else:
                    reward_std = masked_std(verifier_rollout_buffer["scores"], policy_rollout_buffer["action_masks"])
                verifier_rollout_buffer["scores"] += reward_std[:, None] * rule_verifier_rollout_buffer["scores"]

            print(f"Average Rewards: {verifier_rollout_buffer.mean(use_last_token_reward)}")

            rollout_buffer = PolicyRolloutBuffer(
                obs=policy_rollout_buffer["obs"],
                actions=policy_rollout_buffer["actions"],
                rewards=verifier_rollout_buffer["scores"],
                values=verifier_rollout_buffer["scores"],  # pseudo
                action_logits=policy_rollout_buffer["action_logits"],
                action_masks=policy_rollout_buffer["action_masks"],
                action_logprobs=policy_rollout_buffer["action_logprobs"],
                use_last_token_reward=use_last_token_reward,
            )
            rollout_buffer = re_scoring_eos_rewards(rollout_buffer)

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

            if parallel_infos.local_rank == 0:
                torch.save({
                    'obs': rollout_buffer.obs,
                    'actions': rollout_buffer.actions,
                    'rewards': rollout_buffer.origin_rewards,
                    'action_masks': rollout_buffer.action_masks
                }, os.path.join(save_dir, "epoch-%03d" % (epoch + 1), f"buffer.bin"))

            evaluate_actor(
                task="gsm8k",
                label_file=label_file_gsm,
                log_dir=log_dir,
                actor_model_type=policy_model_type,
                actor_config_file=policy_config_file,
                max_seq_len=min(1024, max_seq_len),
                actor_tokenizer_file=policy_tokenizer_file,
                dtype=dtype,
                epoch=epoch,
                actor_save_dir=save_dir,
                max_generate_batch_size=max_generate_batch_size,
                use_chat_template=use_chat_template,
            )

            evaluate_actor(
                task="prm800k",
                label_file=label_file_prm,
                log_dir=log_dir,
                actor_model_type=policy_model_type,
                actor_config_file=policy_config_file,
                max_seq_len=min(1536, max_seq_len),
                actor_tokenizer_file=policy_tokenizer_file,
                dtype=dtype,
                epoch=epoch,
                actor_save_dir=save_dir,
                max_generate_batch_size=max_generate_batch_size,
                use_chat_template=use_chat_template,
            )

            evaluate_actor(
                task="aime2024",
                label_file=label_file_aime,
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
