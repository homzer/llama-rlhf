import os

import fire
import torch

from policy_gradient_train import re_scoring_eos_rewards, train_policy_gradient
from ppo_train import collect_verifier_buffer
from ppo_train_best_of_n import collect_actor_buffer, select_best_of_n_buffer
from src.dataset import JsonDataset
from src.parallel.utils import setup_model_parallel
from src.ppo.buffer import RolloutBuffer
from src.utils import masked_mean, json_load


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
        num_samples_generate_per_prompt: int = 1,
        num_samples_keep_per_prompt: int = 1,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 24,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_seq_len: int = 1024,
        chunk_size: int = None,
        inner_epochs: int = 3,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None
):
    setup_model_parallel(seed=seed)
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
            num_samples_per_prompt=num_samples_generate_per_prompt
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

        policy_rollout_buffer, verifier_rollout_buffer = select_best_of_n_buffer(
            actor_rollout_buffer=policy_rollout_buffer,
            verifier_rollout_buffer=verifier_rollout_buffer,
            num_samples_generate_per_prompt=num_samples_generate_per_prompt,
            num_samples_keep_per_prompt=num_samples_keep_per_prompt,
            use_last_token_reward=False
        )

        print("Average Rewards: ", masked_mean(verifier_rollout_buffer.scores, policy_rollout_buffer.action_masks))

        rollout_buffer = RolloutBuffer(
            obs=policy_rollout_buffer.obs,
            actions=policy_rollout_buffer.actions,
            rewards=verifier_rollout_buffer.scores,
            values=verifier_rollout_buffer.scores,  # pseudo
            action_logits=policy_rollout_buffer.action_logits,
            action_masks=policy_rollout_buffer.action_masks,
            action_logprobs=policy_rollout_buffer.action_logprobs
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

        torch.save({
            'obs': rollout_buffer.obs,
            'actions': rollout_buffer.actions,
            'values': rollout_buffer.values,
            'rewards': rollout_buffer.rewards,
            'action_masks': rollout_buffer.action_masks,
            'advantages': rollout_buffer.advantages,
            'returns': rollout_buffer.returns
        }, os.path.join(save_dir, f"epoch-{epoch + 1}", f"buffer.bin"))


if __name__ == '__main__':
    fire.Fire(run)
