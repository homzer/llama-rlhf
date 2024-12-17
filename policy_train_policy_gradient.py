import gc
import os

import fire
import numpy as np
import torch

from policy_train_ppo import collect_actor_buffer, collect_verifier_buffer
from src.dataset import JsonDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.parallel.utils import setup_model_parallel, set_barrier
from src.ppo.buffer import RolloutBuffer
from src.ppo.trainer import ParallelPolicyGradientTrainerForCausalLM
from src.utils import masked_mean, json_load


def re_scoring_eos_rewards(buffer: RolloutBuffer) -> RolloutBuffer:
    # Setting the reward of [EOS] token to average reward of the sequence.
    for i, action_mask in enumerate(buffer.action_masks):
        nonzero_indices = np.nonzero(action_mask)[0]
        if len(nonzero_indices) > 0:
            buffer.rewards[i][nonzero_indices[-1]] = np.mean(buffer.rewards[i][action_mask])

    return buffer


def re_scoring_token_rewards(buffer: RolloutBuffer) -> RolloutBuffer:
    # Setting the rewards of every token to be the same as the [EOS] token reward.
    for i, action_mask in enumerate(buffer.action_masks):
        nonzero_indices = np.nonzero(action_mask)[0]
        if len(nonzero_indices) > 0:
            buffer.rewards[i][action_mask] = buffer.rewards[i][nonzero_indices[-1]]

    return buffer


def train_policy_gradient(
        rollout_buffer: RolloutBuffer,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_config_file: str,
        policy_tokenizer_file: str,
        max_seq_len: int,
        lora_rank: int,
        dtype: str,
        lora_dtype: str,
        lr: float,
        epoch: int,
        inner_epochs: int,
        save_dir: str,
        max_batch_size: int,
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
    trainer = ParallelPolicyGradientTrainerForCausalLM(policy, optimizer)
    trainer.load_model(policy_ckpt_dir) if (
            epoch == 0
    ) else trainer.load(os.path.join(save_dir, f"epoch-{epoch}"))
    print('Policy training ...')
    timer = Timer(total=(len(rollout_buffer) // max_batch_size) * inner_epochs, episode=100)
    for inner_epoch in range(inner_epochs):
        for data in rollout_buffer.get(max_batch_size):
            timer.step()
            trainer_outputs = trainer.forward(data)
            if trainer.step % 100 == 0:
                print(f'--------- STEP {trainer.step} OF {timer.total} ---------')
                print('Loss: ', trainer_outputs.loss)
                print('Rewards: ', trainer_outputs.rewards)
    trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))

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
        inner_epochs: int = 3,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
        use_last_token_reward: bool = False
):
    parallel_infos = setup_model_parallel(seed=seed)
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

            if use_last_token_reward:
                rewards = []
                for i in range(len(verifier_rollout_buffer)):
                    nonzero_indices = np.nonzero(policy_rollout_buffer.action_masks[i])[0]
                    if len(nonzero_indices) > 0:
                        rewards.append(verifier_rollout_buffer.scores[i][nonzero_indices][-1].item())
                print("Average Rewards: ", np.mean(rewards))
            else:
                print("Average Rewards: ", masked_mean(verifier_rollout_buffer.scores, policy_rollout_buffer.action_masks))

            rollout_buffer = RolloutBuffer(
                obs=policy_rollout_buffer.obs,
                actions=policy_rollout_buffer.actions,
                rewards=verifier_rollout_buffer.scores,
                values=verifier_rollout_buffer.scores,  # pseudo
                action_logits=policy_rollout_buffer.action_logits,
                action_masks=policy_rollout_buffer.action_masks,
                action_logprobs=policy_rollout_buffer.action_logprobs,
                use_last_token_reward=use_last_token_reward
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
                }, os.path.join(save_dir, f"epoch-{epoch + 1}", f"buffer.bin"))


if __name__ == '__main__':
    fire.Fire(run)
