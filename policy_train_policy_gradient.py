import gc
import os
import shutil

import fire
import numpy as np
import torch

from policy_train_ppo import collect_actor_buffer, collect_verifier_buffer
from src.dataset import JsonDataset
from src.entities import Timer, IterationHandler
from src.modeling import get_parallel_model
from src.parallel.initialize import setup_model_parallel, set_barrier, get_rank
from src.parallel.optimizer import ParallelOptimizer
from src.ppo.buffer import PPORolloutBuffer, RolloutBuffer
from src.ppo.trainer import ParallelPolicyGradientTrainerForCausalLM
from src.utils import json_load, print_current_func_args


# TODO: deprecated
def re_scoring_eos_rewards(buffer: PPORolloutBuffer | RolloutBuffer):
    # Setting the reward of [EOS] token to average reward of the sequence.
    if isinstance(buffer, PPORolloutBuffer):
        for i, action_mask in enumerate(buffer.action_masks):
            nonzero_indices = np.nonzero(action_mask)[0]
            if len(nonzero_indices) > 0:
                buffer.rewards[i][nonzero_indices[-1]] = np.mean(buffer.rewards[i][action_mask])
    elif isinstance(buffer, RolloutBuffer):
        for i, action_mask in enumerate(buffer["action_masks"]):
            nonzero_indices = np.nonzero(action_mask)[0]
            if len(nonzero_indices) > 0:
                buffer["rewards"][i][nonzero_indices[-1]] = np.mean(buffer["rewards"][i][action_mask])

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
        save_optim: bool = False,
        accumulation_steps: int = 1,
        max_num_ckpts: int = None
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
    optimizer = ParallelOptimizer(torch.optim.Adam(policy.parameters(), lr=lr))
    trainer = ParallelPolicyGradientTrainerForCausalLM(
        policy=policy,
        optimizer=optimizer,
        save_optim=save_optim,
        accumulation_steps=accumulation_steps
    )

    trainer.load_model(policy_ckpt_dir) if (
            epoch == 0
    ) else trainer.load(os.path.join(save_dir, "epoch-%03d" % epoch))
    print('Policy training ...')
    timer = Timer(total=(rollout_buffer.size() // max_batch_size) * inner_epochs, episode=100)
    for inner_epoch in range(inner_epochs):
        for data in rollout_buffer.get(max_batch_size, shuffle=True, output_tensor=True):
            timer.step()
            trainer_outputs = trainer.forward(data)
            if trainer.step % 100 == 0:
                print(f'--------- STEP {trainer.step} OF {timer.total} ---------')
                print(f'Loss: {trainer_outputs.loss}')
                print(f'Advantages: {trainer_outputs.advantages}')
    trainer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))
    if max_num_ckpts is not None and (epoch + 1 - max_num_ckpts) > 0:
        rm_dir = os.path.join(save_dir, "epoch-%03d" % (epoch + 1 - max_num_ckpts))
        if get_rank() == 0 and os.path.exists(rm_dir):
            shutil.rmtree(rm_dir)

    policy.cpu()
    del policy
    del optimizer
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()


# def train_policy_gradient(
#         rollout_buffer: RolloutBuffer,
#         policy_ckpt_dir: str,
#         policy_model_type: str,
#         policy_config_file: str,
#         policy_tokenizer_file: str,
#         max_seq_len: int,
#         lora_rank: int,
#         dtype: str,
#         lora_dtype: str,
#         lr: float,
#         epoch: int,
#         inner_epochs: int,
#         save_dir: str,
#         max_batch_size: int,
#         train_strategy: str = "vanilla",  # "ratio" or "convex"
#         rho_pos: float = 1.8,
#         rho_neg: float = 0.9,
#         clip_range: float = 0.2,
#         save_optim: bool = False,
#         accumulation_steps: int = 1,
#         max_num_ckpts: int = None
# ):
#     policy, policy_tokenizer = get_parallel_model(
#         model_type=policy_model_type,
#         config_file=policy_config_file,
#         max_seq_len=max_seq_len,
#         tokenizer_file=policy_tokenizer_file,
#         lora_rank=lora_rank,
#         dtype=dtype,
#         lora_dtype=lora_dtype
#     )
#     optimizer = ParallelOptimizer(torch.optim.Adam(policy.parameters(), lr=lr))
#     if train_strategy == "vanilla":
#         print("Using ParallelPolicyGradientTrainerForCausalLM")
#         trainer = ParallelPolicyGradientTrainerForCausalLM(
#             policy, optimizer, save_optim=save_optim, accumulation_steps=accumulation_steps)
#     elif train_strategy == "ratio":
#         print("Using ParallelGRPOTrainerForCausalLM")
#         trainer = ParallelGRPOTrainerForCausalLM(
#             policy, optimizer, clip_range=clip_range, save_optim=save_optim, accumulation_steps=accumulation_steps)
#     elif train_strategy == "convex":
#         print("Using ParallelPolicyGradientConvexBoundedTrainerForCausalLM")
#         trainer = ParallelPolicyGradientLogitsConvexTrainerForCausalLM(
#             policy, optimizer, rho_pos=rho_pos, rho_neg=rho_neg, save_optim=save_optim, accumulation_steps=accumulation_steps)
#     else:
#         raise ValueError(train_strategy)
#
#     trainer.load_model(policy_ckpt_dir) if (
#             epoch == 0
#     ) else trainer.load(os.path.join(save_dir, "epoch-%03d" % epoch))
#     print('Policy training ...')
#     timer = Timer(total=(rollout_buffer.size() // max_batch_size) * inner_epochs, episode=100)
#     for inner_epoch in range(inner_epochs):
#         for data in rollout_buffer.get(max_batch_size, shuffle=True, output_tensor=True):
#             timer.step()
#             trainer_outputs = trainer.forward(data)
#             if trainer.step % 100 == 0:
#                 print(f'--------- STEP {trainer.step} OF {timer.total} ---------')
#                 print(f'Loss: {trainer_outputs.loss}')
#                 print(f'Rewards: {trainer_outputs.rewards}')
#     trainer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))
#     if max_num_ckpts is not None and (epoch + 1 - max_num_ckpts) > 0:
#         rm_dir = os.path.join(save_dir, "epoch-%03d" % (epoch + 1 - max_num_ckpts))
#         if get_rank() == 0 and os.path.exists(rm_dir):
#             shutil.rmtree(rm_dir)
#
#     policy.cpu()
#     del policy
#     del optimizer
#     del trainer
#     torch.cuda.empty_cache()
#     gc.collect()
#     set_barrier()


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
        inner_epochs: int = 1,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        log_dir: str = None,
        seed: int = None,
        train_strategy: str = "vanilla",
        use_last_token_reward: bool = False,
        last_token_reward_only: bool = False,
        save_optim: bool = False,
        accumulation_steps: int = 1,
        max_num_ckpts: int = None,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
):
    parallel_infos = setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        log_mode='w' if begin_epoch == 0 else 'a',
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()
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
            max_forward_batch_size=max_forward_batch_size,
            use_last_token_reward=use_last_token_reward,
            last_token_reward_only=last_token_reward_only
        )
        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")
        verifier_rollout_buffer.normalize()

        rollout_buffer = RolloutBuffer(
            obs=policy_rollout_buffer["obs"],
            actions=policy_rollout_buffer["actions"],
            rewards=verifier_rollout_buffer["scores"],
            action_masks=policy_rollout_buffer["action_masks"],
            action_logprobs=policy_rollout_buffer["action_logprobs"]
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
            max_batch_size=max_batch_size,
            train_strategy=train_strategy,
            accumulation_steps=accumulation_steps,
            save_optim=save_optim,
            max_num_ckpts=max_num_ckpts
        )

        if parallel_infos.local_rank == 0:
            rollout_buffer.save(os.path.join(log_dir, "epoch-%03d" % (epoch + 1)))


if __name__ == '__main__':
    fire.Fire(run)
