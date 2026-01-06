import gc
import os
import shutil

import fire
import numpy as np
import torch

from policy_train_grpo import rearrange_buffers_and_group_scores
from policy_train_ppo import collect_verifier_buffer, collect_actor_buffer
from policy_train_ppo_with_evaluate import evaluate_actor
from src.dataset import JsonDataset
from src.entities import IterationHandler, Timer
from src.modeling import get_parallel_model
from src.parallel.initialize import setup_model_parallel, set_barrier, get_rank
from src.ppo.buffer import RolloutBuffer, CriticRolloutBuffer
from src.ppo.parallel_buffer import ParallelRolloutBuffer
from src.ppo.trainer import ParallelDAPOTrainerForCausalLM
from src.utils import json_load, print_current_func_args


def compute_dapo_rewards(
        policy_rollout_buffer: RolloutBuffer,
        verifier_rollout_buffer: CriticRolloutBuffer,
        num_samples_per_prompt: int
) -> (RolloutBuffer, CriticRolloutBuffer):
    """ Compute DAPO rewards """
    policy_rollout_buffer = ParallelRolloutBuffer(**policy_rollout_buffer)
    policy_rollout_buffer.gather_from_data_parallel_region()
    verifier_rollout_buffer = ParallelRolloutBuffer(**verifier_rollout_buffer)
    verifier_rollout_buffer.gather_from_data_parallel_region()

    policy_rollout_buffer, verifier_rollout_buffer, scores = rearrange_buffers_and_group_scores(
        policy_rollout_buffer=policy_rollout_buffer,
        verifier_rollout_buffer=verifier_rollout_buffer,
        num_samples_per_prompt=num_samples_per_prompt
    )

    # dynamic sampling in DAPO
    scores_std = scores.std(axis=-1, keepdims=True)
    scores_std = np.repeat(scores_std, repeats=num_samples_per_prompt, axis=-1)
    equivalent_masks = (scores_std != 0).reshape(-1)
    filtering_indices = np.arange(len(equivalent_masks))[equivalent_masks]
    policy_rollout_buffer.rearrange(filtering_indices)
    verifier_rollout_buffer.rearrange(filtering_indices)

    scores = (scores.reshape(-1)[equivalent_masks]).reshape(-1, num_samples_per_prompt)
    scores = (scores - scores.mean(axis=-1, keepdims=True)) / scores.std(axis=-1, keepdims=True)

    # token-level policy gradient in DAPO
    action_masks = verifier_rollout_buffer["action_masks"]
    action_weights = action_masks.sum(-1)
    action_weights = action_weights.reshape(-1, num_samples_per_prompt).sum(-1, keepdims=True)
    action_weights[action_weights == 0] = 1e-12
    scores = scores / action_weights

    # update verifier rollout buffer
    scores = scores.reshape(-1).tolist()
    verifier_rollout_buffer = ParallelRolloutBuffer(**CriticRolloutBuffer(
        scores=scores, action_masks=verifier_rollout_buffer["action_masks"]
    ))

    # shuffle
    randon_indices = np.random.permutation(policy_rollout_buffer.size())
    policy_rollout_buffer.rearrange(randon_indices)
    verifier_rollout_buffer.rearrange(randon_indices)

    policy_rollout_buffer.scatter_to_data_parallel_region()
    verifier_rollout_buffer.scatter_to_data_parallel_region()

    return policy_rollout_buffer, verifier_rollout_buffer


def train_dapo(
        policy_model_type: str,
        policy_config_file: str,
        max_seq_len: int,
        policy_tokenizer_file: str,
        lora_rank: int,
        dtype: str,
        lora_dtype: str,
        lr: float,
        epoch: int,
        policy_ckpt_dir: str,
        save_dir: str,
        rollout_buffer: RolloutBuffer,
        max_batch_size: int,
        inner_epochs: int,
        clip_range_higher: float = 0.28,
        clip_range_lower: float = 0.20,
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
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    trainer = ParallelDAPOTrainerForCausalLM(
        policy=policy,
        optimizer=optimizer,
        clip_range_higher=clip_range_higher,
        clip_range_lower=clip_range_lower,
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
                print(f"Ratio: {trainer_outputs.ratio}")
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


def run(
        task: str,
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
        log_dir: str = None,
        label_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 24,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 4,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-6,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        clip_range_higher: float = 0.28,
        clip_range_lower: float = 0.20,
        use_chat_template: bool = False,
        use_last_token_reward: bool = False,
        last_token_reward_only: bool = False,
        seed: int = None,
        save_optim: bool = False,
        accumulation_steps: int = 1,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1
):
    setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        log_mode="w" if begin_epoch == 0 else "a",
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir

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
            use_last_token_reward=use_last_token_reward,
            last_token_reward_only=last_token_reward_only
        )

        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")

        policy_rollout_buffer, verifier_rollout_buffer = compute_dapo_rewards(
            verifier_rollout_buffer=verifier_rollout_buffer,
            policy_rollout_buffer=policy_rollout_buffer,
            num_samples_per_prompt=num_samples_per_prompt
        )

        rollout_buffer = RolloutBuffer(
            obs=policy_rollout_buffer["obs"],
            actions=policy_rollout_buffer["actions"],
            advantages=verifier_rollout_buffer["scores"],
            action_masks=policy_rollout_buffer["action_masks"],
            action_logprobs=policy_rollout_buffer["action_logprobs"],
        )

        train_dapo(
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
            clip_range_higher=clip_range_higher,
            clip_range_lower=clip_range_lower,
            save_optim=save_optim,
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
