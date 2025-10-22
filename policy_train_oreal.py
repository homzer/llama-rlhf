import gc
import os

import fire
import numpy as np
import torch

from policy_train_grpo import rearrange_buffers_and_group_scores
from policy_train_ppo_with_rule_rm import collect_actor_buffer_with_label, collect_rule_based_verifier_buffer
from policy_train_ppo_with_evaluate import evaluate_actor
from src.dataset import JsonDataset
from src.entities import Timer, IterationHandler
from src.modeling import get_parallel_model
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.parallel.optimizer import ParallelOptimizer
from src.ppo.buffer import RolloutBuffer
from src.ppo.collector import ActorForwardBufferCollector
from src.ppo.parallel_buffer import ParallelRolloutBuffer
from src.ppo.trainer import ParallelOREALTrainerForCausalLM
from src.utils import json_load, print_current_func_args


def create_balance_mask(scores: np.ndarray) -> np.ndarray:
    """ Create mask that select one score 1 sample and one score 0 sample at the last dim """
    all_ones = (scores == 1)
    all_zeros = (scores == 0)
    has_1 = all_ones.any(axis=-1)
    has_0 = all_zeros.any(axis=-1)
    condition = has_1 & has_0
    condition_expanded = np.expand_dims(condition, axis=-1) & np.ones_like(scores, dtype=bool)
    cum_one = all_ones.cumsum(axis=-1)
    first_one_mask = (cum_one == 1) & all_ones
    cum_zero = all_zeros.cumsum(axis=-1)
    first_zero_mask = (cum_zero == 1) & all_zeros
    mask = (first_one_mask | first_zero_mask) & condition_expanded
    return mask


def compute_oreal_rewards(
        policy_rollout_buffer: RolloutBuffer,
        verifier_rollout_buffer: RolloutBuffer,
        num_samples_per_prompt: int,
):
    policy_rollout_buffer = ParallelRolloutBuffer(**policy_rollout_buffer)
    policy_rollout_buffer.gather_from_data_parallel_region()
    verifier_rollout_buffer = ParallelRolloutBuffer(**verifier_rollout_buffer)
    verifier_rollout_buffer.gather_from_data_parallel_region()

    policy_rollout_buffer, verifier_rollout_buffer, scores = rearrange_buffers_and_group_scores(
        policy_rollout_buffer=policy_rollout_buffer,
        verifier_rollout_buffer=verifier_rollout_buffer,
        num_samples_per_prompt=num_samples_per_prompt
    )

    balance_masks = create_balance_mask(scores)
    score_indices = np.where(balance_masks.reshape(-1))[0]
    print(f"Filter {len(score_indices)} group scores from {len(balance_masks.reshape(-1))}")
    np.random.shuffle(score_indices)
    policy_rollout_buffer.rearrange(score_indices)
    verifier_rollout_buffer.rearrange(score_indices)
    policy_rollout_buffer.scatter_to_data_parallel_region()
    verifier_rollout_buffer.scatter_to_data_parallel_region()
    return policy_rollout_buffer, verifier_rollout_buffer


def collect_reference_buffer(
        policy_rollout_buffer: RolloutBuffer,
        reference_model_type: str,
        reference_config_file: str,
        reference_tokenizer_file: str,
        reference_ckpt_dir: str,
        max_seq_len: int,
        max_forward_batch_size: int,
        dtype: str
) -> RolloutBuffer:
    reference, reference_tokenizer = get_parallel_model(
        model_type=reference_model_type,
        config_file=reference_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=reference_tokenizer_file,
        lora_rank=-1,
        dtype=dtype
    )
    reference.load(reference_ckpt_dir)
    reference_buffer_collector = ActorForwardBufferCollector(
        actor=reference,
        tokenizer=reference_tokenizer,
        max_seq_len=max_seq_len
    )
    reference_rollout_buffer = RolloutBuffer()
    print("Reference buffer collecting ...")
    timer = Timer(total=policy_rollout_buffer.size() // max_forward_batch_size, episode=10)
    for data in policy_rollout_buffer.get(max_forward_batch_size):
        timer.step()
        reference_rollout_buffer.extend(
            reference_buffer_collector.forward(data.instructions, data.responses)
        )

    reference.cpu()
    del reference
    del reference_buffer_collector
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    return reference_rollout_buffer


def train_oreal(
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
        beta: float = 0.01,
        save_optim: bool = False,
        accumulation_steps: int = 1
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
    trainer = ParallelOREALTrainerForCausalLM(
        policy, optimizer, beta=beta, save_optim=save_optim, accumulation_steps=accumulation_steps)

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
                print(f'Policy Loss: {trainer_outputs.loss}')
                print(f'KL Loss: {trainer_outputs.kl_loss}')
    trainer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))

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
        log_dir: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        reference_ckpt_dir: str,
        label_file: str = None,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 2,
        max_generate_batch_size: int = 256,
        max_forward_batch_size: int = 36,
        max_seq_len: int = 1536,
        temperature: float = 0.6,
        top_p: float = 0.95,
        num_samples_per_prompt: int = 8,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 5e-7,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        beta: float = 0.01,
        use_chat_template: bool = False,
        seed: int = None,
        save_optim: bool = False,
        accumulation_steps: int = 1,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
):
    parallel_infos = setup_model_parallel(
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
            actor_rollout_buffer=policy_rollout_buffer, task=task, error_reward=0.0
        )

        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")

        policy_rollout_buffer, verifier_rollout_buffer = compute_oreal_rewards(
            policy_rollout_buffer=policy_rollout_buffer,
            verifier_rollout_buffer=verifier_rollout_buffer,
            num_samples_per_prompt=num_samples_per_prompt
        )

        reference_rollout_buffer = collect_reference_buffer(
            policy_rollout_buffer=policy_rollout_buffer,
            reference_model_type=policy_model_type,
            reference_config_file=policy_config_file,
            reference_tokenizer_file=policy_tokenizer_file,
            reference_ckpt_dir=reference_ckpt_dir,
            max_seq_len=max_seq_len,
            max_forward_batch_size=max_forward_batch_size,
            dtype=dtype
        )

        rollout_buffer = RolloutBuffer(
            obs=policy_rollout_buffer["obs"],
            actions=policy_rollout_buffer["actions"],
            action_masks=policy_rollout_buffer["action_masks"],
            rewards=verifier_rollout_buffer["scores"],
            action_logprobs=policy_rollout_buffer["action_logprobs"],
            ref_action_logprobs=reference_rollout_buffer["action_logprobs"],
        )

        train_oreal(
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
            beta=beta,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )

        if parallel_infos.global_rank == 0:
            rollout_buffer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))

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
