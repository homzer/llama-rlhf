import gc
import os

import fire
import numpy as np
import torch

from policy_train_ppo_with_rule_rm import collect_actor_buffer_with_label, collect_rule_based_verifier_buffer
from policy_train_ppo_with_evaluate import evaluate_actor
from src.dataset import JsonDataset
from src.entities import Timer, IterationHandler
from src.modeling import get_parallel_model
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.parallel.optimizer import ParallelOptimizer
from src.ppo.buffer import CriticRolloutBuffer, RolloutBuffer
from src.ppo.collector import ActorForwardBufferCollector
from src.ppo.parallel_buffer import ParallelRolloutBuffer
from src.ppo.trainer import ParallelPolicyGradientGuiderTrainerForCausalLM
from src.utils import json_load, print_current_func_args, masked_mean


def split_buffer_into_positive_and_negative(
        policy_rollout_buffer: RolloutBuffer,
        verifier_rollout_buffer: CriticRolloutBuffer,
        use_last_token_reward: bool
):
    if use_last_token_reward:
        scores = []
        for i in range(verifier_rollout_buffer.size()):
            nonzero_indices = np.nonzero(verifier_rollout_buffer["action_masks"][i])[0]
            if len(nonzero_indices) > 0:
                scores.append(verifier_rollout_buffer["scores"][i][nonzero_indices[-1]])
            else:
                scores.append(0.0)
        scores = np.stack(scores, axis=0)
    else:
        scores = masked_mean(verifier_rollout_buffer["scores"], verifier_rollout_buffer["action_masks"], dim=-1)
    pos_scores_indices = np.arange(0, verifier_rollout_buffer.size())[scores > 0]
    neg_scores_indices = np.arange(0, verifier_rollout_buffer.size())[scores <= 0]

    # Select negative reward samples
    pos_policy_buffer = policy_rollout_buffer.copy()
    neg_policy_buffer = policy_rollout_buffer.copy()
    pos_verifier_buffer = verifier_rollout_buffer.copy()
    neg_verifier_buffer = verifier_rollout_buffer.copy()

    pos_policy_buffer.rearrange(pos_scores_indices)
    neg_policy_buffer.rearrange(neg_scores_indices)
    pos_verifier_buffer.rearrange(pos_scores_indices)
    neg_verifier_buffer.rearrange(neg_scores_indices)

    # Gather and distribute to even data
    pos_policy_buffer = ParallelRolloutBuffer(**pos_policy_buffer)
    pos_verifier_buffer = ParallelRolloutBuffer(**pos_verifier_buffer)
    neg_policy_buffer = ParallelRolloutBuffer(**neg_policy_buffer)
    neg_verifier_buffer = ParallelRolloutBuffer(**neg_verifier_buffer)

    pos_policy_buffer.gather_from_data_parallel_region()
    pos_verifier_buffer.gather_from_data_parallel_region()
    neg_policy_buffer.gather_from_data_parallel_region()
    neg_verifier_buffer.gather_from_data_parallel_region()

    print(f"Num of positive samples: {pos_policy_buffer.size()} | "
          f"Num of negative samples: {neg_policy_buffer.size()}")

    pos_policy_buffer.scatter_to_data_parallel_region()
    pos_verifier_buffer.scatter_to_data_parallel_region()
    neg_policy_buffer.scatter_to_data_parallel_region()
    neg_verifier_buffer.scatter_to_data_parallel_region()
    return pos_policy_buffer, pos_verifier_buffer, neg_policy_buffer, neg_verifier_buffer


def collect_guider_buffer(
        policy_rollout_buffer: RolloutBuffer,
        guider_ckpt_dir: str,
        guider_model_type: str,
        guider_config_file: str,
        guider_tokenizer_file: str,
        max_seq_len: int,
        dtype: str,
        max_forward_batch_size: int,
) -> RolloutBuffer:
    guider, guider_tokenizer = get_parallel_model(
        model_type=guider_model_type,
        config_file=guider_config_file,
        tokenizer_file=guider_tokenizer_file,
        max_seq_len=max_seq_len,
        dtype=dtype
    )
    guider.load(guider_ckpt_dir)
    guider_buffer_collector = ActorForwardBufferCollector(
        actor=guider,
        tokenizer=guider_tokenizer,
        max_seq_len=max_seq_len
    )
    guider_rollout_buffer = RolloutBuffer()
    print("Guider buffer collecting ...")
    timer = Timer(total=policy_rollout_buffer.size() // max_forward_batch_size, episode=10)
    for data in policy_rollout_buffer.get(max_forward_batch_size):
        timer.step()
        guider_rollout_buffer.extend(
            guider_buffer_collector.forward(data.instructions, data.responses)
        )

    guider.cpu()
    del guider
    del guider_buffer_collector
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    return guider_rollout_buffer


def create_policy_rollout_buffer_with_guidance(
        pos_policy_buffer: RolloutBuffer,
        pos_verifier_buffer: RolloutBuffer,
        guider_rollout_buffer: RolloutBuffer,
        neg_policy_buffer: RolloutBuffer,
        neg_verifier_buffer: RolloutBuffer,
) -> RolloutBuffer:
    rollout_buffer = RolloutBuffer(
        obs=np.concatenate([pos_policy_buffer["obs"], neg_policy_buffer["obs"]]),
        actions=np.concatenate([pos_policy_buffer["actions"], neg_policy_buffer["actions"]]),
        rewards=np.concatenate([pos_verifier_buffer["scores"], neg_verifier_buffer["scores"]]),
        action_masks=np.concatenate([pos_policy_buffer["action_masks"], neg_policy_buffer["action_masks"]]),
        guider_actions=np.concatenate([pos_policy_buffer["actions"], guider_rollout_buffer["output_actions"]])
    )
    return rollout_buffer


def train_policy_gradient_with_guidance(
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
    trainer = ParallelPolicyGradientGuiderTrainerForCausalLM(
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
                print(f'Positive Action Loss: {trainer_outputs.pos_action_loss}')
                print(f'Guider Action Loss: {trainer_outputs.guider_action_loss}')
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
        guider_ckpt_dir: str,
        policy_model_type: str,
        label_file: str = None,
        guider_config_file: str = None,
        guider_tokenizer_file: str = None,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 36,
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
            actor_rollout_buffer=policy_rollout_buffer, task=task
        )

        print(f"Average Rewards: {verifier_rollout_buffer.mean(use_last_token_reward=True)}")

        pos_policy_buffer, pos_verifier_buffer, neg_policy_buffer, neg_verifier_buffer = split_buffer_into_positive_and_negative(
            policy_rollout_buffer=policy_rollout_buffer,
            verifier_rollout_buffer=verifier_rollout_buffer,
            use_last_token_reward=use_chat_template
        )

        guider_rollout_buffer = collect_guider_buffer(
            policy_rollout_buffer=neg_policy_buffer,
            guider_ckpt_dir=guider_ckpt_dir,
            guider_model_type=policy_model_type,
            guider_config_file=guider_config_file,
            guider_tokenizer_file=guider_tokenizer_file,
            max_seq_len=max_seq_len,
            dtype=dtype,
            max_forward_batch_size=max_forward_batch_size
        )

        # pos_policy_buffer.save(os.path.join(log_dir, "epoch-%03d" % (epoch + 1), "pos-policy"))
        # pos_verifier_buffer.save(os.path.join(log_dir, "epoch-%03d" % (epoch + 1), "pos-verifier"))
        # ParallelRolloutBuffer(**guider_rollout_buffer).save(os.path.join(log_dir, "epoch-%03d" % (epoch + 1), "guider"))
        # neg_verifier_buffer.save(os.path.join(log_dir, "epoch-%03d" % (epoch + 1), "neg-verifier"))
        # neg_policy_buffer.save(os.path.join(log_dir, "epoch-%03d" % (epoch + 1), "neg-policy"))

        rollout_buffer = create_policy_rollout_buffer_with_guidance(
            pos_policy_buffer=pos_policy_buffer,
            pos_verifier_buffer=pos_verifier_buffer,
            guider_rollout_buffer=guider_rollout_buffer,
            neg_verifier_buffer=neg_verifier_buffer,
            neg_policy_buffer=neg_policy_buffer
        )

        train_policy_gradient_with_guidance(
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

        if parallel_infos.global_rank == 0:
            rollout_buffer.save(os.path.join(log_dir, "epoch-%03d" % (epoch + 1)))

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
