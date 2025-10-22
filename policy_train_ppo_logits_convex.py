import gc
import os

import fire
import torch

from policy_train_ppo import (
    collect_reference_buffer,
    collect_critic_buffer,
    train_critic,
    collect_verifier_buffer,
    collect_actor_buffer
)
from src.dataset import JsonDataset
from src.entities import IterationHandler, Timer
from src.modeling import get_parallel_model
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.parallel.optimizer import ParallelOptimizer
from src.ppo.buffer import PPORolloutBuffer
from src.ppo.trainer_logits_convex import ParallelPPOActorLogitsConvexTrainerForCausalLM
from src.utils import json_load, print_current_func_args


def train_actor_logits_convex(
        actor_model_type: str,
        actor_config_file: str,
        max_seq_len: int,
        actor_tokenizer_file: str,
        actor_lora_rank: int,
        dtype: str,
        actor_lora_dtype: str,
        lr: float,
        epoch: int,
        actor_ckpt_dir: str,
        actor_save_dir: str,
        rollout_buffer: PPORolloutBuffer,
        actor_max_batch_size: int,
        inner_epochs: int,
        rho_pos: float,
        rho_neg: float,
        save_optim: bool = False,
        accumulation_steps: int = 1,
        use_logprobs_neg: bool = False
):
    actor, actor_tokenizer = get_parallel_model(
        model_type=actor_model_type,
        config_file=actor_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=actor_tokenizer_file,
        lora_rank=actor_lora_rank,
        dtype=dtype,
        lora_dtype=actor_lora_dtype
    )
    actor_optimizer = ParallelOptimizer(torch.optim.Adam(actor.parameters(), lr=lr))
    actor_trainer = ParallelPPOActorLogitsConvexTrainerForCausalLM(
        actor=actor,
        optimizer=actor_optimizer,
        rho_pos=rho_pos,
        rho_neg=rho_neg,
        save_optim=save_optim,
        accumulation_steps=accumulation_steps,
        use_logprobs_neg=use_logprobs_neg
    )
    actor_trainer.load_model(actor_ckpt_dir) if (
            epoch == 0
    ) else actor_trainer.load(os.path.join(actor_save_dir, "epoch-%03d" % epoch))
    print('Actor training ...')
    timer = Timer(total=(len(rollout_buffer) // actor_max_batch_size) * inner_epochs, episode=100)
    for inner_epoch in range(inner_epochs):
        for data in rollout_buffer.get(actor_max_batch_size):
            timer.step()
            trainer_outputs = actor_trainer.forward(data)
            if actor_trainer.step % 100 == 0:
                print(f'--------- STEP {actor_trainer.step} OF {timer.total} ---------')
                print('Loss: ', trainer_outputs.loss)
                print('Advantages: ', trainer_outputs.advantages)
    actor_trainer.save(os.path.join(actor_save_dir, "epoch-%03d" % (epoch + 1)))

    actor.cpu()
    del actor
    del actor_optimizer
    del actor_trainer
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()


def run(
        train_file: str,
        log_dir: str,
        actor_ckpt_dir: str,
        actor_model_type: str,
        actor_save_dir: str,
        critic_ckpt_dir: str,
        critic_model_type: str,
        critic_save_dir: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,
        actor_config_file: str = None,
        actor_tokenizer_file: str = None,
        critic_config_file: str = None,
        critic_tokenizer_file: str = None,
        verifier_config_file: str = None,
        verifier_tokenizer_file: str = None,
        reference_ckpt_dir: str = None,
        actor_lora_rank: int = -1,
        actor_lora_dtype: str = "bfloat16",
        critic_lora_rank: int = -1,
        critic_lora_dtype: str = "bfloat16",
        actor_max_batch_size: int = 1,
        critic_max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 24,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        actor_lr: float = 1e-6,
        critic_lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        kl_coef: float = 0.1,
        rho_pos: float = 1.8,
        rho_neg: float = 0.8,
        gamma: float = 0.9,
        gae_lambda: float = 0.95,
        use_chat_template: bool = False,
        use_last_token_reward: bool = False,
        last_token_reward_only: bool = False,
        save_optim: bool = False,
        accumulation_steps: int = 1,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
        seed: int = None,
):
    parallel_infos = setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        log_mode="w" if begin_epoch == 0 else "a",
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()
    actor_config_file = actor_config_file or actor_ckpt_dir
    actor_tokenizer_file = actor_tokenizer_file or actor_ckpt_dir
    critic_config_file = critic_config_file or critic_ckpt_dir
    critic_tokenizer_file = critic_tokenizer_file or critic_ckpt_dir
    verifier_config_file = verifier_config_file or verifier_ckpt_dir
    verifier_tokenizer_file = verifier_tokenizer_file or verifier_ckpt_dir

    for epoch, datalist in IterationHandler(json_load(train_file), epochs, chunk_size, begin_epoch):
        dataset = JsonDataset(datalist)
        if len(dataset) == 0:
            continue

        # Collecting actor buffer
        actor_rollout_buffer = collect_actor_buffer(
            actor_model_type=actor_model_type,
            actor_config_file=actor_config_file,
            max_seq_len=max_seq_len,
            actor_tokenizer_file=actor_tokenizer_file,
            dtype=dtype,
            actor_ckpt_dir=actor_ckpt_dir,
            epoch=epoch,
            actor_save_dir=actor_save_dir,
            use_chat_template=use_chat_template,
            dataset=dataset,
            max_generate_batch_size=max_generate_batch_size,
            temperature=temperature,
            top_p=top_p,
            num_samples_per_prompt=num_samples_per_prompt
        )

        reference_rollout_buffer = None
        if reference_ckpt_dir is not None and kl_coef != 0:
            # Collecting reference logprobs
            reference_rollout_buffer = collect_reference_buffer(
                actor_model_type=actor_model_type,
                actor_config_file=actor_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=actor_tokenizer_file,
                dtype=dtype,
                reference_ckpt_dir=reference_ckpt_dir,
                actor_rollout_buffer=actor_rollout_buffer,
                max_forward_batch_size=max_forward_batch_size
            )

        # Collecting critic buffer
        critic_rollout_buffer = collect_critic_buffer(
            critic_model_type=critic_model_type,
            critic_config_file=critic_config_file,
            max_seq_len=max_seq_len,
            critic_tokenizer_file=critic_tokenizer_file,
            dtype=dtype,
            critic_ckpt_dir=critic_ckpt_dir,
            epoch=epoch,
            critic_save_dir=critic_save_dir,
            actor_rollout_buffer=actor_rollout_buffer,
            max_forward_batch_size=max_forward_batch_size
        )

        # Collecting verifier buffer
        verifier_rollout_buffer = collect_verifier_buffer(
            verifier_model_type=verifier_model_type,
            verifier_config_file=verifier_config_file,
            max_seq_len=max_seq_len,
            verifier_tokenizer_file=verifier_tokenizer_file,
            dtype=dtype,
            verifier_ckpt_dir=verifier_ckpt_dir,
            actor_rollout_buffer=actor_rollout_buffer,
            max_forward_batch_size=max_forward_batch_size,
            use_last_token_reward=use_last_token_reward,
            last_token_reward_only=last_token_reward_only
        )

        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")

        rollout_buffer = PPORolloutBuffer(
            obs=actor_rollout_buffer["obs"],
            actions=actor_rollout_buffer["actions"],
            rewards=verifier_rollout_buffer["scores"],
            values=critic_rollout_buffer["scores"],
            action_logits=actor_rollout_buffer["action_logits"],
            action_masks=actor_rollout_buffer["action_masks"],
            action_logprobs=actor_rollout_buffer["action_logprobs"],
            ref_action_logprobs=reference_rollout_buffer.output_tokens_logps if (
                    reference_rollout_buffer is not None
            ) else None,
            use_last_token_reward=use_last_token_reward,
            last_token_reward_only=last_token_reward_only,
            kl_coef=kl_coef,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        train_actor_logits_convex(
            actor_model_type=actor_model_type,
            actor_config_file=actor_config_file,
            max_seq_len=max_seq_len,
            actor_tokenizer_file=actor_tokenizer_file,
            actor_lora_rank=actor_lora_rank,
            dtype=dtype,
            actor_lora_dtype=actor_lora_dtype,
            lr=actor_lr,
            epoch=epoch,
            actor_ckpt_dir=actor_ckpt_dir,
            actor_save_dir=actor_save_dir,
            rollout_buffer=rollout_buffer,
            actor_max_batch_size=actor_max_batch_size,
            inner_epochs=inner_epochs,
            rho_pos=rho_pos,
            rho_neg=rho_neg,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps,
            use_logprobs_neg=True
        )

        if parallel_infos.local_rank == 0:
            torch.save({
                'obs': rollout_buffer.obs,
                'actions': rollout_buffer.actions,
                'values': rollout_buffer.origin_values,
                'rewards': rollout_buffer.origin_rewards,
                'action_masks': rollout_buffer.action_masks,
                'action_logprobs': rollout_buffer.action_logprobs,
                'advantages': rollout_buffer.advantages,
            }, os.path.join(actor_save_dir, "epoch-%03d" % (epoch + 1), "buffer.bin"))

        train_critic(
            critic_model_type=critic_model_type,
            critic_config_file=critic_config_file,
            max_seq_len=max_seq_len,
            critic_tokenizer_file=critic_tokenizer_file,
            critic_lora_rank=critic_lora_rank,
            dtype=dtype,
            lr=critic_lr,
            critic_lora_dtype=critic_lora_dtype,
            critic_ckpt_dir=critic_ckpt_dir,
            epoch=0 if epoch == 0 else 1,  # TODO For saving memory
            critic_save_dir=critic_save_dir,
            rollout_buffer=rollout_buffer,
            critic_max_batch_size=critic_max_batch_size,
            inner_epochs=inner_epochs
        )


if __name__ == '__main__':
    fire.Fire(run)
