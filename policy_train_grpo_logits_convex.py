import gc
import os

import fire
import torch

from policy_train_grpo import compute_grpo_rewards
from policy_train_ppo import collect_reference_buffer, collect_verifier_buffer, collect_actor_buffer
from src.dataset import JsonDataset
from src.entities import IterationHandler, Timer
from src.modeling import get_parallel_model
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.ppo.buffer import RolloutBuffer
from src.ppo.trainer_logits_convex import ParallelGRPOLogitsConvexTrainerForCausalLM
from src.utils import json_load, print_current_func_args


def train_grpo_logits_convex(
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
        rho_pos: float,
        rho_neg: float,
        kl_coef: float = 0.04,
        save_optim: bool = False,
        accumulation_steps: int = 1,
        use_logprobs_neg: bool = False
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
    trainer = ParallelGRPOLogitsConvexTrainerForCausalLM(
        policy=policy,
        optimizer=optimizer,
        rho_pos=rho_pos,
        rho_neg=rho_neg,
        kl_coef=kl_coef,
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
                print(f'Policy Loss: {trainer_outputs.policy_loss}')
                print(f'Rewards: {trainer_outputs.rewards}')
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
        train_file: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,
        reference_ckpt_dir: str = None,
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
        num_samples_per_prompt: int = 4,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-6,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        kl_coef: float = 0.04,
        rho_pos: float = 1.8,
        rho_neg: float = 0.9,
        use_chat_template: bool = False,
        log_dir: str = None,
        seed: int = None,
        accumulation_steps: int = 1,
        save_optim: bool = False,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1
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
    verifier_config_file = verifier_config_file or verifier_ckpt_dir
    verifier_tokenizer_file = verifier_tokenizer_file or verifier_ckpt_dir

    for epoch, datalist in IterationHandler(json_load(train_file), epochs, chunk_size, begin_epoch):
        dataset = JsonDataset(datalist)
        if len(dataset) == 0:
            continue

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

        reference_rollout_buffer = None
        if reference_ckpt_dir is not None:
            reference_rollout_buffer = collect_reference_buffer(
                actor_model_type=policy_model_type,
                actor_config_file=policy_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=policy_tokenizer_file,
                dtype=dtype,
                reference_ckpt_dir=reference_ckpt_dir,
                actor_rollout_buffer=policy_rollout_buffer,
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
            actor_rollout_buffer=policy_rollout_buffer,
            max_forward_batch_size=max_forward_batch_size,
            use_last_token_reward=True
        )

        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")

        # Compute GRPO rewards
        policy_rollout_buffer, verifier_rollout_buffer = compute_grpo_rewards(
            verifier_rollout_buffer=verifier_rollout_buffer,
            policy_rollout_buffer=policy_rollout_buffer,
            num_samples_per_prompt=num_samples_per_prompt
        )

        if reference_rollout_buffer is not None:
            rollout_buffer = RolloutBuffer(
                obs=policy_rollout_buffer["obs"],
                actions=policy_rollout_buffer["actions"],
                rewards=verifier_rollout_buffer["scores"],
                action_masks=policy_rollout_buffer["action_masks"],
                action_logprobs=policy_rollout_buffer["action_logprobs"],
                ref_action_logprobs=reference_rollout_buffer.output_tokens_logps
            )
        else:
            rollout_buffer = RolloutBuffer(
                obs=policy_rollout_buffer["obs"],
                actions=policy_rollout_buffer["actions"],
                rewards=verifier_rollout_buffer["scores"],
                action_masks=policy_rollout_buffer["action_masks"],
                action_logprobs=policy_rollout_buffer["action_logprobs"]
            )

        train_grpo_logits_convex(
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
            kl_coef=kl_coef,
            rho_pos=rho_pos,
            rho_neg=rho_neg,
            accumulation_steps=accumulation_steps,
            save_optim=save_optim,
            use_logprobs_neg=True
        )


if __name__ == '__main__':
    fire.Fire(run)
