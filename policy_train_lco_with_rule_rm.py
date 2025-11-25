import os

import fire
import numpy as np

from policy_train_lco import collect_logits_buffer, train_lco
from policy_train_ppo_with_evaluate import evaluate_actor
from policy_train_ppo_with_rule_rm import collect_actor_buffer_with_label, collect_rule_based_verifier_buffer
from src.dataset import JsonDataset
from src.entities import IterationHandler
from src.parallel.initialize import setup_model_parallel
from src.utils import json_load, print_current_func_args


def run(
        task: str,
        train_file: str,
        label_file: str,
        save_dir: str,
        log_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 24,
        max_seq_len: int = 1024,
        beta: float = 10.0,
        estimate_with_beta: bool = False,
        logits_topk: int = 5,
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
        max_num_ckpts: int = None,
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

        # collecting policy buffer
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

        # collecting logits buffer
        logits_rollout_buffer = collect_logits_buffer(
            policy_rollout_buffer=policy_rollout_buffer,
            policy_model_type=policy_model_type,
            policy_ckpt_dir=policy_ckpt_dir,
            policy_config_file=policy_config_file,
            policy_tokenizer_file=policy_tokenizer_file,
            max_seq_len=max_seq_len,
            max_forward_batch_size=max_forward_batch_size,
            dtype=dtype,
            logits_topk=logits_topk
        )

        # collecting verifier buffer
        verifier_rollout_buffer = collect_rule_based_verifier_buffer(
            actor_rollout_buffer=policy_rollout_buffer, task=task
        )

        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")
        action_probs = np.exp(policy_rollout_buffer["action_logprobs"][policy_rollout_buffer["action_masks"]]).mean()
        print(f"Average Action Probs: {action_probs}")

        logits_rollout_buffer["advantages"] = verifier_rollout_buffer["scores"]

        train_lco(
            rollout_buffer=logits_rollout_buffer,
            policy_ckpt_dir=policy_ckpt_dir,
            policy_model_type=policy_model_type,
            policy_config_file=policy_config_file,
            policy_tokenizer_file=policy_tokenizer_file,
            max_seq_len=max_seq_len,
            dtype=dtype,
            lora_rank=lora_rank,
            lora_dtype=lora_dtype,
            lr=lr,
            epoch=epoch,
            inner_epochs=inner_epochs,
            save_dir=save_dir,
            max_batch_size=max_batch_size,
            beta=beta,
            estimate_with_beta=estimate_with_beta,
            max_num_ckpts=max_num_ckpts,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )

        if parallel_infos.global_rank == 0:
            logits_rollout_buffer.save(os.path.join(log_dir, "epoch-%03d" % (epoch + 1)))

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
