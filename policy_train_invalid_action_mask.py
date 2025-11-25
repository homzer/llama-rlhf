import gc
import os
import shutil

import fire
import torch.cuda

from policy_train_gkd import collect_logits_buffer
from policy_train_ppo_with_evaluate import evaluate_actor
from policy_train_ppo_with_rule_rm import collect_actor_buffer_with_label, collect_rule_based_verifier_buffer
from src.dataset import JsonDataset
from src.entities import IterationHandler, Timer
from src.modeling import get_parallel_model
from src.parallel.initialize import setup_model_parallel, set_barrier, get_rank
from src.parallel.optimizer import ParallelOptimizer
from src.ppo.buffer import LogitsRolloutBuffer
from src.ppo.trainer import ParallelInvalidActionMaskTrainerForCausalLM
from src.utils import json_load, print_current_func_args


def train_invalid_action_mask(
        rollout_buffer: LogitsRolloutBuffer,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_config_file: str,
        policy_tokenizer_file: str,
        max_seq_len: int,
        dtype: str,
        lora_rank: int,
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
    trainer = ParallelInvalidActionMaskTrainerForCausalLM(
        policy=policy,
        optimizer=optimizer,
        save_optim=save_optim,
        accumulation_steps=accumulation_steps,
    )
    trainer.load_model(policy_ckpt_dir) if (
            epoch == 0
    ) else trainer.load(os.path.join(save_dir, "epoch-%03d" % epoch))
    print("Policy training ...")
    timer = Timer(total=(rollout_buffer.size() // max_batch_size) * inner_epochs, episode=100)
    for inner_epoch in range(inner_epochs):
        for data in rollout_buffer.get(max_batch_size, shuffle=True, output_tensor=True):
            timer.step()
            trainer_outputs = trainer.forward(data)
            if trainer.step % 100 == 0:
                print(f'--------- STEP {trainer.step} OF {timer.total} ---------')
                print(f'Loss: {trainer_outputs.loss}')
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
        log_dir: str,
        save_dir: str,
        train_file: str,
        label_file: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_config_file: str,
        policy_tokenizer_file: str,
        teacher_ckpt_dir: str,
        teacher_model_type: str,
        teacher_config_file: str,
        teacher_tokenizer_file: str,
        max_batch_size: int,
        max_generate_batch_size: int,
        max_forward_batch_size: int,
        max_seq_len: int,
        logits_topk: int = 5,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-5,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
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
    teacher_config_file = teacher_config_file or teacher_ckpt_dir
    teacher_tokenizer_file = teacher_tokenizer_file or teacher_ckpt_dir

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

        verifier_rollout_buffer = collect_rule_based_verifier_buffer(
            actor_rollout_buffer=policy_rollout_buffer, task=task
        )

        # collecting logits buffer
        logits_rollout_buffer = collect_logits_buffer(
            policy_rollout_buffer=policy_rollout_buffer,
            teacher_model_type=teacher_model_type,
            teacher_ckpt_dir=teacher_ckpt_dir,
            teacher_config_file=teacher_config_file,
            teacher_tokenizer_file=teacher_tokenizer_file,
            max_seq_len=max_seq_len,
            max_forward_batch_size=max_forward_batch_size,
            dtype=dtype,
            logits_topk=logits_topk
        )

        logits_rollout_buffer["rewards"] = verifier_rollout_buffer["scores"]

        train_invalid_action_mask(
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
            save_optim=save_optim,
            accumulation_steps=accumulation_steps,
            max_num_ckpts=max_num_ckpts
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
