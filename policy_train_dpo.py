import gc
import os

import fire
import torch

from policy_train_ppo_with_evaluate import evaluate_actor
from src.dataset import PairwiseDataset, ChatTemplateDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.initialize import (
    setup_model_parallel,
    set_barrier,
    get_data_parallel_src_rank,
    get_data_parallel_rank
)
from src.parallel.optimizer import ParallelOptimizer
from src.ppo.buffer import LogitsRolloutBuffer
from src.ppo.collector import LogitsBufferCollector
from src.trainer import ParallelSolverDPOTrainer
from src.utils import json_load, print_current_func_args


def collect_reference_buffer(
        dataset: PairwiseDataset,
        reference_ckpt_dir: str,
        reference_model_type: str,
        reference_config_file: str,
        reference_tokenizer_file: str,
        forward_batch_size: int,
        max_seq_len: int,
        dtype: str,
        use_chat_template: bool,
        save_dir: str,
        local_epoch: int,
        reuse_buffer: bool = False
) -> (LogitsRolloutBuffer, LogitsRolloutBuffer):
    print("Reference buffer collecting ...")
    worker_id = get_data_parallel_rank()
    ref_chosen_buffer_save_dir = os.path.join(save_dir, "epoch-%03d" % local_epoch, "chosen", str(worker_id))
    ref_rejected_buffer_save_dir = os.path.join(save_dir, "epoch-%03d" % local_epoch, "rejected", str(worker_id))
    if (reuse_buffer and os.path.exists(os.path.join(ref_chosen_buffer_save_dir, "buffer.jsonl")) and
            os.path.exists(os.path.join(ref_rejected_buffer_save_dir, "buffer.jsonl"))):
        ref_chosen_rollout_buffer = LogitsRolloutBuffer()
        ref_rejected_rollout_buffer = LogitsRolloutBuffer()
        ref_chosen_rollout_buffer.load(os.path.join(ref_chosen_buffer_save_dir, "buffer.jsonl"))
        ref_rejected_rollout_buffer.load(os.path.join(ref_rejected_buffer_save_dir, "buffer.jsonl"))
    else:
        reference, reference_tokenizer = get_parallel_model(
            model_type=reference_model_type,
            config_file=reference_config_file,
            tokenizer_file=reference_tokenizer_file,
            max_seq_len=max_seq_len,
            dtype=dtype
        )
        if use_chat_template:
            dataset = ChatTemplateDataset(dataset, reference_tokenizer)
        dataloader = ParallelDataLoader(dataset, batch_size=forward_batch_size)
        reference.load(reference_ckpt_dir)
        reference_buffer_collector = LogitsBufferCollector(
            model=reference,
            tokenizer=reference_tokenizer,
            max_seq_len=max_seq_len
        )
        ref_chosen_rollout_buffer = LogitsRolloutBuffer()
        ref_rejected_rollout_buffer = LogitsRolloutBuffer()
        timer = Timer(len(dataloader), episode=10)
        for data in dataloader:
            timer.step()
            ref_chosen_rollout_buffer.extend(
                reference_buffer_collector.forward(data["instruction"], data["chosen"])
            )
            ref_rejected_rollout_buffer.extend(
                reference_buffer_collector.forward(data["instruction"], data["rejected"])
            )
        assert len(ref_chosen_rollout_buffer) == len(ref_rejected_rollout_buffer)
        if get_data_parallel_src_rank() == 0:
            ref_chosen_rollout_buffer.save(ref_chosen_buffer_save_dir)
            ref_rejected_rollout_buffer.save(ref_rejected_buffer_save_dir)

        reference.cpu()
        del reference
        del reference_buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

    return ref_chosen_rollout_buffer, ref_rejected_rollout_buffer


def train_dpo(
        ref_chosen_rollout_buffer: LogitsRolloutBuffer,
        ref_rejected_rollout_buffer: LogitsRolloutBuffer,
        policy_model_type: str,
        policy_ckpt_dir: str,
        policy_config_file: str,
        policy_tokenizer_file: str,
        save_dir: str,
        max_seq_len: int,
        max_batch_size: int,
        dtype: str,
        lora_rank: int,
        lora_dtype: str,
        beta: float,
        lr: float,
        epoch: int,
):
    print("DPO policy training ...")
    policy, policy_tokenizer = get_parallel_model(
        model_type=policy_model_type,
        config_file=policy_config_file,
        tokenizer_file=policy_tokenizer_file,
        max_seq_len=max_seq_len,
        dtype=dtype,
        lora_rank=lora_rank,
        lora_dtype=lora_dtype
    )
    optimizer = ParallelOptimizer(torch.optim.Adam(policy.parameters(), lr=lr))
    trainer = ParallelSolverDPOTrainer(
        model=policy,
        tokenizer=policy_tokenizer,
        optimizer=optimizer,
        max_seq_len=max_seq_len,
        beta=beta
    )
    policy.load(policy_ckpt_dir, merge_lora=True) if (
            epoch == 0
    ) else trainer.load(os.path.join(save_dir, "epoch-%03d" % epoch))
    timer = Timer(len(ref_chosen_rollout_buffer) // max_batch_size, episode=100)
    for chosen_data, rejected_data in zip(
            ref_chosen_rollout_buffer.get(max_batch_size),
            ref_rejected_rollout_buffer.get(max_batch_size)
    ):
        timer.step()
        trainer_outputs = trainer.forward(
            instructions=chosen_data.instructions,
            chosen=chosen_data.outputs,
            rejected=rejected_data.outputs,
            reference_chosen_log_probs=chosen_data.output_tokens_logps,
            reference_rejected_log_probs=rejected_data.output_tokens_logps
        )
        if trainer.step % 100 == 0:
            print(f'step {trainer.step} of {len(ref_chosen_rollout_buffer) // max_batch_size} ---------------')
            print(f'DPO LOSS: {trainer_outputs.loss_dpo} CE LOSS: {trainer_outputs.loss_ce}')
            trainer.predict(trainer_outputs.logits, chosen_data.instructions, chosen_data.outputs)
        if trainer.step % 10000 == 0:
            trainer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))
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
        log_dir: str,
        label_file: str,
        train_file: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_tokenizer_file: str,
        policy_config_file: str,
        max_seq_len: int = 1024,
        max_batch_size: int = 1,
        generate_batch_size: int = 1,
        forward_batch_size: int = 1,
        lr: float = 1e-6,
        dtype: str = "bfloat16",
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        chunk_size: int = None,
        beta: float = 0.1,
        epochs: int = 1,
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
        seed: int = None,
        reuse_buffer: bool = False
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

    datalist = json_load(train_file)
    chunk_size = chunk_size or len(datalist)
    local_epochs = len(datalist) // chunk_size
    begin_global_epoch = begin_epoch // local_epochs
    begin_local_epoch = begin_epoch % local_epochs
    for global_epoch in range(begin_global_epoch, epochs):
        for local_epoch in range(begin_local_epoch, local_epochs):
            epoch = local_epoch + global_epoch * local_epochs
            print(f"Epoch - {epoch} of {local_epochs * epochs}")
            dataset = PairwiseDataset(f=datalist[local_epoch * chunk_size: (local_epoch + 1) * chunk_size])
            if len(dataset) == 0:
                continue

            # Reference model logprobs collecting ...
            ref_chosen_rollout_buffer, ref_rejected_rollout_buffer = collect_reference_buffer(
                dataset=dataset,
                reference_ckpt_dir=policy_ckpt_dir,
                reference_model_type=policy_model_type,
                reference_config_file=policy_config_file,
                reference_tokenizer_file=policy_tokenizer_file,
                forward_batch_size=forward_batch_size,
                max_seq_len=max_seq_len,
                dtype=dtype,
                use_chat_template=use_chat_template,
                save_dir=save_dir,
                local_epoch=local_epoch,
                reuse_buffer=reuse_buffer
            )

            # policy DPO training ...
            train_dpo(
                ref_chosen_rollout_buffer=ref_chosen_rollout_buffer,
                ref_rejected_rollout_buffer=ref_rejected_rollout_buffer,
                policy_model_type=policy_model_type,
                policy_ckpt_dir=policy_ckpt_dir,
                policy_config_file=policy_config_file,
                policy_tokenizer_file=policy_tokenizer_file,
                save_dir=save_dir,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                dtype=dtype,
                lora_rank=lora_rank,
                lora_dtype=lora_dtype,
                beta=beta,
                lr=lr,
                epoch=epoch
            )

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
                max_generate_batch_size=generate_batch_size,
                use_chat_template=use_chat_template
            )


if __name__ == '__main__':
    fire.Fire(run)
