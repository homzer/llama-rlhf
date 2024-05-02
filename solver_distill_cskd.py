import gc
import os
import random
from typing import Union

import fire
import torch
from torch.utils.data import DataLoader

from src.dataset import MultiOutputsDataset
from src.entities import Timer, AverageMeter
from src.models.modeling_utils import get_parallel_model
from src.ppo.buffer import LogitsRolloutBuffer
from src.ppo.collector import LogitsBufferCollector
from src.trainer import ParallelSolverReferenceDistillTrainer, ParallelSolverDistillTrainer
from src.utils import setup_model_parallel, set_barrier, json_load


def main(
        train_file: str,

        student_ckpt_dir: str,
        student_save_dir: str,
        student_model_type: str,
        student_lora_rank: int,
        student_tokenizer_file: str,
        student_config_file: str,

        teacher_ckpt_dir: Union[str, list],
        teacher_model_type: str,
        teacher_tokenizer_file: str,
        teacher_config_file: str,
        teacher_forward_batch_size: int,

        student_max_seq_len: int = 1024,
        teacher_max_seq_len: int = 1024,
        alpha: float = 1.0,
        T: float = 1.0,
        max_batch_size: int = 1,
        lr: float = 1e-5,
        chunk_size: int = 10000,
        seed: int = None,
        dtype: str = "float16",
        lora_dtype: str = "float32",
        use_reference: bool = False
):
    teacher_ckpt_dir = [teacher_ckpt_dir] if isinstance(teacher_ckpt_dir, str) else list(teacher_ckpt_dir)
    local_rank, world_size = setup_model_parallel(seed=seed)
    datalist = json_load(train_file)
    epochs = len(datalist) // chunk_size
    for epoch in range(epochs):
        print(f"Epoch - {epoch} of {epochs}")
        dataset = MultiOutputsDataset(datalist[epoch * chunk_size: (epoch + 1) * chunk_size])
        if len(dataset) == 0:
            return
        dataloader = DataLoader(dataset, batch_size=teacher_forward_batch_size)

        teacher, teacher_tokenizer = get_parallel_model(
            model_type=teacher_model_type,
            config_file=teacher_config_file,
            local_rank=local_rank,
            world_size=world_size,
            max_seq_len=teacher_max_seq_len,
            tokenizer_file=teacher_tokenizer_file,
            lora_rank=-1,
            dtype=dtype,
            lora_dtype=lora_dtype
        )
        # randomly sample a teacher checkpoint
        teacher.load(random.sample(teacher_ckpt_dir, 1)[0], merge_lora=True)
        buffer_collector = LogitsBufferCollector(teacher, teacher_tokenizer, teacher_max_seq_len)
        rollout_buffer = LogitsRolloutBuffer()
        timer = Timer(len(dataloader), episode=10)
        for data in dataloader:
            timer.step()
            rollout_buffer.extend(
                buffer_collector.forward(data['instruction'], data['output'])
            )

        teacher.cpu()
        del teacher
        del buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        student, student_tokenizer = get_parallel_model(
            model_type=student_model_type,
            config_file=student_config_file,
            local_rank=local_rank,
            world_size=world_size,
            max_seq_len=student_max_seq_len,
            tokenizer_file=student_tokenizer_file,
            lora_rank=student_lora_rank,
            dtype=dtype,
            lora_dtype=lora_dtype
        )
        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        ref_logps = None
        if use_reference:
            # compute reference point
            print("Computing reference log probs ......")
            meter = AverageMeter()
            timer = Timer(len(rollout_buffer), episode=64)
            for data in rollout_buffer.get(1):
                timer.step()
                # with average logps
                for logps in (data.output_tokens_logps.sum(-1) / ((data.output_tokens_logps != 0).sum(-1) + 1e-12)):
                    logps = logps.item()
                    if logps != 0:  # omit for zero
                        meter.forward(logps)
            ref_logps = meter.avg
            print("Reference log probs: ", ref_logps, type(ref_logps))
            trainer = ParallelSolverReferenceDistillTrainer(
                model=student,
                tokenizer=student_tokenizer,
                optimizer=optimizer,
                max_seq_len=student_max_seq_len
            )
        else:
            trainer = ParallelSolverDistillTrainer(
                model=student,
                tokenizer=student_tokenizer,
                optimizer=optimizer,
                max_seq_len=student_max_seq_len
            )
        trainer.load(student_ckpt_dir) if (
                epoch == 0
        ) else trainer.load(os.path.join(student_save_dir, f"epoch-{epoch}"))
        timer = Timer(len(rollout_buffer) // max_batch_size, episode=100)
        for data in rollout_buffer.get(max_batch_size):
            timer.step()
            if use_reference:
                outputs = trainer.distill(
                    instructions=data.instructions,
                    outputs=data.outputs,
                    target_logits=data.logits,
                    ref_logps=ref_logps,
                    alpha=alpha,
                    temperature=T
                )
                if trainer.step % 100 == 0:
                    print(f'step {trainer.step} of {len(rollout_buffer) // max_batch_size} ---------------')
                    print(f'CE LOSS: ', outputs.loss_ce.item(), 'KL LOSS: ', outputs.loss_kl.item())
                    print(f'References: ', outputs.references)
                    print('Ref Log Probs: ', ref_logps, 'Target Log Probs: ', outputs.target_logps)
                    trainer.predict(outputs.logits, data.instructions, data.outputs)
            else:
                outputs = trainer.distill(
                    instructions=data.instructions,
                    outputs=data.outputs,
                    target_logits=data.logits,
                    beta=alpha,
                    temperature=T
                )
                if trainer.step % 100 == 0:
                    print(f'step {trainer.step} of {len(rollout_buffer) // max_batch_size} ---------------')
                    print(f'CE LOSS: ', outputs.loss_ce.item(), 'KL LOSS: ', outputs.loss_kl.item())
                    trainer.predict(outputs.logits, data.instructions, data.outputs)
        trainer.save(os.path.join(student_save_dir, f"epoch-{epoch + 1}"))

        student.cpu()
        del student
        del optimizer
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()


if __name__ == '__main__':
    fire.Fire(main)
