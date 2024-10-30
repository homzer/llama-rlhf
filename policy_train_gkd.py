import gc
import os

import fire
import torch
from torch.utils.data import DataLoader

from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.parallel.utils import setup_model_parallel, set_barrier
from src.ppo.buffer import ActorRolloutBuffer, LogitsRolloutBuffer
from src.ppo.collector import ActorBufferCollector, LogitsBufferCollector
from src.trainer import ParallelSolverDistillTrainer
from src.utils import json_load


def run(
        train_file: str,
        save_dir: str,
        student_ckpt_dir: str,
        student_model_type: str,
        teacher_ckpt_dir: str,
        teacher_model_type: str,
        student_config_file: str = None,
        student_tokenizer_file: str = None,
        teacher_config_file: str = None,
        teacher_tokenizer_file: str = None,
        logits_topk: int = 5,
        kl_coef: float = 1.0,
        ce_coef: float = 1.0,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        generate_batch_size: int = 48,
        forward_batch_size: int = 24,
        max_seq_len: int = 1024,
        chunk_size: int = None,
        inner_epochs: int = 3,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False
):
    setup_model_parallel()
    os.makedirs(save_dir, exist_ok=True)
    student_config_file = student_config_file or student_ckpt_dir
    student_tokenizer_file = student_tokenizer_file or student_ckpt_dir
    teacher_config_file = teacher_config_file or teacher_ckpt_dir
    teacher_tokenizer_file = teacher_tokenizer_file or teacher_ckpt_dir

    datalist = json_load(train_file)
    chunk_size = chunk_size or len(datalist)
    epochs = len(datalist) // chunk_size
    for epoch in range(begin_epoch, epochs):
        print(f"Epoch - {epoch} of {epochs}")
        dataset = JsonDataset(f=datalist[epoch * chunk_size: (epoch + 1) * chunk_size])
        student, student_tokenizer = get_parallel_model(
            model_type=student_model_type,
            config_file=student_config_file,
            max_seq_len=max_seq_len,
            tokenizer_file=student_tokenizer_file,
            lora_rank=-1,
            dtype=dtype
        )
        student.load(student_ckpt_dir if epoch == 0 else os.path.join(save_dir, f"epoch-{epoch}"))
        student_buffer_collector = ActorBufferCollector(student, student_tokenizer, max_seq_len, temperature=1.1)
        student_rollout_buffer = ActorRolloutBuffer()
        print("Student buffer collecting ...")
        if use_chat_template:
            dataset = ChatTemplateDataset(dataset, student_tokenizer)
        dataloader = DataLoader(dataset, batch_size=generate_batch_size)
        timer = Timer(len(dataloader))
        for data in dataloader:
            timer.step()
            student_rollout_buffer.extend(student_buffer_collector.forward(data['instruction']))
            print(data['instruction'][-1])
            print(student_rollout_buffer.responses[-1])

        student.cpu()
        del student
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        teacher, teacher_tokenizer = get_parallel_model(
            model_type=teacher_model_type,
            config_file=teacher_config_file,
            max_seq_len=max_seq_len,
            tokenizer_file=teacher_tokenizer_file,
            lora_rank=-1,
            dtype=dtype
        )
        teacher.load(teacher_ckpt_dir)
        teacher_buffer_collector = LogitsBufferCollector(
            teacher, teacher_tokenizer, max_seq_len, logits_topk=logits_topk
        )
        teacher_rollout_buffer = LogitsRolloutBuffer()
        print("Teacher buffer collecting ...")
        timer = Timer(total=len(student_rollout_buffer) // forward_batch_size, episode=10)
        for data in student_rollout_buffer.get(forward_batch_size):
            timer.step()
            teacher_rollout_buffer.extend(
                teacher_buffer_collector.forward(data.instructions, data.responses)
            )

        teacher.cpu()
        del teacher
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        student, student_tokenizer = get_parallel_model(
            model_type=student_model_type,
            config_file=student_config_file,
            max_seq_len=max_seq_len,
            tokenizer_file=student_tokenizer_file,
            lora_rank=lora_rank,
            lora_dtype=lora_dtype,
            dtype=dtype
        )
        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        trainer = ParallelSolverDistillTrainer(student, student_tokenizer, optimizer, max_seq_len)
        trainer.load_model(student_ckpt_dir) if (
            epoch == 0
        ) else trainer.load(os.path.join(save_dir, f"epoch-{epoch}"))
        print("Student training ...")
        timer = Timer(total=(len(teacher_rollout_buffer) // max_batch_size) * inner_epochs, episode=100)
        for inner_epoch in range(inner_epochs):
            for data in teacher_rollout_buffer.get(max_batch_size):
                timer.step()
                trainer_outputs = trainer.distill(
                    instructions=data.instructions,
                    outputs=data.outputs,
                    target_logits=data.logits,
                    kl_coef=kl_coef,
                    ce_coef=ce_coef
                )
                if trainer.step % 100 == 0:
                    print(f'--------- STEP {trainer.step} OF {timer.total} ---------')
                    print('Loss: ', trainer_outputs.loss)
                    print(f'CE Loss: {trainer_outputs.loss_ce} | KL Loss: {trainer_outputs.loss_kl}')
        trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))

        student.cpu()
        del student
        del optimizer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()


if __name__ == '__main__':
    fire.Fire(run)
