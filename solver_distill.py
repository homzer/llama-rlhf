import gc
import os

import fire
import torch
from torch.utils.data import DataLoader

from src.dataset import MultiOutputsDataset, JsonDataset
from src.entities import Timer
from src.evaluator import PolicyEvaluator
from src.modeling import get_parallel_model
from src.ppo.buffer import LogitsRolloutBuffer
from src.ppo.collector import LogitsBufferCollector
from src.trainer import ParallelSolverDistillTrainer
from src.utils import json_dump
from src.parallel.initialize import setup_model_parallel, set_barrier


def main(
        train_file: str,

        student_ckpt_dir: str,
        student_save_dir: str,
        student_model_type: str,
        student_lora_rank: int,
        student_tokenizer_file: str,
        student_config_file: str,

        teacher_ckpt_dir: str,
        teacher_model_type: str,
        teacher_tokenizer_file: str,
        teacher_config_file: str,
        teacher_forward_batch_size: int,

        student_max_seq_len: int = 512,
        teacher_max_seq_len: int = 512,
        alpha: float = 1.0,
        beta: float = 1.0,
        T: float = 1.0,
        task: str = None,
        label_file: str = None,
        eval_batch_size: int = None,
        max_batch_size: int = 1,
        lr: float = 1e-5,
        epochs: int = 1,
        log_dir: str = None,
        seed: int = None
):
    if task is not None:
        assert label_file is not None
        assert eval_batch_size is not None
        assert log_dir is not None
        os.makedirs(log_dir, exist_ok=True)
    setup_model_parallel(seed=seed)
    dataset = MultiOutputsDataset(train_file)
    dataloader = DataLoader(dataset, batch_size=teacher_forward_batch_size)

    for epoch in range(epochs):
        teacher, teacher_tokenizer = get_parallel_model(
            model_type=teacher_model_type,
            config_file=teacher_config_file,
            max_seq_len=teacher_max_seq_len,
            tokenizer_file=teacher_tokenizer_file,
            lora_rank=-1
        )
        teacher.load(teacher_ckpt_dir, merge_lora=True)
        buffer_collector = LogitsBufferCollector(teacher, teacher_tokenizer, teacher_max_seq_len, logits_topk=5)
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
            max_seq_len=student_max_seq_len,
            tokenizer_file=student_tokenizer_file,
            lora_rank=student_lora_rank
        )
        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
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
            outputs = trainer.forward(
                instructions=data.instructions,
                outputs=data.outputs,
                target_logits=data.logits,
                kl_coef=alpha,
                beta=beta,
                temperature=T
            )
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {len(rollout_buffer) // max_batch_size} ---------------')
                print(f'CE LOSS: ', outputs.loss_ce.item(), 'KL LOSS: ', outputs.loss_kl.item())
                trainer.predict(outputs.logits, data.instructions, data.outputs)
            if trainer.step % 7200 == 0:
                trainer.save(os.path.join(student_save_dir, f"epoch-{epoch + 1}"))
        trainer.save(os.path.join(student_save_dir, f"epoch-{epoch + 1}"))

        if task is not None:
            evaluator = PolicyEvaluator(student, student_tokenizer, eval_batch_size, student_max_seq_len)
            eval_outputs = evaluator.forward(task, JsonDataset(label_file))
            print("Evaluate Accuracy: ", eval_outputs.acc, "Missing: ", eval_outputs.missing)
            json_dump(eval_outputs.datalist, os.path.join(
                log_dir, f'results-epoch-{epoch + 1}-{round(eval_outputs.acc, 4)}.json'), indent=4
            )

        student.cpu()
        del student
        del optimizer
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()


if __name__ == '__main__':
    fire.Fire(main)
