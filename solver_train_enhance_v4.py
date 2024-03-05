import gc
import os

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import JsonDataset, MultiOutputsDataset
from src.entities import Timer
from src.evaluator import SolverEvaluator
from src.models.modeling_utils import get_parallel_model
from src.ppo.buffer import SolverRolloutBuffer, LogitsRolloutBuffer
from src.ppo.collector import SolverBufferCollector, LogitsBufferCollector
from src.trainer import ParallelSolverDistillTrainer
from src.utils import setup_model_parallel, json_dump, set_barrier, json_load


def run(
        task: str,
        train_file: str,
        label_file: str,
        solver_model_type: str,
        solver_ckpt_dir: str,
        solver_config_file: str,
        solver_tokenizer_file: str,
        solver_save_dir: str,
        solver_max_seq_len: int,
        solver_lora_rank: int,
        solver_eval_batch_size: int,
        reviser_model_type: str,
        reviser_ckpt_dir: str,
        reviser_config_file: str,
        reviser_tokenizer_file: str,
        reviser_max_seq_len: int,
        reviser_eval_batch_size: int,
        log_dir: str = None,
        max_batch_size: int = 6,
        epochs: int = 1,
        lr: float = 1e-5,
        offset: int = 0
):
    local_rank, world_size = setup_model_parallel(use_float16=True)
    if log_dir is not None and local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
    dataset = MultiOutputsDataset(json_load(train_file)[offset:])
    for epoch in range(epochs):
        solver_model, solver_tokenizer = get_parallel_model(
            model_type=solver_model_type,
            config_file=solver_config_file,
            local_rank=local_rank,
            world_size=world_size,
            max_seq_len=solver_max_seq_len,
            tokenizer_file=solver_tokenizer_file,
            lora_rank=-1
        )
        solver_model.load(solver_ckpt_dir if (
            epoch == 0
        ) else os.path.join(solver_save_dir, f"epoch-{epoch}"), merge_lora=True)
        # Solver Model Evaluation
        evaluator = SolverEvaluator(
            model=solver_model,
            tokenizer=solver_tokenizer,
            batch_size=solver_eval_batch_size,
            max_seq_len=solver_max_seq_len
        )
        evaluator_outputs = evaluator.forward(task, JsonDataset(label_file))
        print("Evaluate Accuracy: ", evaluator_outputs.acc, "Missing: ", evaluator_outputs.missing)
        if log_dir is not None:
            json_dump(evaluator_outputs.datalist, os.path.join(
                log_dir, f'results-epoch-{epoch}-{round(evaluator_outputs.acc, 4)}.json'
            ), indent=4)
        # Solver Model Collection
        solver_buffer_collector = SolverBufferCollector(solver_model, solver_tokenizer, solver_max_seq_len)
        solver_rollout_buffer = SolverRolloutBuffer()
        print('Solver buffer collecting ...')
        eval_dataloader = DataLoader(dataset, batch_size=solver_eval_batch_size)
        timer = Timer(len(eval_dataloader))
        for data in tqdm(eval_dataloader):
            timer.step()
            solver_rollout_buffer.extend(
                solver_buffer_collector.forward(data['instruction'], t=1.2)
            )
            print(data['instruction'][-1] + solver_tokenizer.decode(
                solver_rollout_buffer.actions[-1][solver_rollout_buffer.action_masks[-1]].tolist()
            ))

        solver_model.cpu()
        del solver_model
        del solver_buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        # Reviser Model Collection
        reviser_model, reviser_tokenizer = get_parallel_model(
            model_type=reviser_model_type,
            config_file=reviser_config_file,
            local_rank=local_rank,
            world_size=world_size,
            max_seq_len=reviser_max_seq_len,
            tokenizer_file=reviser_tokenizer_file,
            lora_rank=-1
        )
        reviser_model.load(reviser_ckpt_dir, merge_lora=True)
        reviser_buffer_collector = LogitsBufferCollector(reviser_model, reviser_tokenizer, reviser_max_seq_len)
        reviser_rollout_buffer = LogitsRolloutBuffer()
        timer = Timer(len(solver_rollout_buffer) // reviser_eval_batch_size, episode=20)
        for data in solver_rollout_buffer.get(reviser_eval_batch_size):
            timer.step()
            outputs = []
            for action, action_mask in zip(data.actions, data.action_masks):
                outputs.append(reviser_tokenizer.decode(action[action_mask].tolist()))
            reviser_rollout_buffer.extend(
                reviser_buffer_collector.forward(data.instructions, outputs)
            )

        reviser_model.cpu()
        del reviser_model
        del reviser_buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        solver_model, solver_tokenizer = get_parallel_model(
            model_type=solver_model_type,
            config_file=solver_config_file,
            local_rank=local_rank,
            world_size=world_size,
            max_seq_len=solver_max_seq_len,
            tokenizer_file=solver_tokenizer_file,
            lora_rank=solver_lora_rank
        )
        optimizer = torch.optim.Adam(solver_model.parameters(), lr=lr)
        trainer = ParallelSolverDistillTrainer(
            model=solver_model,
            tokenizer=solver_tokenizer,
            optimizer=optimizer,
            max_seq_len=solver_max_seq_len
        )
        solver_model.load(solver_ckpt_dir, merge_lora=True) if (
                epoch == 0
        ) else trainer.load(os.path.join(solver_save_dir, f"epoch-{epoch}"))
        print('Solver Training ...')
        assert len(solver_rollout_buffer) == len(reviser_rollout_buffer)
        for reviser_data in reviser_rollout_buffer.get(max_batch_size):
            trainer_outputs = trainer.distill(
                instructions=reviser_data.instructions,
                outputs=reviser_data.outputs,
                target_logits=reviser_data.logits,
                alpha=0.0
            )
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {len(solver_rollout_buffer) // max_batch_size} ---------------')
                print(f'CE LOSS: ', trainer_outputs.loss_ce.item(), 'KL LOSS: ', trainer_outputs.loss_kl.item())
                predict = trainer.predict(trainer_outputs.logits, reviser_data.instructions, reviser_data.outputs)[0]
                print(predict['instruction'] + predict['output'])

        trainer.save(os.path.join(solver_save_dir, f"epoch-{epoch + 1}"))

        solver_model.cpu()
        del solver_model
        del optimizer
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()


if __name__ == '__main__':
    fire.Fire(run)
