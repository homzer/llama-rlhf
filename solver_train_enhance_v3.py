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
from src.ppo.buffer import SolverRolloutBuffer, LogitsRolloutBufferV0
from src.ppo.collector import SolverBufferCollector, LogitsBufferCollectorV0
from src.trainer import ParallelSolverDistillTrainer
from src.utils import setup_model_parallel, json_dump, set_barrier, json_load


def run(
        task: str,
        solver_model_type: str,
        solver_ckpt_dir: str,
        solver_config_file: str,
        solver_tokenizer_file: str,
        solver_save_dir: str,
        solver_lora_rank: int,
        reviser_ckpt_dir: str,
        reviser_config_file: str,
        reviser_tokenizer_file: str,
        reviser_model_type: str,
        train_file: str,
        label_file: str,
        log_dir: str = None,
        max_batch_size: int = 6,
        solver_eval_batch_size: int = 384,
        reviser_eval_batch_size: int = 196,
        solver_max_seq_len: int = 512,
        reviser_max_seq_len: int = 512,
        max_seq_len: int = 512,
        epochs: int = 1,
        lr: float = 1e-5,
        use_float16: bool = True,
        offset: int = 0,
        begin_epoch: int = 0
):
    local_rank, world_size = setup_model_parallel(use_float16=use_float16)
    if log_dir is not None and local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)

    dataset = MultiOutputsDataset(json_load(train_file)[offset:])

    for epoch in range(begin_epoch, begin_epoch + epochs):
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
        evaluator = SolverEvaluator(solver_model, solver_tokenizer, solver_eval_batch_size, max_seq_len)
        outputs = evaluator.forward(task, JsonDataset(label_file))
        print("Evaluate Accuracy: ", outputs.acc, "Missing: ", outputs.missing)
        if log_dir is not None:
            json_dump(outputs.datalist, os.path.join(
                log_dir, f'results-epoch-{epoch}-{round(outputs.acc, 4)}.json'
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
        reviser_buffer_collector = LogitsBufferCollectorV0(reviser_model, reviser_tokenizer, max_seq_len)
        reviser_rollout_buffer = LogitsRolloutBufferV0()
        timer = Timer(len(solver_rollout_buffer) // reviser_eval_batch_size, episode=20)
        for data in solver_rollout_buffer.get(reviser_eval_batch_size):
            timer.step()
            instructions = []
            for instruction, action, action_mask in zip(data.instructions, data.actions, data.action_masks):
                instructions.append(instruction + " " + reviser_tokenizer.decode(action[action_mask].tolist()))
            reviser_rollout_buffer.extend(
                reviser_buffer_collector.forward(instructions)
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
        trainer = ParallelSolverDistillTrainer(solver_model, solver_tokenizer, optimizer, max_seq_len)
        solver_model.load(solver_ckpt_dir, merge_lora=True) if (
                epoch == 0
        ) else trainer.load(os.path.join(solver_save_dir, f"epoch-{epoch}"))
        print('Solver Training ...')
        assert len(solver_rollout_buffer) == len(reviser_rollout_buffer)
        for solver_data, reviser_data in zip(
                solver_rollout_buffer.get(max_batch_size),
                reviser_rollout_buffer.get(max_batch_size)
        ):
            outputs = []
            for action, action_mask in zip(solver_data.actions, solver_data.action_masks):
                outputs.append(solver_tokenizer.decode(action[action_mask].tolist()))
            assert len(outputs) == len(reviser_data.logits), f"{len(outputs)} {len(reviser_data.logits)}"  # 1, 6
            trainer_outputs = trainer.distill(
                instructions=solver_data.instructions,
                outputs=outputs,
                target_logits=reviser_data.logits,
                alpha=0.0
            )
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {len(solver_rollout_buffer) // max_batch_size} ---------------')
                print(f'KL LOSS: ', trainer_outputs.loss_kl.item())
                predict = trainer.predict(trainer_outputs.logits, solver_data.instructions, outputs)[0]
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
