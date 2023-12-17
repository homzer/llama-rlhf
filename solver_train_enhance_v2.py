import gc
import os
import random

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import JsonDataset, MultiOutputsDataset
from src.evaluator import SolverEvaluator
from src.modeling.llama import Llama
from src.modeling.llama_lora import LoraLlama
from src.modeling.modeling_args import LoraLlamaArgs, LlamaArgs
from src.ppo.buffer import CriticRolloutBuffer, SolverRolloutBuffer
from src.ppo.collector import SolverBufferCollector, LabelBufferCollector
from src.tokenizer import LlamaTokenizer
from src.trainer import ParallelSolverTrainer
from src.utils import setup_model_parallel, Timer, json_dump, set_barrier, deduplicate_texts

RETHINKING = '\n\n<|rethinking|>\n\n'


class EnhanceDataset(MultiOutputsDataset):
    def __init__(self, train_file):
        super().__init__(train_file)
        self.map = {}
        for i, data in enumerate(self.datalist):
            self.map[data['instruction']] = i
        assert len(list(self.map.keys())) == len(self.datalist)

    def enhance(self, datalist: list) -> int:
        cnt = self.statistic()
        for data in datalist:
            assert data['instruction'] in self.map.keys()
            i = self.map[data['instruction']]
            self.datalist[i]['output'].append(data['output'])
            self.datalist[i]['output'] = deduplicate_texts(self.datalist[i]['output'])
        return self.statistic() - cnt

    def __getitem__(self, i):
        data = self.datalist[i].copy()
        outputs = []
        b = len(data['output'])
        for a in range(b):  # Bigger chances for later outputs
            if random.randint(a + 1, b) == b:
                outputs.append(data['output'][a])
        assert len(outputs) != 0
        data['output'] = random.sample(outputs, 1)[0]
        return data

    def statistic(self) -> int:
        """ Return the total number of outputs. """
        cnt = 0
        for data in self.datalist:
            cnt += len(data['output'])
        return cnt


def run(
        task: str,
        solver_ckpt_dir: str,
        solver_config_file: str,
        solver_save_dir: str,
        solver_lora_rank: int,
        reviser_ckpt_dir: str,
        reviser_config_file: str,
        reviser_lora_rank: int,
        train_file: str,
        label_file: str,
        log_dir: str = None,
        max_batch_size: int = 4,
        eval_batch_size: int = 96,
        max_seq_len: int = 512,
        epochs: int = 1,
        inner_epochs: int = 1,
        lr: float = 1e-5,
        tokenizer_path: str = None,
):
    tokenizer_path = tokenizer_path if tokenizer_path else 'config/tokenizer.model'
    local_rank, world_size = setup_model_parallel()
    if log_dir is not None and local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
    dataset = EnhanceDataset(train_file)
    train_dataloader = DataLoader(dataset, batch_size=max_batch_size)
    eval_dataloader = DataLoader(dataset, batch_size=eval_batch_size)

    tokenizer = LlamaTokenizer(tokenizer_path)
    solver_args = LoraLlamaArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=solver_lora_rank
    ).from_json(solver_config_file)
    reviser_args = LoraLlamaArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=reviser_lora_rank
    ).from_json(reviser_config_file) if reviser_lora_rank > 0 else LlamaArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size).from_json(reviser_config_file)

    for epoch in range(epochs):

        # Solver Model Collecting
        solver_model = LoraLlama(solver_args)
        solver_model.load(solver_ckpt_dir if epoch == 0 else os.path.join(solver_save_dir, f"epoch-{epoch}"))
        solver_buffer_collector = SolverBufferCollector(solver_model, tokenizer, max_seq_len)
        solver_rollout_buffer = SolverRolloutBuffer()
        print('Solver buffer collecting ...')
        timer = Timer(len(eval_dataloader))
        for data in tqdm(eval_dataloader):
            timer.step()
            solver_rollout_buffer.extend(
                solver_buffer_collector.forward(data['instruction'], t=1.0)
            )

        solver_model.cpu()
        del solver_model
        del solver_buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        # Reward Model Scoring
        reward_buffer_collector = LabelBufferCollector(task, dataset, tokenizer, max_seq_len)
        reward_rollout_buffer = CriticRolloutBuffer()
        for data in solver_rollout_buffer.get(1):
            reward_rollout_buffer.extend(
                reward_buffer_collector.forward(
                    data.instructions, data.actions, data.action_masks
                )
            )

        # Filtering for Error
        error_rollout_buffer = SolverRolloutBuffer()
        for solver_data, reward_data in zip(solver_rollout_buffer.get(1), reward_rollout_buffer.get(1)):
            if reward_data.scores.mean() == 0:  # incorrect
                error_rollout_buffer.extend(
                    SolverRolloutBuffer(
                        instructions=solver_data.instructions,
                        actions=solver_data.actions,
                        action_masks=solver_data.action_masks
                    )
                )
        print("Error Rate on Training set: ", len(error_rollout_buffer) / len(solver_rollout_buffer))

        # Reviser Model Collecting
        reviser_model = LoraLlama(reviser_args) if reviser_lora_rank > 0 else Llama(reviser_args)
        reviser_model.load(reviser_ckpt_dir)
        reviser_buffer_collector = SolverBufferCollector(reviser_model, tokenizer, max_seq_len)
        reviser_rollout_buffer = SolverRolloutBuffer()
        print("Revising ...")
        timer = Timer(len(error_rollout_buffer) // eval_batch_size)
        for data in error_rollout_buffer.get(eval_batch_size):
            timer.step()
            instructions = []
            for instruction, action, action_mask in zip(
                    data.instructions, data.actions, data.action_masks
            ):
                output = tokenizer.decode(action[action_mask].tolist())
                instructions.append(instruction + output + RETHINKING)
            reviser_rollout_buffer.extend(
                reviser_buffer_collector.forward(instructions, t=0.2)
            )

        reviser_model.cpu()
        del reviser_model
        del reviser_buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        # Reward Model Re-scoring
        reward_buffer_collector = LabelBufferCollector(task, dataset, tokenizer, max_seq_len)
        reward_rollout_buffer = CriticRolloutBuffer()
        for error_data, revise_data in zip(error_rollout_buffer.get(1), reviser_rollout_buffer.get(1)):
            reward_rollout_buffer.extend(
                reward_buffer_collector.forward(
                    error_data.instructions, revise_data.actions, revise_data.action_masks
                )
            )

        # Filtering for Correct
        correct_rollout_buffer = SolverRolloutBuffer()
        for error_data, revise_data, reward_data in zip(
                error_rollout_buffer.get(1), reviser_rollout_buffer.get(1), reward_rollout_buffer.get(1)
        ):
            if reward_data.scores.mean() > 0:  # correct
                correct_rollout_buffer.extend(
                    SolverRolloutBuffer(
                        instructions=error_data.instructions,
                        actions=revise_data.actions,
                        action_masks=revise_data.action_masks
                    )
                )

        # Add to Dataset
        datalist = []
        for data in correct_rollout_buffer.get(1):
            for instruction, action, action_mask in zip(data.instructions, data.actions, data.action_masks):
                datalist.append(dict(instruction=instruction, output=tokenizer.decode(action[action_mask].tolist())))
        print(f'Successfully revised {len(datalist)} instances! Increase {dataset.enhance(datalist)} instances.')

        # Training
        solver_model = LoraLlama(solver_args)
        optimizer = torch.optim.Adam(solver_model.parameters(), lr=lr)
        trainer = ParallelSolverTrainer(solver_model, tokenizer, optimizer, max_seq_len)
        evaluator = SolverEvaluator(solver_model, tokenizer, eval_batch_size, max_seq_len)
        trainer.load_model(solver_ckpt_dir) if (
                epoch == 0
        ) else trainer.load(os.path.join(solver_save_dir, f"epoch-{epoch}"))
        print('Solver Training ...')
        for inner_epoch in range(inner_epochs):
            for data in tqdm(train_dataloader):
                outputs = trainer.forward(
                    instructions=data['instruction'],
                    outputs=data['output']
                )
                if trainer.step % 100 == 0:
                    print(f'step {trainer.step} of {len(train_dataloader)} -------------------------------')
                    print(f'LOSS: ', outputs.loss.item())
                    predict = trainer.predict(outputs.logits, data['instruction'], data['output'])[0]
                    print(predict['instruction'] + predict['output'])
        outputs = evaluator.forward(task, JsonDataset(label_file))
        print("Evaluate Accuracy: ", outputs.acc, "Missing: ", outputs.missing)
        if log_dir is not None:
            json_dump(outputs.datalist, os.path.join(
                log_dir, f'results-epoch-{epoch + 1}-{round(outputs.acc, 4)}.json'
            ), indent=4)
            json_dump(dataset.datalist, os.path.join(
                log_dir, 'teacher.json'
            ), indent=4)
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
