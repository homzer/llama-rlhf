import os
import random

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import JsonDataset
from src.entities import Timer
from src.evaluator import SolverEvaluator
from src.models.llama import LoraLlama
from src.models.modeling_args import LoraLlamaArgs
from src.ppo.buffer import CriticRolloutBuffer, ActorRolloutBuffer
from src.ppo.collector import ActorBufferCollector, LabelBufferCollector
from src.tokenizers import LlamaTokenizer
from src.trainer import ParallelSolverTrainer
from src.utils import setup_model_parallel, json_dump, json_load, deduplicate_texts

RETHINKING = '\n\n<|rethinking|>\n\n'


class EnhanceDataset(JsonDataset):
    def __init__(self, train_file, revise_file):
        super().__init__(train_file)
        self.revise_datalist = json_load(revise_file)
        self.enhance_datalist = [dict(instruction=data['instruction'], output=[]) for data in self.datalist]
        self.map = {}
        for i, data in enumerate(self.datalist):
            self.map[data['instruction']] = i
        assert len(list(self.map.keys())) == len(self.datalist)

    def enhance(self, enhance_datalist):
        for data in enhance_datalist:
            assert data['instruction'] in self.map.keys()
            i = self.map[data['instruction']]
            self.enhance_datalist[i]['output'].append(data['output'])
            self.enhance_datalist[i]['output'] = deduplicate_texts(self.enhance_datalist[i]['output'])

    def __getitem__(self, i):
        data = self.datalist[i].copy()
        revise_data = self.revise_datalist[i].copy()
        enhance_data = self.enhance_datalist[i].copy()
        # 50% revising
        if random.randint(0, 1) == 1 and len(revise_data['student_output']) != 0:
            j = random.randint(0, len(revise_data['student_output']) - 1)
            data['instruction'] += revise_data['student_output'][j] + RETHINKING
            data['output'] = revise_data['teacher_output'][j]
        else:  # 25% enhancing
            if random.randint(0, 1) == 1 and len(enhance_data['output']) != 0:
                data['output'] = random.sample(enhance_data['output'], 1)[0]
            else:  # 25% original
                data['output'] = random.sample(data['output'], 1)[0]
        return data


def run(
        task: str,
        solver_ckpt_dir: str,
        solver_config_file: str,
        solver_save_dir: str,
        reviser_ckpt_dir: str,
        reviser_config_file: str,
        train_file: str,
        revise_file: str,
        label_file: str,
        log_dir: str = None,
        lora_rank: int = 16,
        max_batch_size: int = 4,
        eval_batch_size: int = 96,
        max_seq_len: int = 512,
        epochs: int = 1,
        lr: float = 1e-5,
        tokenizer_path: str = None,
):
    tokenizer_path = tokenizer_path if tokenizer_path else 'config/tokenizer.model'
    dataset = EnhanceDataset(train_file, revise_file)
    train_dataloader = DataLoader(dataset, batch_size=max_batch_size)
    eval_dataloader = DataLoader(dataset, batch_size=eval_batch_size)

    local_rank, world_size = setup_model_parallel()
    tokenizer = LlamaTokenizer(tokenizer_path)
    solver_args = LoraLlamaArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=lora_rank
    ).from_json(solver_config_file)
    solver_model = LoraLlama(solver_args)
    solver_model.init_weights()
    optimizer = torch.optim.Adam(solver_model.parameters(), lr=lr)
    trainer = ParallelSolverTrainer(solver_model, tokenizer, optimizer, max_seq_len)
    evaluator = SolverEvaluator(solver_model, tokenizer, eval_batch_size, max_seq_len)
    trainer.load(solver_ckpt_dir)

    for epoch in range(epochs):

        # Solver Model Collecting
        solver_buffer_collector = ActorBufferCollector(solver_model, tokenizer, max_seq_len)
        solver_rollout_buffer = ActorRolloutBuffer()
        print('Solver buffer collecting ...')
        timer = Timer(len(eval_dataloader))
        for data in tqdm(eval_dataloader):
            timer.step()
            solver_rollout_buffer.extend(
                solver_buffer_collector.forward(data['instruction'])
            )

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
        error_rollout_buffer = ActorRolloutBuffer()
        for solver_data, reward_data in zip(solver_rollout_buffer.get(1), reward_rollout_buffer.get(1)):
            if reward_data.scores.mean() == 0:  # incorrect
                error_rollout_buffer.extend(
                    ActorRolloutBuffer(
                        instructions=solver_data.instructions,
                        obs=solver_data.obs,
                        actions=solver_data.actions,
                        action_logits=solver_data.action_logits,
                        action_masks=solver_data.action_masks
                    )
                )

        # Solver Model Revising
        revise_buffer_collector = ActorBufferCollector(solver_model, tokenizer, max_seq_len)
        revise_rollout_buffer = ActorRolloutBuffer()
        print("Solver model revising ...")
        timer = Timer(len(error_rollout_buffer) // eval_batch_size)
        for data in error_rollout_buffer.get(eval_batch_size):
            timer.step()
            instructions = []
            for instruction, action, action_mask in zip(
                    data.instructions, data.actions, data.action_masks
            ):
                output = tokenizer.decode(action[action_mask].tolist())
                instructions.append(instruction + output + RETHINKING)
            revise_rollout_buffer.extend(
                revise_buffer_collector.forward(instructions)
            )

        # Reward Model Re-scoring
        reward_buffer_collector = LabelBufferCollector(task, dataset, tokenizer, max_seq_len)
        reward_rollout_buffer = CriticRolloutBuffer()
        for error_data, revise_data in zip(error_rollout_buffer.get(1), revise_rollout_buffer.get(1)):
            reward_rollout_buffer.extend(
                reward_buffer_collector.forward(
                    error_data.instructions, revise_data.actions, revise_data.action_masks
                )
            )

        # Filtering for Correct
        correct_rollout_buffer = ActorRolloutBuffer()
        for error_data, revise_data, reward_data in zip(
                error_rollout_buffer.get(1), revise_rollout_buffer.get(1), reward_rollout_buffer.get(1)
        ):
            if reward_data.scores.mean() > 0:  # correct
                correct_rollout_buffer.extend(
                    ActorRolloutBuffer(
                        instructions=error_data.instructions,
                        obs=revise_data.obs,
                        actions=revise_data.actions,
                        action_logits=revise_data.action_logits,
                        action_masks=revise_data.action_masks
                    )
                )

        # Add to Dataset
        datalist = []
        for data in correct_rollout_buffer.get(1):
            for instruction, action, action_mask in zip(data.instructions, data.actions, data.action_masks):
                datalist.append(dict(instruction=instruction, output=tokenizer.decode(action[action_mask].tolist())))
        print(f'Successfully revised {len(datalist)} instances!')
        dataset.enhance(datalist)

        # Training
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
                log_dir, f'results-epoch-{epoch + 1}-{round(outputs.acc, 4)}.json'), indent=4
            )
        trainer.save(os.path.join(solver_save_dir, f"epoch-{epoch + 1}"))


if __name__ == '__main__':
    fire.Fire(run)
