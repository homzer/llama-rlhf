import gc
import os

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import JsonDataset, EvoMultiOutputsDataset, ReviseDataset
from src.entities import Timer
from src.evaluator import SolverEvaluator
from src.modeling.llama import Llama
from src.modeling.llama_lora import LoraLlama
from src.modeling.modeling_args import LoraLlamaArgs, LlamaArgs
from src.ppo.buffer import CriticRolloutBuffer, SolverRolloutBuffer
from src.ppo.collector import SolverBufferCollector, LabelBufferCollector
from src.tokenizer import LlamaTokenizer
from src.trainer import ParallelSolverTrainer
from src.utils import setup_model_parallel, json_dump, set_barrier

RETHINKING = '\n\n<|rethinking|>\n\n'


def training_dataset(dataset: EvoMultiOutputsDataset) -> EvoMultiOutputsDataset:
    datalist = []
    for data in dataset.datalist:
        if len(data['output']) > 0:  #
            datalist.append(data)
    return EvoMultiOutputsDataset(datalist)


def revise(
        error_rollout_buffer: SolverRolloutBuffer,
        reviser_buffer_collector: SolverBufferCollector,
        reward_buffer_collector: LabelBufferCollector,
        tokenizer: LlamaTokenizer,
        batch_size: int,
        revising_turns: int
) -> (SolverRolloutBuffer, list):
    error_rollout_buffer = error_rollout_buffer.copy()
    correct_rollout_buffer = SolverRolloutBuffer()
    reviser_datalist = []
    for turn in range(revising_turns):
        print(f"Revising {turn+1}/{revising_turns} ...")
        reviser_rollout_buffer = SolverRolloutBuffer()
        timer = Timer(len(error_rollout_buffer) // batch_size)
        for data in error_rollout_buffer.get(batch_size):
            timer.step()
            instructions = []
            for instruction, action, action_mask in zip(
                    data.instructions, data.actions, data.action_masks
            ):
                output = tokenizer.decode(action[action_mask].tolist())
                instructions.append(instruction + output + RETHINKING)
            reviser_rollout_buffer.extend(
                reviser_buffer_collector.forward(instructions, t=1.0)
            )

        # Reward Model Re-scoring
        reward_rollout_buffer = CriticRolloutBuffer()
        for error_data, revise_data in zip(error_rollout_buffer.get(1), reviser_rollout_buffer.get(1)):
            reward_rollout_buffer.extend(
                reward_buffer_collector.forward(
                    error_data.instructions, revise_data.actions, revise_data.action_masks
                )
            )

        # Filtering for Correct
        error_rollout_buffer_ = SolverRolloutBuffer()
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
                for instruction, student_action, student_action_mask, teacher_action, teacher_action_mask in zip(
                        error_data.instructions, error_data.actions, error_data.action_masks,
                        revise_data.actions, revise_data.action_masks
                ):
                    reviser_datalist.append(dict(
                        instruction=str(instruction),
                        teacher_output=[tokenizer.decode(teacher_action[teacher_action_mask].tolist())],
                        student_output=[tokenizer.decode(student_action[student_action_mask].tolist())]
                    ))
            else:
                error_rollout_buffer_.extend(
                    SolverRolloutBuffer(
                        instructions=error_data.instructions,
                        actions=error_data.actions,
                        action_masks=error_data.action_masks
                    )
                )
        error_rollout_buffer = error_rollout_buffer_
        print(f"Successfully Revised {len(correct_rollout_buffer)} Instances!")

    return correct_rollout_buffer, reviser_datalist


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
        revising_turns: int = 3,
        max_batch_size: int = 6,
        eval_batch_size: int = 384,
        max_seq_len: int = 512,
        reviser_max_seq_len: int = 768,
        epochs: int = 1,
        inner_epochs: int = 1,
        lr: float = 2e-6,
        tokenizer_path: str = None,
):
    tokenizer_path = tokenizer_path if tokenizer_path else 'config/tokenizer.model'
    local_rank, world_size = setup_model_parallel()
    if log_dir is not None and local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)

    dataset = EvoMultiOutputsDataset(train_file)
    revise_dataset = ReviseDataset(train_file)
    tokenizer = LlamaTokenizer(tokenizer_path)
    solver_args = LoraLlamaArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=solver_lora_rank
    ).from_json(solver_config_file)
    reviser_args = LoraLlamaArgs(
        max_seq_len=reviser_max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=reviser_lora_rank
    ).from_json(reviser_config_file) if reviser_lora_rank > 0 else LlamaArgs(
        max_seq_len=reviser_max_seq_len,
        local_rank=local_rank,
        world_size=world_size).from_json(reviser_config_file)

    for epoch in range(epochs):

        # Solver Model Collecting
        solver_model = Llama(solver_args)
        solver_model.load(solver_ckpt_dir if (
                epoch == 0
        ) else os.path.join(solver_save_dir, f"epoch-{epoch}"), merge_lora=True)
        solver_buffer_collector = SolverBufferCollector(solver_model, tokenizer, max_seq_len)
        solver_rollout_buffer = SolverRolloutBuffer()
        print('Solver buffer collecting ...')
        eval_dataloader = DataLoader(dataset, batch_size=eval_batch_size)
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
        correct_rollout_buffer = SolverRolloutBuffer()
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
            else:  # correct
                correct_rollout_buffer.extend(
                    SolverRolloutBuffer(
                        instructions=solver_data.instructions,
                        actions=solver_data.actions,
                        action_masks=solver_data.action_masks
                    )
                )
        print("Error Rate on Training set: ", len(error_rollout_buffer) / len(solver_rollout_buffer))

        # Reviser Model Collecting
        reviser_model = Llama(reviser_args)
        reviser_model.load(reviser_ckpt_dir, merge_lora=reviser_lora_rank > 0)
        reviser_buffer_collector = SolverBufferCollector(reviser_model, tokenizer, reviser_max_seq_len)
        revise_correct_rollout_buffer, reviser_datalist = revise(
            error_rollout_buffer=error_rollout_buffer,
            reviser_buffer_collector=reviser_buffer_collector,
            reward_buffer_collector=reward_buffer_collector,
            tokenizer=tokenizer,
            batch_size=196,
            revising_turns=revising_turns
        )
        reviser_model.cpu()
        del reviser_model
        del reviser_buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        # Add to Dataset
        datalist = []
        for data in revise_correct_rollout_buffer.get(1):
            for instruction, action, action_mask in zip(data.instructions, data.actions, data.action_masks):
                datalist.append(dict(instruction=instruction, output=[tokenizer.decode(action[action_mask].tolist())]))
        print(f'Successfully revised {len(datalist)} of {len(error_rollout_buffer)} instances!')
        for data in correct_rollout_buffer.get(1):
            for instruction, action, action_mask in zip(data.instructions, data.actions, data.action_masks):
                datalist.append(dict(instruction=instruction, output=[tokenizer.decode(action[action_mask].tolist())]))
        print(f"Increasing {dataset.extend(EvoMultiOutputsDataset(datalist), deduplicate=True)} instances.")
        revise_dataset.extend(ReviseDataset(reviser_datalist))

        # Training
        solver_model = LoraLlama(solver_args)
        optimizer = torch.optim.Adam(solver_model.parameters(), lr=lr)
        trainer = ParallelSolverTrainer(solver_model, tokenizer, optimizer, max_seq_len)
        evaluator = SolverEvaluator(solver_model, tokenizer, eval_batch_size, max_seq_len)
        solver_model.load(solver_ckpt_dir, merge_lora=True) if (
                epoch == 0
        ) else trainer.load(os.path.join(solver_save_dir, f"epoch-{epoch}"))
        print('Solver Training ...')
        train_dataloader = DataLoader(training_dataset(dataset), batch_size=max_batch_size)
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
            json_dump(revise_dataset.datalist, os.path.join(
                log_dir, 'revise.json'
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
