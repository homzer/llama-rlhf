import os
import random

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import JsonDataset
from src.models.llama import LoraLlama
from src.models.modeling_args import LoraLlamaArgs
from src.tokenizers import LlamaTokenizer
from src.trainer import ParallelSolverTrainer
from src.utils import setup_model_parallel


def create_questioner_batch_data(data, dataset: JsonDataset, num_shots: int = 3):
    bzs = len(data['instruction'])
    samples = [item['instruction'] for item in random.sample(dataset.datalist, num_shots * bzs)]
    data['output'] = data['instruction'].copy()
    instructions = []
    for i in range(bzs):
        instructions.append(
            f"[QUESTION] {'[QUESTION] '.join(samples[i * num_shots: (i + 1) * num_shots])}[QUESTION] "
        )
    data['instruction'] = instructions
    return data


def main(
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        model_type: str = "7B",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        eval_batch_size: int = 64,
        lr: float = 1e-5,
        epochs: int = 1,
        lora_rank: int = 16,
        tokenizer_path: str = None,
        config_file: str = None,
        seed: int = None
):
    tokenizer_path = 'config/tokenizer.model' if tokenizer_path is None else tokenizer_path
    config_file = f"config/{model_type}/params.json" if config_file is None else config_file
    seed = 1 if seed is None else seed
    local_rank, world_size = setup_model_parallel(
        seed=seed
    )
    params = LoraLlamaArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=lora_rank
    ).from_json(config_file)
    model = LoraLlama(params)
    model.init_weights()

    dataset = JsonDataset(f=train_file)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = ParallelSolverTrainer(
        model=model,
        tokenizer=LlamaTokenizer(tokenizer_path),
        optimizer=optimizer,
        max_seq_len=max_seq_len
    )
    trainer.load(ckpt_dir)

    for epoch in range(epochs):
        for data in tqdm(dataloader):
            data = create_questioner_batch_data(data, dataset)
            outputs = trainer.forward(
                instructions=data['instruction'],
                outputs=data['output']
            )
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} ----------------------------------')
                print("LOSS: ", outputs.loss.item())
                trainer.predict(outputs.logits, data['instruction'], data['output'])
            if trainer.step > 1000:
                break
        trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))


if __name__ == '__main__':
    fire.Fire(main)
