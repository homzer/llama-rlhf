import os

import fire
import torch
from torch.utils.data import DataLoader

from src.dataset import MultiOutputsDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.trainer import ParallelSolverTrainer
from src.utils import json_load
from src.parallel import setup_model_parallel


def main(
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        model_type: str = "llama-1-7b",
        tokenizer_file: str = None,
        config_file: str = None,
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lr: float = 1e-5,
        dtype: str = "float16",
        lora_rank: int = -1,
        lora_dtype: str = "float32",
        chunk_size: int = 10000,
        seed: int = None,
):
    if tokenizer_file is None:
        tokenizer_file = ckpt_dir
    if config_file is None:
        config_file = ckpt_dir
    setup_model_parallel(seed=seed)

    model, tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        tokenizer_file=tokenizer_file,
        max_seq_len=max_seq_len,
        lora_rank=lora_rank,
        dtype=dtype,
        lora_dtype=lora_dtype
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = ParallelSolverTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        max_seq_len=max_seq_len
    )
    trainer.load(ckpt_dir)
    datalist = json_load(train_file)
    epochs = len(datalist) // chunk_size

    for epoch in range(epochs):
        print(f"Epoch - {epoch} of {epochs}")
        dataset = MultiOutputsDataset(datalist[epoch * chunk_size: (epoch + 1) * chunk_size])
        if len(dataset) == 0:
            return
        dataloader = DataLoader(dataset, batch_size=max_batch_size)
        timer = Timer(total=len(dataloader), episode=100)
        for data in dataloader:
            outputs = trainer.forward(
                instructions=data['instruction'],
                outputs=data['output']
            )
            timer.step()
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {len(dataloader)} -------------------------------')
                print(f'LOSS: ', outputs.loss.item())
                trainer.predict(outputs.logits, data['instruction'], data['output'])
        trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))


if __name__ == '__main__':
    fire.Fire(main)
