import os

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PairwiseDataset, ChatTemplateDataset
from src.entities import Timer
from src.modeling import get_parallel_verifier
from src.trainer import ParallelVerifierPairwiseTrainer
from src.parallel.utils import setup_model_parallel


def main(
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        model_type: str = "qwen-2-7b",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lr: float = 1e-5,
        epochs: int = 1,
        lora_rank: int = -1,
        tokenizer_file: str = None,
        config_file: str = None,
        dtype: str = "bfloat16",
        lora_dtype: str = "float32",
        use_chat_template: bool = False,
        seed: int = None
):
    tokenizer_file = ckpt_dir if tokenizer_file is None else tokenizer_file
    config_file = ckpt_dir if config_file is None else config_file
    setup_model_parallel(seed=seed)
    model, tokenizer = get_parallel_verifier(
        model_type=model_type,
        config_file=config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=lora_rank,
        dtype=dtype,
        lora_dtype=lora_dtype
    )

    dataset = PairwiseDataset(f=train_file)
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, tokenizer)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = ParallelVerifierPairwiseTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer
    )
    trainer.load(ckpt_dir)
    for epoch in range(epochs):
        timer = Timer(len(dataloader), episode=100)
        for data in tqdm(dataloader):
            outputs = trainer.forward(
                instructions=data['instruction'],
                chosen=data['chosen'],
                rejected=data['rejected']
            )
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {len(dataloader)} -------------------------------')
                print("LOSS: ", outputs.loss.item())
            timer.step()
        trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))


if __name__ == '__main__':
    fire.Fire(main)
