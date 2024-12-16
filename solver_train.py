import os

import fire
import torch
from torch.utils.data import DataLoader

from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.parallel.utils import setup_model_parallel
from src.trainer import ParallelSolverTrainer


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
        epochs: int = 1,
        dtype: str = "bfloat16",
        lora_rank: int = -1,
        lora_dtype: str = "float32",
        save_steps: int = 10000,
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
):
    tokenizer_file = tokenizer_file or ckpt_dir
    config_file = config_file or ckpt_dir
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
    dataset = JsonDataset(f=train_file)
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, tokenizer)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = ParallelSolverTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        max_seq_len=max_seq_len
    )
    trainer.load(ckpt_dir if (begin_epoch == 0) else os.path.join(save_dir, f"epoch-{begin_epoch}"))
    for epoch in range(begin_epoch, epochs):
        timer = Timer(total=len(dataloader), episode=100)
        for data in dataloader:
            outputs = trainer.forward(
                instructions=data['instruction'],
                outputs=data['output']
            )
            timer.step()
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {len(dataloader)} -----------------------------')
                print(f'LOSS: ', outputs.loss.item())
                trainer.predict(outputs.logits, data['instruction'], data['output'])
            if trainer.step % save_steps == 0:
                trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))
        trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))


if __name__ == '__main__':
    fire.Fire(main)
