import os

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PairwiseDataset
from src.evaluator import VerifierEvaluator
from src.modeling import get_parallel_verifier
from src.trainer import ParallelVerifierTrainer
from src.utils import setup_model_parallel, json_dump


def main(
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        model_type: str = "qwen-2-7b",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lr: float = 1e-5,
        epochs: int = 1,
        lora_rank: int = 16,
        tokenizer_file: str = None,
        config_file: str = None,
        label_file: str = None,
        eval_batch_size: int = None,
        log_dir: str = None,
        dtype: str = "bfloat16",
        lora_dtype: str = "float32",
        seed: int = None
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    tokenizer_file = ckpt_dir if tokenizer_file is None else tokenizer_file
    config_file = ckpt_dir if config_file is None else config_file
    local_rank, world_size = setup_model_parallel(
        use_float16=True, seed=seed
    )
    model, tokenizer = get_parallel_verifier(
        model_type=model_type,
        config_file=config_file,
        local_rank=local_rank,
        world_size=world_size,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=lora_rank,
        dtype=dtype,
        lora_dtype=lora_dtype
    )

    dataset = PairwiseDataset(f=train_file)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = ParallelVerifierTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer
    )
    evaluator = VerifierEvaluator(model, tokenizer, eval_batch_size, max_seq_len)
    trainer.load(ckpt_dir)
    for epoch in range(epochs):
        for data in tqdm(dataloader):
            outputs = trainer.forward(
                instructions=data['instruction'],
                chosen=data['chosen'],
                rejected=data['rejected']
            )
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {len(dataloader)} -------------------------------')
                print("LOSS: ", outputs.loss.item())
        outputs = evaluator.forward(PairwiseDataset(label_file))
        print('Evaluate Accuracy: ', outputs.acc)
        json_dump(outputs.datalist, os.path.join(
            log_dir, f'results-epoch-{epoch + 1}-{round(outputs.acc, 4)}.jsonl')
        )
        trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))


if __name__ == '__main__':
    fire.Fire(main)
