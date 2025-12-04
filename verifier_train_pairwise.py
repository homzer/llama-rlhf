import os

import fire
import torch

from src.dataset import PairwiseDataset, ChatTemplateDataset
from src.entities import Timer
from src.modeling import get_parallel_verifier
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.initialize import setup_model_parallel
from src.parallel.optimizer import ParallelOptimizer
from src.rewards.trainer import (
    ParallelVerifierTrainerForLastToken,
    ParallelVerifierTrainerForMeanScore,
    ParallelVerifierTrainerForFocalMeanScore,
    ParallelVerifierTrainerForFocalLoss, ParallelVerifierTrainerForPGTG,
)
from src.utils import print_current_func_args


def main(
        strategy: str,
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        model_type: str,
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lr: float = 1e-5,
        epochs: int = 1,
        lora_rank: int = -1,
        tokenizer_file: str = None,
        config_file: str = None,
        dtype: str = "bfloat16",
        lora_dtype: str = "bfloat16",
        use_chat_template: bool = False,
        seed: int = None,
        beta: float = 1.0,
        gamma: float = 0.0,
        log_dir: str = None,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
):
    tokenizer_file = tokenizer_file or ckpt_dir
    config_file = config_file or ckpt_dir
    setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()
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
    dataloader = ParallelDataLoader(dataset, batch_size=max_batch_size)
    optimizer = ParallelOptimizer(torch.optim.Adam(model.parameters(), lr=lr))
    if "last-token" in strategy:
        trainer = ParallelVerifierTrainerForLastToken(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer
        )
    elif "mean-score" in strategy:
        trainer = ParallelVerifierTrainerForMeanScore(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            beta=beta,
            gamma=gamma
        )
    elif "focal-loss" in strategy:
        trainer = ParallelVerifierTrainerForFocalLoss(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer
        )
    elif "focal-mean-score" in strategy:
        trainer = ParallelVerifierTrainerForFocalMeanScore(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer
        )
    elif "pgtg" in strategy:
        trainer = ParallelVerifierTrainerForPGTG(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer
        )
    else:
        raise ValueError(strategy)

    trainer.load(ckpt_dir)
    for epoch in range(epochs):
        timer = Timer(len(dataloader), episode=10)
        for data in dataloader:
            outputs = trainer.forward(
                instructions=data['instruction'],
                chosen=data['chosen'],
                rejected=data['rejected']
            )
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {len(dataloader)} -------------------------------')
                print(f"LOSS: {outputs.loss} Acc: {trainer.verifier_accuracy()}")
            timer.step()
            if trainer.step % 10000 == 0:
                trainer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))
        trainer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))


if __name__ == '__main__':
    fire.Fire(main)
