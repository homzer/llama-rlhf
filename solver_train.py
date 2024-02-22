import os

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MultiOutputsDataset, JsonDataset
from src.entities import Timer
from src.evaluator import SolverEvaluator
from src.models.modeling_utils import get_parallel_model
from src.trainer import ParallelSolverTrainer
from src.utils import setup_model_parallel, json_dump


def main(
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        model_type: str = "llama-1-7b",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lr: float = 1e-5,
        epochs: int = 1,
        lora_rank: int = 16,
        task: str = None,
        tokenizer_file: str = None,
        config_file: str = None,
        label_file: str = None,
        eval_batch_size: int = None,
        log_dir: str = None,
        seed: int = None,
        use_float16: bool = True
):
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
    config_file = f"config/{model_type}/params.json" if config_file is None else config_file
    local_rank, world_size = setup_model_parallel(
        use_float16=use_float16, seed=seed
    )

    model, tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        local_rank=local_rank,
        world_size=world_size,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=lora_rank
    )
    dataset = MultiOutputsDataset(f=train_file)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = ParallelSolverTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        max_seq_len=max_seq_len
    )
    evaluator = SolverEvaluator(
        model, tokenizer, eval_batch_size, max_seq_len
    ) if task is not None else None
    trainer.load(ckpt_dir)
    for epoch in range(epochs):
        timer = Timer(total=len(dataloader), episode=100)
        for data in tqdm(dataloader):
            outputs = trainer.forward(
                instructions=data['instruction'],
                outputs=data['output']
            )
            timer.step()
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {len(dataloader)} -------------------------------')
                print(f'LOSS: ', outputs.loss.item())
                predict = trainer.predict(outputs.logits, data['instruction'], data['output'])[0]
                print(predict['instruction'] + predict['output'])
            if trainer.step % 7200 == 0:
                trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))
        trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))
        if evaluator is not None:
            outputs = evaluator.forward(task, JsonDataset(label_file))
            print("Evaluate Accuracy: ", outputs.acc, "Missing: ", outputs.missing)
            if log_dir is not None:
                json_dump(outputs.datalist, os.path.join(
                    log_dir, f'results-epoch-{epoch + 1}-{round(outputs.acc, 4)}.json'), indent=4
                )


if __name__ == '__main__':
    fire.Fire(main)
