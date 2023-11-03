import os

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MultiOutputsDataset
from src.evaluator import SolverEvaluator
from src.modeling.llama_lora import LoraLlama
from src.modeling.modeling_args import LoraLlamaArgs
from src.tokenizer import LlamaTokenizer
from src.trainer import ParallelSolverTrainer
from src.utils import setup_model_parallel, json_dump


def main(
        task: str,
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        model_type: str = "7B",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lr: float = 1e-5,
        epochs: int = 1,
        lora_rank: int = 16,
        tokenizer_path: str = None,
        config_file: str = None,
        label_file: str = None,
        eval_batch_size: int = None,
        log_dir: str = None,
        seed: int = None
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    tokenizer_path = 'config/tokenizer.model' if tokenizer_path is None else tokenizer_path
    config_file = f"config/{model_type}/params.json" if config_file is None else config_file
    local_rank, world_size = setup_model_parallel(
        use_float16=True, seed=seed
    )
    params = LoraLlamaArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=lora_rank
    ).from_json(config_file)

    model = LoraLlama(params)
    dataset = MultiOutputsDataset(filename=train_file)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tokenizer = LlamaTokenizer(tokenizer_path)
    trainer = ParallelSolverTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        max_seq_len=max_seq_len
    )
    evaluator = SolverEvaluator(model, tokenizer, eval_batch_size, max_seq_len)
    trainer.load(ckpt_dir)
    for epoch in range(epochs):
        for data in tqdm(dataloader):
            outputs = trainer.forward(
                instructions=data['instruction'],
                outputs=data['output']
            )
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {len(dataloader)} -------------------------------')
                print(f'LOSS: ', outputs.loss.item())
                predict = trainer.predict(outputs.logits, data['instruction'], data['output'])[0]
                print(predict['instruction'] + predict['output'])
        outputs = evaluator.forward(task, label_file)
        print("Evaluate Accuracy: ", outputs.acc, "Missing: ", outputs.missing)
        json_dump(outputs.datalist, os.path.join(
            log_dir, f'results-epoch-{epoch + 1}-{round(outputs.acc, 4)}.json'), indent=4
        )
        trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))


if __name__ == '__main__':
    fire.Fire(main)
