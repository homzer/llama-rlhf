import os

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ReviseDataset
from src.models.llama import LoraLlama
from src.models.modeling_args import LoraLlamaArgs
from src.tokenizers import LlamaTokenizer
from src.trainer import ParallelSolverTrainer
from src.utils import setup_model_parallel


def preprocess(data: dict):
    new_data = dict(instruction=[], output=[])
    for i in range(len(data['instruction'])):
        new_data['instruction'].append(data['instruction'][i])
        if len(data['student_output'][i]) > 0:
            new_data['instruction'][i] += data['student_output'][i] + '\n\n<|rethinking|>\n\n'
        new_data['output'].append(data['teacher_output'][i])
    return new_data


def main(
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
        seed: int = None
):
    tokenizer_path = 'config/tokenizer.model' if tokenizer_path is None else tokenizer_path
    config_file = f"config/{model_type}/params.json" if config_file is None else config_file
    dataset = ReviseDataset(f=train_file)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)
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
    model.init_weights()
    model.load(ckpt_dir)
    # model.save_merge(os.path.join(save_dir, f"epoch-0"))
    # model.load_merge(os.path.join(save_dir, f"epoch-0"))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tokenizer = LlamaTokenizer(tokenizer_path)
    trainer = ParallelSolverTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        max_seq_len=max_seq_len
    )
    for epoch in range(epochs):
        for data in tqdm(dataloader):
            data = preprocess(data)
            outputs = trainer.forward(
                instructions=data['instruction'],
                outputs=data['output']
            )
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {len(dataloader)} -------------------------------')
                print(f'LOSS: ', outputs.loss.item())
                predict = trainer.predict(outputs.logits, data['instruction'], data['output'])[0]
                print(predict['instruction'] + predict['output'])
        trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))


if __name__ == '__main__':
    fire.Fire(main)
