import os

import fire
import torch
from torch.utils.data import DataLoader

from src.dataset import PairwiseDataset, ChatTemplateDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.parallel.initialize import setup_model_parallel
from src.trainer import ParallelSolverSimPOTrainer
from src.utils import json_load


def run(
        train_file: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_tokenizer_file: str,
        policy_config_file: str,
        max_seq_len: int = 1024,
        max_batch_size: int = 1,
        gamma: float = 1.0,
        beta: float = 2.0,
        lr: float = 1e-6,
        dtype: str = "bfloat16",
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        chunk_size: int = None,
        epochs: int = 1,
        ce_coef: float = 0.0,
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
):
    setup_model_parallel(seed=seed)
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir

    policy, policy_tokenizer = get_parallel_model(
        model_type=policy_model_type,
        config_file=policy_config_file,
        tokenizer_file=policy_tokenizer_file,
        max_seq_len=max_seq_len,
        dtype=dtype,
        lora_rank=lora_rank,
        lora_dtype=lora_dtype
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    trainer = ParallelSolverSimPOTrainer(
        model=policy,
        tokenizer=policy_tokenizer,
        optimizer=optimizer,
        max_seq_len=max_seq_len,
        beta=beta,
        gamma=gamma,
        ce_coef=ce_coef
    )
    policy.load(policy_ckpt_dir, merge_lora=True) if (
            begin_epoch == 0
    ) else trainer.load(os.path.join(save_dir, f"epoch-{begin_epoch}"))

    datalist = json_load(train_file)
    chunk_size = chunk_size or len(datalist)
    local_epochs = len(datalist) // chunk_size
    begin_global_epoch = begin_epoch // local_epochs
    begin_local_epoch = begin_epoch % local_epochs
    for global_epoch in range(begin_global_epoch, epochs):
        for local_epoch in range(begin_local_epoch, local_epochs):
            epoch = local_epoch + global_epoch * local_epochs
            print(f"Epoch - {epoch} of {local_epochs * epochs}")
            dataset = PairwiseDataset(f=datalist[local_epoch * chunk_size: (local_epoch + 1) * chunk_size])
            if len(dataset) == 0:
                continue
            if use_chat_template:
                dataset = ChatTemplateDataset(dataset, policy_tokenizer)
            dataloader = DataLoader(dataset, batch_size=max_batch_size)
            timer = Timer(len(dataloader), episode=100)
            for data in dataloader:
                timer.step()
                trainer_outputs = trainer.forward(
                    instructions=data["instruction"],
                    chosen=data["chosen"],
                    rejected=data["rejected"]
                )
                if trainer.step % 100 == 0:
                    print(f'step {trainer.step} of {len(dataloader)} ---------------')
                    print('SimPO LOSS: ', trainer_outputs.loss_simpo, f'CE LOSS: ', trainer_outputs.loss_ce)
                    trainer.predict(trainer_outputs.logits, data["instruction"], data["chosen"])
            trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))


if __name__ == '__main__':
    fire.Fire(run)
