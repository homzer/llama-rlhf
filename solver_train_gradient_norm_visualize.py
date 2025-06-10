import os
import random

import fire
import torch

from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.models.modeling import ParallelModelForCausalLM
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.initialize import setup_model_parallel
from src.parallel.optimizer import ParallelOptimizer
from src.tokenizers import Tokenizer
from src.trainer import prepare_for_forward
from policy_train_policy_gradient_norm_visualize import compute_result_dict
from src.utils import json_dump, json_load


def sft_forward(
        model: ParallelModelForCausalLM,
        data: dict,
        tokenizer: Tokenizer,
        max_seq_len: int
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    model.train()
    example = prepare_for_forward(data["instruction"], data["output"], tokenizer, max_seq_len)
    logits = model.forward(example.tokens).logits
    labels = example.labels.view(-1).to(logits.device)
    masks = example.masks.view(-1).to(logits.device)
    loss = criterion.forward(
        input=logits.view(-1, logits.size(-1)),
        target=labels
    )
    labels = torch.masked_select(labels, masks)
    logits = logits.view(-1, logits.shape[-1])[masks]
    action_logprobs = torch.gather(
        torch.log_softmax(logits, dim=-1), dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    return loss, action_logprobs


def run(
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        model_type: str,
        tokenizer_file: str = None,
        config_file: str = None,
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lr: float = 1e-5,
        epochs: int = 1,
        inner_epochs: int = 1000,
        dtype: str = "bfloat16",
        use_chat_template: bool = False,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
        seed: int = 0
):
    tokenizer_file = tokenizer_file or ckpt_dir
    config_file = config_file or ckpt_dir
    parallel_infos = setup_model_parallel(
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size,
        seed=seed
    )
    model, tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        tokenizer_file=tokenizer_file,
        max_seq_len=max_seq_len,
        dtype=dtype,
    )
    datalist = json_load(train_file)
    random.shuffle(datalist)
    dataset = JsonDataset(f=datalist)
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, tokenizer)
    dataloader = ParallelDataLoader(dataset, batch_size=max_batch_size)
    optimizer = ParallelOptimizer(torch.optim.Adam(model.parameters(), lr=lr))
    model.load(ckpt_dir)
    results = []
    timer = Timer(total=len(dataloader), episode=100)
    for epoch in range(epochs):
        for data in dataloader:
            for inner_epoch in range(inner_epochs):
                timer.step()
                loss, action_logprobs = sft_forward(model, data, tokenizer, max_seq_len)
                optimizer.zero_grad()
                loss.backward()
                results.append(compute_result_dict(model, action_logprobs))
                optimizer.step()
                print(f'Loss: {loss}')
            break
        if parallel_infos.global_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            json_dump(results, os.path.join(save_dir, "grad.jsonl"))
        exit(0)


if __name__ == '__main__':
    fire.Fire(run)
