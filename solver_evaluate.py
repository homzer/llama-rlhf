import os

import fire

from src.dataset import JsonDataset, ChatTemplateDataset
from src.evaluator import SolverEvaluator
from src.modeling import get_parallel_model
from src.utils import json_dump
from src.parallel import setup_model_parallel


def main(
        task: str,
        ckpt_dir: str,
        log_dir: str,
        label_file: str,
        model_type: str = "llama-2-7b",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lora_rank: int = -1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tokenizer_file: str = None,
        config_file: str = None,
        use_chat_template: bool = False,
        dtype: str = "bfloat16",
        seed: int = None
):
    parallel_infos = setup_model_parallel(seed=seed)
    if tokenizer_file is None:
        tokenizer_file = ckpt_dir
    if config_file is None:
        config_file = ckpt_dir

    model, tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=lora_rank,
        dtype=dtype
    )
    dataset = JsonDataset(label_file)
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, tokenizer)
    model.load(ckpt_dir)
    evaluator = SolverEvaluator(model, tokenizer, max_batch_size, max_seq_len, temperature, top_p)
    outputs = evaluator.forward(task, dataset)
    print("Evaluate Accuracy: ", outputs.acc, "Missing: ", outputs.missing)
    if parallel_infos.local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        json_dump(outputs.datalist, os.path.join(
            log_dir, f'results-{round(outputs.acc, 4)}.json'
        ), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
