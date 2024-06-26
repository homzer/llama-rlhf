import os

import fire

from src.dataset import JsonDataset, JsonDatasetForChatTemplate
from src.evaluator import SolverEvaluator
from src.modeling import get_parallel_model
from src.utils import setup_model_parallel, json_dump


def main(
        task: str,
        ckpt_dir: str,
        log_dir: str,
        label_file: str,
        model_type: str = "llama-2-7b",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lora_rank: int = -1,
        t: float = 0.0,
        p: float = 1.0,
        tokenizer_file: str = None,
        config_file: str = None,
        use_chat_template: bool = False,
        seed: int = None
):
    local_rank, world_size = setup_model_parallel(
        seed=seed
    )
    if tokenizer_file is None:
        tokenizer_file = ckpt_dir
    if config_file is None:
        config_file = ckpt_dir

    model, tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        local_rank=local_rank,
        world_size=world_size,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=lora_rank
    )
    dataset = JsonDataset(label_file)
    if use_chat_template:
        dataset = JsonDatasetForChatTemplate(dataset, tokenizer)
    model.load(ckpt_dir)
    evaluator = SolverEvaluator(model, tokenizer, max_batch_size, max_seq_len)
    outputs = evaluator.forward(task, dataset, t=t, p=p)
    print("Evaluate Accuracy: ", outputs.acc, "Missing: ", outputs.missing)
    if local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        json_dump(outputs.datalist, os.path.join(
            log_dir, f'results-{round(outputs.acc, 4)}.json'
        ), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
