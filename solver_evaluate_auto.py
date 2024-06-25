import os
from pathlib import Path

import fire

from src.dataset import JsonDataset
from src.evaluator import SolverEvaluator
from src.modeling import get_parallel_model
from src.utils import setup_model_parallel, json_dump

TASKS = ['GSM8K', 'BBH', 'ARC', 'AGIEval', 'CSQA', 'MMLU']
LABEL_FILES = {
    'GSM8K': 'data/GSM8K/test.json',
    'BBH': 'data/BBH/test.json',
    'ARC': 'data/ARC/test.json',
    'AGIEval': 'data/AGIEval/test.json',
    'CSQA': 'data/CSQA/test.json',
    'MMLU': 'data/MMLU/test-preview.json',
}


# Tasks: GSM8K BBH ARC AGIEval MMLU CSQA
def main(
        ckpt_dir: str,
        log_dir: str,
        model_type: str = "llama-2-14b-chat",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        t: float = 0.0,
        p: float = 1.0,
        tokenizer_file: str = None,
        config_file: str = None,
        seed: int = None
):
    local_rank, world_size = setup_model_parallel(
        use_float16=True, seed=seed
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
        lora_rank=-1
    )
    checkpoints = sorted(Path(ckpt_dir).glob("epoch-*"), key=lambda x: int(x.name.replace('epoch-', "")))
    done_checkpoints = [checkpoint.name for checkpoint in Path(log_dir).glob("epoch-*")]
    if len(checkpoints) == 0:
        checkpoints = [Path(ckpt_dir)]
    for checkpoint in checkpoints:
        if checkpoint.name in done_checkpoints:
            continue
        model.load(checkpoint, merge_lora=True)
        evaluator = SolverEvaluator(
            model=model,
            tokenizer=tokenizer,
            batch_size=max_batch_size,
            max_seq_len=max_seq_len
        )
        for task in TASKS:
            label_file = LABEL_FILES[task]
            outputs = evaluator.forward(task, JsonDataset(label_file), t=t, p=p)
            print("Task: ", task, "Evaluate Accuracy: ", outputs.acc, "Missing: ", outputs.missing)
            if local_rank == 0:
                os.makedirs(os.path.join(log_dir, checkpoint.name), exist_ok=True)
                json_dump(outputs.datalist, os.path.join(
                    log_dir, checkpoint.name, f'{task}-results-{round(outputs.acc, 5)}.json'
                ), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
