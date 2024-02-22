import os

import fire

from prompt_solver_forward import PromptDataset
from src.evaluator import SolverEvaluator
from src.models.modeling_utils import get_parallel_model
from src.utils import setup_model_parallel, json_dump


def main(
        task: str,
        ckpt_dir: str,
        log_dir: str,
        label_file: str,
        model_type: str = "7B",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lora_rank: int = 16,
        t: float = 0.0,
        p: float = 1.0,
        tokenizer_file: str = None,
        config_file: str = None,
        seed: int = None
):
    local_rank, world_size = setup_model_parallel(
        use_float16=True, seed=seed
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
    model.load(ckpt_dir, merge_lora=not lora_rank > 0)
    evaluator = SolverEvaluator(model, tokenizer, max_batch_size, max_seq_len)
    outputs = evaluator.forward(task, PromptDataset(label_file), t=t, p=p)
    print("Evaluate Accuracy: ", outputs.acc, "Missing: ", outputs.missing)
    os.makedirs(log_dir, exist_ok=True)
    json_dump(outputs.datalist, os.path.join(
        log_dir, f'results-{round(outputs.acc, 4)}.json'
    ), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
