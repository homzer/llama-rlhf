import os

import fire

from src.dataset import PairwiseDataset, ChatTemplateDataset
from src.evaluator import VerifierEvaluator
from src.modeling import get_parallel_verifier
from src.utils import setup_model_parallel, json_dump


def main(
        ckpt_dir: str,
        log_dir: str,
        label_file: str,
        model_type: str = "qwen-2-7b",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        tokenizer_file: str = None,
        config_file: str = None,
        dtype: str = "bfloat16",
        use_chat_template: bool = False,
        seed: int = None
):
    os.makedirs(log_dir, exist_ok=True)
    tokenizer_file = ckpt_dir if tokenizer_file is None else tokenizer_file
    config_file = ckpt_dir if config_file is None else config_file
    local_rank, world_size = setup_model_parallel(seed=seed)
    model, tokenizer = get_parallel_verifier(
        model_type=model_type,
        config_file=config_file,
        local_rank=local_rank,
        world_size=world_size,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=-1,
        dtype=dtype
    )
    model.load(ckpt_dir)

    dataset = PairwiseDataset(f=label_file)
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, tokenizer)

    evaluator = VerifierEvaluator(model, tokenizer, max_batch_size, max_seq_len)
    evaluator_outputs = evaluator.forward(dataset)
    print("Accuracy: ", evaluator_outputs.acc)
    json_dump(evaluator_outputs.datalist, os.path.join(log_dir, "results.json"), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
