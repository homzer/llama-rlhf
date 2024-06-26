import os

import fire

from src.dataset import PairwiseDataset
from src.evaluator import VerifierEvaluator
from src.models.llama import LoraLlamaVerifier
from src.models.modeling_args import LoraLlamaArgs
from src.tokenizers import LlamaTokenizer
from src.utils import setup_model_parallel, json_dump


def main(
        ckpt_dir: str,
        log_dir: str,
        label_file: str,
        model_type: str = "7B",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lora_rank: int = 16,
        tokenizer_path: str = None,
        config_file: str = None,
        seed: int = None
):
    tokenizer_path = 'config/tokenizer.model' if tokenizer_path is None else tokenizer_path
    config_file = f"config/{model_type}/params.json" if config_file is None else config_file
    local_rank, world_size = setup_model_parallel(
        seed=seed
    )
    params = LoraLlamaArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=lora_rank
    ).from_json(config_file)

    model = LoraLlamaVerifier(params)
    model.init_weights()
    model.load(ckpt_dir)
    evaluator = VerifierEvaluator(model, LlamaTokenizer(tokenizer_path), max_batch_size, max_seq_len)
    outputs = evaluator.forward(PairwiseDataset(label_file))
    print('Evaluate Accuracy: ', outputs.acc)
    os.makedirs(log_dir, exist_ok=True)
    json_dump(outputs.datalist, os.path.join(
        log_dir, f'results-{round(outputs.acc, 4)}.json'
    ), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
