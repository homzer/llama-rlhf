import os

import fire

from src.dataset import JsonDataset
from src.evaluator import SolverEvaluator
from src.modeling.llama_lora import LoraLlama
from src.modeling.modeling_args import LoraLlamaArgs
from src.tokenizer import LlamaTokenizer
from src.utils import setup_model_parallel, json_dump, json_load


def main(
        task: str,
        ckpt_dir: str,
        log_dir: str,
        label_file: str,
        model_type: str = "llama-1-7b",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lora_rank: int = 16,
        t: float = 0.0,
        p: float = 1.0,
        tokenizer_path: str = None,
        config_file: str = None,
        seed: int = None
):
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
    model.load(ckpt_dir)
    evaluator = SolverEvaluator(model, LlamaTokenizer(tokenizer_path), max_batch_size, max_seq_len)

    log_datalist = json_load(label_file)
    # Collect Error Instances
    err_datalist = []
    correct_count = 0
    for data in log_datalist:
        if data['label'] not in data['predict'][-1:]:
            data = data.copy()
            data.pop('predict')
            data['instruction'] += data['output'] + '\n\n<|rethinking|>\n\n'
            data['output'] = ''
            err_datalist.append(data)
        else:
            correct_count += 1

    print('Revising ...')
    revise_outputs = evaluator.forward(task, JsonDataset(err_datalist), t=t, p=p)
    revise_acc = revise_outputs.correct / (len(log_datalist) - correct_count)
    print(f"Revised {revise_outputs.correct} of {len(log_datalist) - correct_count}. Revising Acc {revise_acc}")

    os.makedirs(log_dir, exist_ok=True)
    json_dump(revise_outputs.datalist, os.path.join(log_dir, f'results-{round(revise_acc, 4)}.json'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
