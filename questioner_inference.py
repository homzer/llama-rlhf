import json
import os
import random

import fire

from src.dataset import JsonDataset
from src.generator import GeneratorForCausalLM
from src.modeling.llama import Llama
from src.modeling.llama_lora import LoraLlama
from src.modeling.modeling_args import LoraLlamaArgs, LlamaArgs
from src.tokenizer import LlamaTokenizer
from src.utils import setup_model_parallel, set_barrier, Timer


def create_questioner_batch_data(batch_size: int, dataset: JsonDataset, num_shots: int = 3):
    samples = [item['instruction'] for item in random.sample(dataset.datalist, num_shots * batch_size)]
    instructions = []
    for i in range(batch_size):
        instructions.append(
            f"[QUESTION] {'[QUESTION] '.join(samples[i * num_shots: (i + 1) * num_shots])}[QUESTION] "
        )
    return instructions


def main(
        ckpt_dir: str,
        config_file: str,
        train_file: str,
        num_shots: int = 3,
        num_generated_samples: int = 8000,
        eval_batch_size: int = 128,
        tokenizer_path: str = "config/tokenizer.model",
        max_seq_len: int = 512,
        lora_rank: int = -1,
        t: float = 0.0,
        p: float = 1.0,
        output_dir: str = "log/questioner",
        seed: int = None,
):
    local_rank, world_size = setup_model_parallel(
        use_float16=True, seed=seed)
    if lora_rank < 0:
        params = LlamaArgs(
            max_seq_len=max_seq_len,
            local_rank=local_rank,
            world_size=world_size
        ).from_json(config_file)
        model = Llama(params)
    else:
        params = LoraLlamaArgs(
            max_seq_len=max_seq_len,
            local_rank=local_rank,
            world_size=world_size,
            r=lora_rank).from_json(config_file)
        model = LoraLlama(params)
    model.load(ckpt_dir)
    set_barrier()
    dataset = JsonDataset(train_file)
    generator = GeneratorForCausalLM(model, LlamaTokenizer(tokenizer_path), max_seq_len)
    os.makedirs(output_dir, exist_ok=True)
    timer = Timer(num_generated_samples // eval_batch_size)
    with open(os.path.join(output_dir, 'questions.jsonl'), 'a', encoding='utf-8') as writer:
        for i in range(num_generated_samples // eval_batch_size):
            timer.step()
            data = create_questioner_batch_data(eval_batch_size, dataset, num_shots)
            outputs = generator.forward(data, t=t, p=p)
            for output in outputs:
                writer.write(json.dumps(
                    {'instruction': output['output']}
                ) + '\n')


if __name__ == '__main__':
    fire.Fire(main)
