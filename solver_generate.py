import json
import os

import fire

from src.parallel.dataloader import ParallelDataLoader
from src.parallel.utils import setup_model_parallel
from src.parallel.datawriter import ParallelDataWriter
from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.generator import GeneratorForCausalLM
from src.modeling import get_parallel_model


def main(
        ckpt_dir: str,
        label_file: str,
        log_dir: str,
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
        model_parallel_size: int = None,
        seed: int = None
):
    os.makedirs(log_dir, exist_ok=True)
    setup_model_parallel(model_parallel_size=model_parallel_size, seed=seed)
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
    dataloader = ParallelDataLoader(dataset, batch_size=max_batch_size)
    model.load(ckpt_dir)
    generator = GeneratorForCausalLM(model, tokenizer, max_seq_len, temperature=temperature, top_p=top_p)
    timer = Timer(len(dataloader))
    writer = ParallelDataWriter(os.path.join(log_dir, "results.jsonl"), 'w')
    for data in dataloader:
        timer.step()
        outputs = generator.forward(data['instruction'])
        print(data['instruction'][-1] + "\n" + outputs[-1])
        for instruction, output in zip(data["instruction"], outputs):
            writer.write(json.dumps(dict(instruction=instruction, output=output), ensure_ascii=False) + '\n')


if __name__ == '__main__':
    fire.Fire(main)
