import json
import os

import fire

from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.generator import GroupGeneratorForCausalLM
from src.modeling import get_parallel_model
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.data_parallel.datawriter import ParallelDataWriter
from src.parallel.initialize import setup_model_parallel
from src.utils import convert_dataloader_data_to_list, print_current_func_args


def main(
        ckpt_dir: str,
        label_file: str,
        log_dir: str,
        model_type: str,
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lora_rank: int = -1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tokenizer_file: str = None,
        config_file: str = None,
        use_chat_template: bool = False,
        num_samples_per_prompt: int = 1,
        dtype: str = "bfloat16",
        model_parallel_size: int = None,
        seed: int = None
):
    setup_model_parallel(
        model_parallel_size=model_parallel_size,
        seed=seed,
        log_dir=log_dir
    )
    print_current_func_args()
    tokenizer_file = tokenizer_file or ckpt_dir
    config_file = config_file or ckpt_dir

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
    generator = GroupGeneratorForCausalLM(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        temperature=temperature,
        top_p=top_p,
        num_samples_per_prompt=num_samples_per_prompt
    )
    timer = Timer(len(dataloader))
    os.makedirs(log_dir, exist_ok=True)
    writer = ParallelDataWriter(os.path.join(log_dir, "results.jsonl"), 'w')
    for data in dataloader:
        timer.step()
        responses = generator.forward(data['instruction'])
        for i, result in enumerate(convert_dataloader_data_to_list(data)):
            result["output"] = responses[i]
            writer.write(json.dumps(result, ensure_ascii=False) + '\n')
    writer.flush()


if __name__ == '__main__':
    fire.Fire(main)
