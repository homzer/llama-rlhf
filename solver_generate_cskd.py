import json
import os

import fire
from torch.utils.data import DataLoader

from src.dataset import MultiOutputsDataset
from src.entities import Timer
from src.generator import GeneratorForCausalLM
from src.models.modeling_utils import get_parallel_model
from src.utils import setup_model_parallel, json_load, set_barrier


def main(
        ckpt_dir: str,
        log_dir: str,
        label_file: str,
        model_type: str = "llama-2-70b",
        max_seq_len: int = 1024,
        max_batch_size: int = 384,
        t: float = 0.0,
        p: float = 1.0,
        tokenizer_file: str = None,
        config_file: str = None,
        seed: int = None,
        sequential_load: bool = False
):
    os.makedirs(log_dir, exist_ok=True)
    local_rank, world_size = setup_model_parallel(seed=seed)
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
        lora_rank=-1,
        dtype='bfloat16'
    )
    model.load(ckpt_dir, merge_lora=True, sequential_load=sequential_load)
    generator = GeneratorForCausalLM(model, tokenizer, max_seq_len)
    datalist = json_load(label_file)
    if os.path.exists(os.path.join(log_dir, "results.jsonl")):
        datalist = datalist[len(json_load(os.path.join(log_dir, "results.jsonl"))):]
    dataset = MultiOutputsDataset(datalist)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)
    timer = Timer(len(dataloader))
    with open(os.path.join(log_dir, 'results.jsonl'), 'a', encoding='utf-8') as writer:
        for data in dataloader:
            timer.step()
            outputs = generator.forward(data['instruction'], t=t, p=p)
            for output, instruction in zip(outputs, data['instruction']):
                if local_rank == 0:
                    writer.write(json.dumps({"instruction": instruction, 'output': [output]}, ensure_ascii=False) + '\n')
                    writer.flush()
            set_barrier()
            print(data['instruction'][0], outputs[0])


if __name__ == '__main__':
    fire.Fire(main)
