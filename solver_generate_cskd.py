import json
import os

import fire
from torch.utils.data import DataLoader

from src.dataset import MultiOutputsDataset
from src.entities import Timer
from src.generator import GeneratorForCausalLM
from src.modeling import get_parallel_model
from src.utils import json_load
from src.parallel.utils import setup_model_parallel, set_barrier


def main(
        ckpt_dir: str,
        log_dir: str,
        label_file: str,
        model_type: str = "llama-2-70b",
        max_seq_len: int = 1024,
        max_batch_size: int = 384,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tokenizer_file: str = None,
        config_file: str = None,
        seed: int = None,
        sequential_load: bool = False,
        begin: int = None,
        end: int = None
):
    os.makedirs(log_dir, exist_ok=True)
    parallel_infos = setup_model_parallel(seed=seed)
    if tokenizer_file is None:
        tokenizer_file = ckpt_dir
    if config_file is None:
        config_file = ckpt_dir

    model, tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=-1,
        dtype='bfloat16'
    )
    model.load(ckpt_dir, merge_lora=True, sequential_load=sequential_load)
    generator = GeneratorForCausalLM(model, tokenizer, max_seq_len, temperature=temperature, top_p=top_p)
    save_name = "results.jsonl"
    datalist = json_load(label_file)
    if begin is not None:
        assert end is not None
        datalist = datalist[begin: end]
        save_name = f"results-{begin}-{end}.jsonl"
    if os.path.exists(os.path.join(log_dir, save_name)):
        datalist = datalist[len(json_load(os.path.join(log_dir, save_name))):]
    dataset = MultiOutputsDataset(datalist)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)
    timer = Timer(len(dataloader))
    with open(os.path.join(log_dir, save_name), 'a', encoding='utf-8') as writer:
        for data in dataloader:
            timer.step()
            outputs = generator.forward(data['instruction'])
            for output, instruction in zip(outputs, data['instruction']):
                if parallel_infos.local_rank == 0:
                    writer.write(json.dumps({"instruction": instruction, 'output': [output]}, ensure_ascii=False) + '\n')
                    writer.flush()
            set_barrier()
            print(data['instruction'][0], outputs[0])


if __name__ == '__main__':
    fire.Fire(main)
