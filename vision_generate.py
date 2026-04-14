import json
import os

import fire

from src.dataset import VisionDataset
from src.entities import Timer
from src.generator_vl import GeneratorForVisualLM
from src.models.modeling import AutoModelForVisualLM
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.data_parallel.datawriter import ParallelDataWriter
from src.parallel.initialize import setup_model_parallel
from src.tokenizers.processor import AutoProcessor
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
        dtype: str = "bfloat16",
        model_parallel_size: int = None,
        system_prompt: str = None,
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

    model = AutoModelForVisualLM.from_pretrained(
        model_type=model_type,
        config_file=config_file,
        max_seq_len=max_seq_len,
        lora_rank=lora_rank,
        dtype=dtype
    )
    processor = AutoProcessor.from_pretrained(
        model_type=model_type,
        tokenizer_file=tokenizer_file
    )
    processor.system_prompt = system_prompt
    model.load(ckpt_dir)
    generator = GeneratorForVisualLM(
        model=model,
        processor=processor,
        max_seq_len=max_seq_len,
        temperature=temperature,
        top_p=top_p,
    )
    dataset = VisionDataset(label_file)
    os.makedirs(log_dir, exist_ok=True)
    writer = ParallelDataWriter(os.path.join(log_dir, "results.jsonl"), 'w')
    dataloader = ParallelDataLoader(dataset, batch_size=max_batch_size)
    timer = Timer(len(dataloader))
    for data in dataloader:
        timer.step()
        responses = generator.forward(data['text'], videos=data['video'])
        print(data['text'][-1] + '\n=======\n' + responses[-1])
        for i, result in enumerate(convert_dataloader_data_to_list(data)):
            result["output"] = responses[i]
            writer.write(json.dumps(result, ensure_ascii=False) + '\n')
        writer.flush()


if __name__ == '__main__':
    fire.Fire(main)
