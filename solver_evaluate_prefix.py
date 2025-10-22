import os

import fire

from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.evaluator import get_evaluator
from src.generator import PrefixGeneratorForCausalLM
from src.modeling import get_parallel_model
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.initialize import setup_model_parallel
from src.utils import json_dump, print_current_func_args, json_load, convert_dataloader_data_to_list


def process_prefix_file(label_file: str, ratio: float = 0.5) -> JsonDataset:
    datalist = json_load(label_file)
    for data in datalist:
        if isinstance(data["output"], str):
            response_tokens = data["output"].split(" ")
            data["prefix"] = " ".join(response_tokens[: int(ratio * len(response_tokens))])
        elif isinstance(data["output"], list):
            data["prefix"] = []
            for response in data["output"]:
                response_tokens = response.split(" ")
                data["prefix"].append(" ".join(response_tokens[: int(ratio * len(response_tokens))]))
            data["prefix"] = data["prefix"][0]  # TODO
    return JsonDataset(datalist)


def main(
        task: str,
        ckpt_dir: str,
        log_dir: str,
        label_file: str,
        model_type: str = "llama-2-7b",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tokenizer_file: str = None,
        config_file: str = None,
        use_chat_template: bool = False,
        dtype: str = "bfloat16",
        seed: int = None,
        model_parallel_size: int = None
):
    parallel_infos = setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        model_parallel_size=model_parallel_size
    )
    print_current_func_args()
    tokenizer_file = tokenizer_file or ckpt_dir
    config_file = config_file or ckpt_dir

    model, tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=-1,
        dtype=dtype
    )
    dataset = process_prefix_file(label_file)
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, tokenizer)
    model.load(ckpt_dir)
    generator = PrefixGeneratorForCausalLM(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        temperature=temperature,
        top_p=top_p
    )
    dataloader = ParallelDataLoader(dataset, batch_size=max_batch_size)
    print(f"Evaluating {task}")
    results = []
    timer = Timer(len(dataloader))
    for data in dataloader:
        timer.step()
        responses = generator.forward(data["instruction"], data["prefix"])
        datalist = convert_dataloader_data_to_list(data)
        for i, response in enumerate(responses):
            datalist[i]['output'] = response
        results.extend(datalist)
        print(data['instruction'][0].strip() + '\n' + responses[0])
        print("---" * 10)

    evaluator = get_evaluator(task)
    for data in results:
        data['predict'] = evaluator.forward(data['output'], data['label'])
        data['score'] = 1 if evaluator.eval(data['output'], label=data['label']) is True else 0

    print(f"Evaluate Accuracy: {evaluator.accuracy}")
    if parallel_infos.global_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        json_dump(results, os.path.join(log_dir, f'results-{round(evaluator.accuracy, 4)}.jsonl'))


if __name__ == '__main__':
    fire.Fire(main)
