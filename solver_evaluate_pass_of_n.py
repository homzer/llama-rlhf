import os

import fire

from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import AverageMeter
from src.evaluator import DataParallelPolicyEvaluator
from src.modeling import get_parallel_model
from src.parallel.initialize import setup_model_parallel
from src.utils import json_dump, print_current_func_args


def process_results(results: list, num_samples_per_prompts: int) -> list:
    results = sorted(results, key=lambda x: x['instruction'])
    datalist = []
    for i in range(0, len(results), num_samples_per_prompts):
        assert len(set([result["instruction"] for result in results[i: i + num_samples_per_prompts]])) == 1
        data = dict(
            instruction=results[i]["instruction"],
            label=results[i]["label"],
            output=[],
            predict=[],
            score=[]
        )
        for j in range(i, i + num_samples_per_prompts):
            data["output"].append(results[j]["output"])
            data["predict"].append(results[j]["predict"])
            data["score"].append(results[j]["score"])
        datalist.append(data)
    return datalist


def compute_pass_of_n(results: list, num_samples_per_prompts: int) -> dict:
    result_dict = {}
    n = 1
    while num_samples_per_prompts // n > 0:
        result_dict[n] = None
        n *= 2
    for n in result_dict.keys():
        meter = AverageMeter()
        for data in results:
            if 1 in data["score"][:n]:
                meter.forward(1)
            else:
                meter.forward(0)
        result_dict[n] = meter.average
    return result_dict


def main(
        task: str,
        ckpt_dir: str,
        log_dir: str,
        label_file: str,
        model_type: str,
        max_seq_len: int,
        num_samples_per_prompt: int = 1,
        max_batch_size: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        tokenizer_file: str = None,
        config_file: str = None,
        use_chat_template: bool = False,
        dtype: str = "bfloat16",
        seed: int = None,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
):
    parallel_infos = setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
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
    model.load(ckpt_dir)
    dataset = JsonDataset(label_file)
    dataset.repeat(n=num_samples_per_prompt).shuffle()
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, tokenizer)

    evaluator = DataParallelPolicyEvaluator(
        model=model,
        tokenizer=tokenizer,
        batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        temperature=temperature,
        top_p=top_p
    )
    evaluator_outputs = evaluator.forward(task=task, dataset=dataset)

    results = process_results(evaluator_outputs.datalist, num_samples_per_prompt)
    result_dict = compute_pass_of_n(results, num_samples_per_prompt)
    result_name = []
    for n in result_dict.keys():
        print(f"Pass@{n}:", result_dict.get(n))
        result_name.append(f"Pass@{n}:{round(result_dict.get(n), 4)}")
    if parallel_infos.global_rank == 0:
        os.makedirs(os.path.join(log_dir, task), exist_ok=True)
        json_dump(results, os.path.join(log_dir, task, f'results-{"-".join(result_name)}.jsonl'))


if __name__ == '__main__':
    fire.Fire(main)
