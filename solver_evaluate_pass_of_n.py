import os

import fire
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer, AverageMeter
from src.evaluator import EVALUATORS, Evaluator
from src.generator import GroupGeneratorForCausalLM
from src.modeling import get_parallel_model
from src.parallel.utils import setup_model_parallel
from src.utils import json_dump, convert_dataloader_data_to_list


def compute_pass_of_n(evaluator: Evaluator, results: list, num_samples_per_prompts: int) -> dict:
    for result in results:
        assert isinstance(result["output"], list)
        result["predict"] = []
        for output in result["output"]:
            result["predict"].append(evaluator.eval(output, result["label"]))
    result_dict = {}
    n = 1
    while num_samples_per_prompts // n > 0:
        result_dict[n] = None
        n *= 2
    for n in result_dict.keys():
        meter = AverageMeter()
        for result in results:
            if True in result["predict"][:n]:
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
        seed: int = None
):
    parallel_infos = setup_model_parallel(seed=seed)
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
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, tokenizer)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)
    generator = GroupGeneratorForCausalLM(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        num_samples_per_prompt=num_samples_per_prompt,
        temperature=temperature,
        top_p=top_p
    )
    print(f"Evaluating {task}.........")
    evaluator = EVALUATORS.get(task.lower())()
    results = []
    timer = Timer(len(dataloader))
    for data in tqdm(dataloader):
        timer.step()
        responses = generator.forward(data["instruction"])
        for i, result in enumerate(convert_dataloader_data_to_list(data)):
            result['output'] = responses[i]
            results.append(result)
        print(data["instruction"][0] + '\n' + responses[0][0])

    result_dict = compute_pass_of_n(evaluator, results, num_samples_per_prompt)
    result_name = []
    for n in result_dict.keys():
        print(f"Pass@{n}:", result_dict.get(n))
        result_name.append(f"Pass@{n}:{round(result_dict.get(n), 4)}")
    if parallel_infos.local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        json_dump(results, os.path.join(log_dir, f'results-{"-".join(result_name)}.jsonl'))


if __name__ == '__main__':
    fire.Fire(main)
