import gc
import os

import fire
import torch.cuda
from torch.utils.data import DataLoader

from src.dataset import JsonDataset, ChatTemplateDataset, MultiOutputsDataset
from src.entities import Timer
from src.generator import GroupGeneratorForCausalLM, GeneratorForVerifier
from src.modeling import get_parallel_model, get_parallel_verifier
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.utils import convert_dataloader_data_to_list, json_dump


def format_multi_outputs_datalist_with_scores(datalist: list, num_samples_per_prompt: int) -> list:
    results = []
    assert len(datalist) % num_samples_per_prompt == 0
    origin_size = len(datalist) // num_samples_per_prompt
    for i in range(origin_size):
        results.append(datalist[i].copy())
        results[i]["output"] = []
        results[i]["score"] = []

    for i, data in enumerate(datalist):
        i = i % origin_size
        assert results[i]["instruction"] == data["instruction"]
        results[i]["output"].append(data["output"])
        results[i]["score"].append(data["score"])

    return results


def main(
        label_file: str,
        log_dir: str,
        num_samples_per_prompt: int,

        policy_ckpt_dir: str,
        policy_model_type: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,
        policy_tokenizer_file: str = None,
        policy_config_file: str = None,
        verifier_tokenizer_file: str = None,
        verifier_config_file: str = None,
        max_seq_len: int = 1024,
        max_generate_batch_size: int = 1,
        max_forward_batch_size: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        dtype: str = "bfloat16",
        use_chat_template: bool = False,
        use_last_token_reward: bool = True,
        seed: int = None
):
    os.makedirs(log_dir, exist_ok=True)
    setup_model_parallel(seed=seed)
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir
    policy_config_file = policy_config_file or policy_ckpt_dir

    # policy buffer collecting
    policy, policy_tokenizer = get_parallel_model(
        model_type=policy_model_type,
        config_file=policy_config_file,
        tokenizer_file=policy_tokenizer_file,
        max_seq_len=max_seq_len,
        lora_rank=-1,
        dtype=dtype
    )
    policy.load(policy_ckpt_dir)
    generator = GroupGeneratorForCausalLM(
        model=policy,
        tokenizer=policy_tokenizer,
        max_seq_len=max_seq_len,
        num_samples_per_prompt=num_samples_per_prompt,
        temperature=temperature,
        top_p=top_p,
    )
    dataset = JsonDataset(label_file)
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, policy_tokenizer)
    dataloader = DataLoader(dataset, batch_size=max_generate_batch_size)
    results = []
    timer = Timer(len(dataloader))
    for data in dataloader:
        timer.step()
        responses = generator.forward(data["instruction"])
        print(data['instruction'][-1] + "\n" + responses[-1][0])
        for i, result in enumerate(convert_dataloader_data_to_list(data)):
            result["instruction"] = result.pop("origin_instruction")
            result["output"] = responses[i]
            results.append(result)

    policy.cpu()
    del policy
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    # verifier buffer collecting
    verifier, verifier_tokenizer = get_parallel_verifier(
        model_type=verifier_model_type,
        config_file=verifier_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=verifier_tokenizer_file,
        lora_rank=-1,
        dtype=dtype
    )
    verifier.load(verifier_ckpt_dir)
    generator = GeneratorForVerifier(
        model=verifier,
        tokenizer=verifier_tokenizer,
        max_seq_len=max_seq_len,
        reduce="last" if use_last_token_reward else "mean"
    )
    dataset = MultiOutputsDataset(results)
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, tokenizer=verifier_tokenizer)
    dataloader = DataLoader(dataset, batch_size=max_forward_batch_size)
    results = []
    timer = Timer(len(dataloader), episode=10)
    for data in dataloader:
        timer.step()
        scores = generator.forward(data["instruction"], data["output"]).scores
        for i, result in enumerate(convert_dataloader_data_to_list(data)):
            result["score"] = scores[i]
            results.append(result)

    results = format_multi_outputs_datalist_with_scores(results, num_samples_per_prompt)
    json_dump(results, os.path.join(log_dir, "results.jsonl"))


if __name__ == '__main__':
    fire.Fire(main)
