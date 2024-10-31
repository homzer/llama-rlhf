import gc
import os

import fire
import torch
from torch.utils.data import DataLoader

from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.generator import GeneratorForCausalLM
from src.modeling import get_parallel_model
from src.parallel.utils import setup_model_parallel, set_barrier
from src.utils import json_dump

REVISER_PROMPT = """###[QUESTION]\n{question}\n\n###[ANSWER]\n{rejected}\n\n###[REVISED ANSWER]\n"""


def get_reviser_dataset(origin_instructions: list, responses: list) -> JsonDataset:
    """ format datalist for reviser """
    assert len(origin_instructions) == len(responses)
    results = []
    for instruction, response in zip(origin_instructions, responses):
        results.append(dict(
            instruction=REVISER_PROMPT.format_map({
                "question": instruction,
                "rejected": response
            }),
        ))
    return JsonDataset(results)


def run(
        log_dir: str,
        label_file: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        reviser_ckpt_dir: str,
        reviser_model_type: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        reviser_config_file: str = None,
        reviser_tokenizer_file: str = None,
        policy_generate_batch_size: int = 1,
        reviser_generate_batch_size: int = 1,
        policy_max_seq_len: int = 1024,
        reviser_max_seq_len: int = 1536,
        dtype: str = "bfloat16",
        temperature: float = 1.0,
        top_p: float = 1.0,
        use_chat_template: bool = False,
        model_parallel_size: int = None,
        seed: int = None
):
    setup_model_parallel(model_parallel_size=model_parallel_size, seed=seed)
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir
    reviser_config_file = reviser_config_file or reviser_ckpt_dir
    reviser_tokenizer_file = reviser_tokenizer_file or reviser_ckpt_dir

    dataset = JsonDataset(label_file)
    origin_instructions = [data["instruction"] for data in dataset]

    # policy collection ...
    policy, policy_tokenizer = get_parallel_model(
        model_type=policy_model_type,
        config_file=policy_config_file,
        max_seq_len=policy_max_seq_len,
        tokenizer_file=policy_tokenizer_file,
        dtype=dtype
    )
    policy.load(policy_ckpt_dir)
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, policy_tokenizer)
    generator = GeneratorForCausalLM(
        model=policy,
        tokenizer=policy_tokenizer,
        max_seq_len=policy_max_seq_len,
        temperature=temperature,
        top_p=top_p
    )
    dataloader = DataLoader(dataset, batch_size=policy_generate_batch_size)
    policy_responses = []
    timer = Timer(len(dataloader))
    for data in dataloader:
        timer.step()
        responses = generator.forward(data["instruction"])
        policy_responses.extend(responses)
        print(data['instruction'][-1] + "\n" + responses[-1])

    policy.cpu()
    del policy
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    # reviser collection ....
    reviser, reviser_tokenizer = get_parallel_model(
        model_type=reviser_model_type,
        config_file=reviser_config_file,
        max_seq_len=reviser_max_seq_len,
        tokenizer_file=reviser_tokenizer_file,
        dtype=dtype
    )
    reviser.load(reviser_ckpt_dir)
    dataset = get_reviser_dataset(origin_instructions, policy_responses)
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, reviser_tokenizer)
    generator = GeneratorForCausalLM(
        model=reviser,
        tokenizer=reviser_tokenizer,
        max_seq_len=reviser_max_seq_len,
        temperature=temperature,
        top_p=top_p
    )
    dataloader = DataLoader(dataset, batch_size=reviser_generate_batch_size)
    reviser_responses = []
    timer = Timer(len(dataloader))
    for data in dataloader:
        timer.step()
        responses = generator.forward(data["instruction"])
        reviser_responses.extend(responses)
        print(data['instruction'][-1] + "\n" + responses[-1])

    results = []
    for instruction, policy_response, reviser_response in zip(
        origin_instructions, policy_responses, reviser_responses
    ):
        results.append(dict(
            instruction=instruction,
            policy_response=policy_response,
            reviser_response=reviser_response
        ))

    os.makedirs(log_dir, exist_ok=True)
    json_dump(results, os.path.join(log_dir, "results.jsonl"))


if __name__ == '__main__':
    fire.Fire(run)
