import gc
import json
import os

import fire
import torch.cuda
from torch.utils.data import DataLoader

from src.parallel.dataloader import ParallelDataLoader
from src.parallel.utils import setup_model_parallel, set_barrier
from src.parallel.datawriter import ParallelDataWriter
from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.generator import DiversityGeneratorForCausalLM, GeneratorForVerifier
from src.modeling import get_parallel_model, get_parallel_verifier
from src.utils import convert_dataloader_data_to_list


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
        diverse_prob: float = 0.1,
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
    generator = DiversityGeneratorForCausalLM(
        model=policy,
        tokenizer=policy_tokenizer,
        max_seq_len=max_seq_len,
        num_samples_per_prompt=num_samples_per_prompt,
        temperature=temperature,
        top_p=top_p,
        diverse_prob=diverse_prob
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
        for i, result in enumerate(convert_dataloader_data_to_list(data)):
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


