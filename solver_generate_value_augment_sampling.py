import json
import os

import fire
from torch.utils.data import DataLoader

from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.generator import ValueAugmentedSamplingGeneratorForCausalLM
from src.modeling import get_parallel_model, get_parallel_verifier
from src.parallel import setup_model_parallel
from src.utils import convert_dataloader_data_to_list


def run(
        label_file: str,
        log_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,
        max_batch_size: int,
        max_seq_len: int,
        beam_size: int,
        span_size: int,
        tree_size: int,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        verifier_config_file: str = None,
        verifier_tokenizer_file: str = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        dtype: str = "bfloat16",
        use_chat_template: bool = False,
        model_parallel_size: int = None,
        seed: int = None
):
    os.makedirs(log_dir, exist_ok=True)
    setup_model_parallel(model_parallel_size=model_parallel_size, seed=seed)
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir
    verifier_config_file = verifier_config_file or verifier_ckpt_dir
    verifier_tokenizer_file = verifier_tokenizer_file or verifier_ckpt_dir

    policy, policy_tokenizer = get_parallel_model(
        model_type=policy_model_type,
        config_file=policy_config_file,
        tokenizer_file=policy_tokenizer_file,
        max_seq_len=max_seq_len,
        lora_rank=-1,
        dtype=dtype
    )
    policy.load(policy_ckpt_dir)

    verifier, verifier_tokenizer = get_parallel_verifier(
        model_type=verifier_model_type,
        config_file=verifier_config_file,
        tokenizer_file=verifier_tokenizer_file,
        max_seq_len=max_seq_len,
        lora_rank=-1,
        dtype=dtype
    )
    verifier.load(verifier_ckpt_dir)

    dataset = JsonDataset(label_file)
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, tokenizer=policy_tokenizer)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)
    generator = ValueAugmentedSamplingGeneratorForCausalLM(
        policy=policy,
        critic=verifier,
        tokenizer=policy_tokenizer,
        max_seq_len=max_seq_len,
        beam_size=beam_size,
        span_size=span_size,
        tree_size=tree_size,
        temperature=temperature,
        top_p=top_p
    )
    timer = Timer(len(dataloader))
    with open(os.path.join(log_dir, "results.jsonl"), 'w', encoding='utf-8') as writer:
        for data in dataloader:
            timer.step()
            responses, scores = generator.forward(data["instruction"])
            print(data["instruction"][0] + "\n" + responses[0][0])
            for i, result in enumerate(convert_dataloader_data_to_list(data)):
                result["output"] = responses[i]
                result["score"] = scores[i]
                writer.write(json.dumps(result, ensure_ascii=False) + '\n')
            writer.flush()


if __name__ == '__main__':
    fire.Fire(run)
