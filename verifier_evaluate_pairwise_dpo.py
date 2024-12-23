import gc
import os

import fire
import torch
from torch.utils.data import DataLoader

from src.rewards.generator import VerifierGeneratorForDPO
from src.dataset import PairwiseDataset, ChatTemplateDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.parallel import setup_model_parallel, set_barrier
from src.parallel.utils import get_local_rank
from src.ppo.buffer import LogitsRolloutBuffer
from src.ppo.collector import LogitsBufferCollector
from src.ppo.generator import LogitsGeneratorForCausalLM
from src.utils import json_dump


def main(
        policy_ckpt_dir: str,
        reference_ckpt_dir: str,
        log_dir: str,
        label_file: str,
        buffer_dir: str = None,
        model_type: str = "qwen",
        max_seq_len: int = 512,
        max_batch_size: int = 32,
        tokenizer_file: str = None,
        config_file: str = None,
        dtype: str = "bfloat16",
        use_chat_template: bool = False,
        seed: int = None,
):
    os.makedirs(log_dir, exist_ok=True)
    tokenizer_file = policy_ckpt_dir if tokenizer_file is None else tokenizer_file
    config_file = policy_ckpt_dir if config_file is None else config_file
    buffer_dir = log_dir if buffer_dir is None else buffer_dir
    chosen_buffer_file = os.path.join(buffer_dir, "chosen", "buffer.jsonl")
    rejected_buffer_file = os.path.join(buffer_dir, "rejected", "buffer.jsonl")
    setup_model_parallel(seed=seed)
    dataset = PairwiseDataset(f=label_file)

    if not os.path.exists(chosen_buffer_file) or not os.path.exists(rejected_buffer_file):
        reference, reference_tokenizer = get_parallel_model(
            model_type=model_type,
            config_file=config_file,
            max_seq_len=max_seq_len,
            tokenizer_file=tokenizer_file,
            lora_rank=-1,
            dtype=dtype
        )
        reference.load(reference_ckpt_dir)
        reference_buffer_collector = LogitsBufferCollector(
            model=reference,
            tokenizer=reference_tokenizer,
            max_seq_len=max_seq_len
        )
        reference_chosen_rollout_buffer = LogitsRolloutBuffer()
        reference_rejected_rollout_buffer = LogitsRolloutBuffer()
        reference_dataloader = DataLoader(
            ChatTemplateDataset(dataset, reference_tokenizer) if use_chat_template else dataset,
            batch_size=max_batch_size
        )
        timer = Timer(len(reference_dataloader), episode=10)
        for data in reference_dataloader:
            timer.step()
            reference_chosen_rollout_buffer.extend(
                reference_buffer_collector.forward(
                    data["instruction"], data["chosen"]
                )
            )
            reference_rejected_rollout_buffer.extend(
                reference_buffer_collector.forward(
                    data["instruction"], data["rejected"]
                )
            )

        reference.cpu()
        del reference
        del reference_buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()
        if get_local_rank() == 0:
            reference_chosen_rollout_buffer.save(os.path.join(log_dir, "chosen"))
            reference_rejected_rollout_buffer.save(os.path.join(log_dir, "rejected"))
        set_barrier()
    else:
        reference_chosen_rollout_buffer = LogitsRolloutBuffer().load(chosen_buffer_file)
        reference_rejected_rollout_buffer = LogitsRolloutBuffer().load(rejected_buffer_file)

    policy, policy_tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=-1,
        dtype=dtype
    )
    policy.load(policy_ckpt_dir)
    generator = VerifierGeneratorForDPO(policy, policy_tokenizer, max_seq_len)
    timer = Timer(len(reference_chosen_rollout_buffer) // max_batch_size, episode=10)
    datalist = []
    policy_dataloader = DataLoader(
        ChatTemplateDataset(dataset, policy_tokenizer) if use_chat_template else dataset,
        batch_size=max_batch_size
    )
    for data, ref_chosen_log_probs, ref_rejected_log_probs in zip(
            policy_dataloader,
            reference_chosen_rollout_buffer.get_logps(max_batch_size),
            reference_rejected_rollout_buffer.get_logps(max_batch_size)
    ):
        timer.step()
        chosen_outputs = generator.forward(
            instructions=data["instruction"],
            outputs=data["chosen"],
            reference_log_probs=ref_chosen_log_probs
        )
        rejected_outputs = generator.forward(
            instructions=data["instruction"],
            outputs=data["rejected"],
            reference_log_probs=ref_rejected_log_probs
        )
        for i in range(len(data["instruction"])):
            datalist.append(dict(
                instruction=data["instruction"][i],
                chosen=data["chosen"][i],
                rejected=data["rejected"][i],
                chosen_score=chosen_outputs.scores[i],
                rejected_score=rejected_outputs.scores[i]
            ))
    json_dump(datalist, os.path.join(log_dir, "results.json"), indent=4)

    datalist = []
    timer = Timer(len(reference_chosen_rollout_buffer) // max_batch_size, episode=10)
    generator = LogitsGeneratorForCausalLM(policy, policy_tokenizer, max_seq_len)
    for chosen_data, rejected_data in zip(
            reference_chosen_rollout_buffer.get(max_batch_size),
            reference_rejected_rollout_buffer.get(max_batch_size)
    ):
        timer.step()
        chosen_examples = generator.prepare_for_generation(chosen_data.instructions, chosen_data.outputs)
        chosen_outputs = generator.forward(chosen_data.instructions, chosen_data.outputs)
        rejected_examples = generator.prepare_for_generation(rejected_data.instructions, rejected_data.outputs)
        rejected_outputs = generator.forward(rejected_data.instructions, rejected_data.outputs)
        for i in range(len(chosen_data.instructions)):
            chosen_token_scores = (torch.masked_select(
                chosen_outputs.tokens_logps[i].cpu(), chosen_examples.masks[i]
            ) - torch.masked_select(
                chosen_data.output_tokens_logps[i].cpu(), chosen_examples.masks[i]
            )) * 0.1
            rejected_token_scores = (torch.masked_select(
                rejected_outputs.tokens_logps[i].cpu(), rejected_examples.masks[i]
            ) - torch.masked_select(
                rejected_data.output_tokens_logps[i].cpu(), rejected_examples.masks[i]
            )) * 0.1
            datalist.append(dict(
                instruction=chosen_data.instructions[i],
                chosen=chosen_data.outputs[i],
                rejected=rejected_data.outputs[i],
                chosen_token_scores=chosen_token_scores.tolist(),
                rejected_token_scores=rejected_token_scores.tolist()
            ))
    json_dump(datalist, os.path.join(log_dir, "token-results.jsonl"))


if __name__ == '__main__':
    fire.Fire(main)
