import os

import fire
import torch
from torch.utils.data import DataLoader

from src.rewards.generator import VerifierGeneratorForLastToken, VerifierGeneratorForMeanScores, \
    VerifierGeneratorForFocalLoss, VerifierGeneratorForFocalMeanScores
from src.dataset import PairwiseDataset, ChatTemplateDataset
from src.entities import Timer
from src.generator import GeneratorForVerifier
from src.modeling import get_parallel_verifier
from src.parallel.initialize import setup_model_parallel
from src.utils import json_dump


def main(
        ckpt_dir: str,
        log_dir: str,
        label_file: str,
        model_type: str = "qwen-2-7b",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        tokenizer_file: str = None,
        config_file: str = None,
        dtype: str = "bfloat16",
        use_chat_template: bool = False,
        seed: int = None,
        strategy: str = "sum-score",  # "last-token"
):
    os.makedirs(log_dir, exist_ok=True)
    tokenizer_file = ckpt_dir if tokenizer_file is None else tokenizer_file
    config_file = ckpt_dir if config_file is None else config_file
    parallel_infos = setup_model_parallel(seed=seed)
    model, tokenizer = get_parallel_verifier(
        model_type=model_type,
        config_file=config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=-1,
        dtype=dtype
    )
    model.load(ckpt_dir)

    dataset = PairwiseDataset(f=label_file)
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, tokenizer)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)

    if "last-token" in strategy:
        generator = VerifierGeneratorForLastToken(model, tokenizer, max_seq_len)
    elif "mean-score" in strategy:
        generator = VerifierGeneratorForMeanScores(model, tokenizer, max_seq_len)
    elif "focal-loss" in strategy:
        generator = VerifierGeneratorForFocalLoss(model, tokenizer, max_seq_len)
    elif "focal-mean-score" in strategy:
        generator = VerifierGeneratorForFocalMeanScores(model, tokenizer, max_seq_len)
    else:
        raise ValueError(strategy)
    datalist = []
    timer = Timer(len(dataloader))
    for data in dataloader:
        chosen_outputs = generator.forward(data['instruction'], data['chosen'])
        rejected_outputs = generator.forward(data['instruction'], data['rejected'])
        timer.step()
        for i in range(len(data['instruction'])):
            chosen_score = chosen_outputs.scores[i]
            rejected_score = rejected_outputs.scores[i]
            datalist.append(dict(
                instruction=data['instruction'][i],
                chosen=data['chosen'][i],
                rejected=data['rejected'][i],
                chosen_score=chosen_score,
                rejected_score=rejected_score
            ))
    if parallel_infos.local_rank == 0:
        json_dump(datalist, os.path.join(log_dir, "results.json"), indent=4)

    # For token-level scores
    datalist = []
    generator = GeneratorForVerifier(model, tokenizer, max_seq_len)
    timer = Timer(len(dataloader))
    for data in dataloader:
        timer.step()
        chosen_examples = generator.prepare_for_generation(data['instruction'], data['chosen'])
        rejected_examples = generator.prepare_for_generation(data['instruction'], data['rejected'])
        with torch.no_grad():
            chosen_scores = model.forward(chosen_examples.tokens).scores
            rejected_scores = model.forward(rejected_examples.tokens).scores
        if "discriminative-dpo" in strategy:
            chosen_scores = torch.nn.functional.logsigmoid(chosen_scores)
            rejected_scores = torch.nn.functional.logsigmoid(rejected_scores)
        for i in range(len(data['instruction'])):
            chosen_token_scores = torch.masked_select(
                chosen_scores[i].cpu(), chosen_examples.masks[i]
            )
            rejected_token_scores = torch.masked_select(
                rejected_scores[i].cpu(), rejected_examples.masks[i]
            )
            datalist.append(dict(
                instruction=data['instruction'][i],
                chosen=data['chosen'][i],
                rejected=data['rejected'][i],
                chosen_token_scores=chosen_token_scores.tolist(),
                rejected_token_scores=rejected_token_scores.tolist()
            ))
    if parallel_infos.local_rank == 0:
        json_dump(datalist, os.path.join(log_dir, "token-results.jsonl"))


if __name__ == '__main__':
    fire.Fire(main)
