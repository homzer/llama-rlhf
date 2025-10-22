import os

import fire
from torch.utils.data import DataLoader

from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer, AverageMeter
from src.modeling import get_parallel_verifier
from src.parallel.initialize import setup_model_parallel
from src.rewards.generator import PointwiseVerifierGeneratorForFocalLoss, PointwiseVerifierGeneratorForLastToken
from src.utils import json_dump, convert_dataloader_data_to_list


def main(
        strategy: str,
        ckpt_dir: str,
        log_dir: str,
        label_file: str,
        model_type: str,
        max_seq_len: int,
        max_batch_size: int = 1,
        tokenizer_file: str = None,
        config_file: str = None,
        dtype: str = "bfloat16",
        use_chat_template: bool = False,
        seed: int = None,
):
    tokenizer_file = tokenizer_file or ckpt_dir
    config_file = config_file or ckpt_dir
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

    dataset = JsonDataset(f=label_file)
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, tokenizer)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)

    if "last-token" in strategy:
        generator = PointwiseVerifierGeneratorForLastToken(model, tokenizer, max_seq_len)
    elif "focal-loss" in strategy:
        generator = PointwiseVerifierGeneratorForFocalLoss(model, tokenizer, max_seq_len)
    else:
        raise ValueError(strategy)

    results = []
    meter = AverageMeter()
    timer = Timer(len(dataloader))
    for data in dataloader:
        outputs = generator.forward(data['instruction'], data['output'])
        timer.step()
        for i, result in enumerate(convert_dataloader_data_to_list(data)):
            result['score'] = outputs.scores[i]
            results.append(result)
            assert result['label'] in [0, 1]
            meter.forward(1 if (
                    (result['label'] == 0 and result['score'] <= 0.5) or
                    (result['label'] == 1 and result['score'] >= 0.5)
            ) else 0)
    print("Acc:", meter.average)
    if parallel_infos.local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        json_dump(results, os.path.join(log_dir, "results.json"), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
