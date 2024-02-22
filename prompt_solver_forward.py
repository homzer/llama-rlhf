import os
import re

import fire
import torch
from torch.utils.data import DataLoader

from src.dataset import MultiOutputsDataset, JsonDataset
from src.entities import Timer
from src.models.modeling_utils import get_parallel_model
from src.ppo.buffer import LogitsRolloutBuffer
from src.ppo.collector import LogitsBufferCollector
from src.utils import setup_model_parallel, set_barrier


class PromptMultiOutputsDataset(MultiOutputsDataset):
    def __init__(self, f):
        super().__init__(f, False)

    def __getitem__(self, i):
        data = super().__getitem__(i)
        instruction = re.sub(r"\n\nUser:\s*|\n\nAssistant:\s*", "", data['instruction'])
        data['instruction'] = f"\n\nUser: {instruction}\n\nAssistant: "
        return data


class PromptDataset(JsonDataset):
    def __init__(self, f):
        super().__init__(f)

    def __getitem__(self, i):
        data = super().__getitem__(i)
        instruction = re.sub(r"\n\nUser:\s*|\n\nAssistant:\s*", "", data['instruction'])
        data['instruction'] = f"\n\nUser: {instruction}\n\nAssistant: "
        return data


def main(
        ckpt_dir: str,
        save_dir: str,
        model_type: str = "llama-1-7b",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lora_rank: int = 16,
        tokenizer_file: str = None,
        config_file: str = None,
        label_file: str = None,
        seed: int = None,
        use_float16: bool = True
):
    local_rank, world_size = setup_model_parallel(
        use_float16=use_float16, seed=seed
    )
    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    dataset = PromptMultiOutputsDataset(label_file)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)

    model, tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        local_rank=local_rank,
        world_size=world_size,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=lora_rank
    )
    model.load(ckpt_dir, merge_lora=not lora_rank > 0)

    buffer_collector = LogitsBufferCollector(model, tokenizer, max_seq_len)
    rollout_buffer = LogitsRolloutBuffer()
    if local_rank == 0:
        torch.save(rollout_buffer, os.path.join(save_dir, "logits.bin"))
    set_barrier()
    timer = Timer(len(dataloader), episode=100)
    for data in dataloader:
        timer.step()
        rollout_buffer.extend(
            buffer_collector.forward(data['instruction'], data['output'])
        )
        if timer.ticktock % 300 == 0:
            if local_rank == 0:
                old_rollout_buffer = torch.load(os.path.join(save_dir, "logits.bin"), )
                old_rollout_buffer.extend(rollout_buffer)
                torch.save(old_rollout_buffer, os.path.join(save_dir, "logits.bin"), pickle_protocol=4)
            set_barrier()
            rollout_buffer = LogitsRolloutBuffer()

    if local_rank == 0:
        old_rollout_buffer = torch.load(os.path.join(save_dir, "logits.bin"))
        old_rollout_buffer.extend(rollout_buffer)
        torch.save(old_rollout_buffer, os.path.join(save_dir, "logits.bin"), pickle_protocol=4)


if __name__ == '__main__':
    fire.Fire(main)
