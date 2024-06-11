import os

import fire
from torch.utils.data import DataLoader

from src.dataset import MultiOutputsDataset
from src.entities import Timer
from src.models.modeling_utils import get_parallel_model
from src.ppo.buffer import LogitsRolloutBuffer
from src.ppo.collector import LogitsBufferCollector
from src.utils import setup_model_parallel, json_load, set_barrier


def main(
        ckpt_dir: str,
        save_dir: str,
        label_file: str,
        model_type: str = "llama-2-70b",
        max_seq_len: int = 768,
        forward_batch_size: int = 48,
        tokenizer_file: str = None,
        config_file: str = None,
        chunk_size: int = 40000,
        begin_epoch: int = 0,
        logits_topk: int = 5,
):
    os.makedirs(save_dir, exist_ok=True)
    local_rank, world_size = setup_model_parallel()
    if tokenizer_file is None:
        tokenizer_file = ckpt_dir
    if config_file is None:
        config_file = ckpt_dir

    model, tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        local_rank=local_rank,
        world_size=world_size,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=-1,
        dtype='bfloat16',
    )
    # randomly sample a model checkpoint
    model.load(ckpt_dir, merge_lora=True)
    buffer_collector = LogitsBufferCollector(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        logits_topk=logits_topk
    )

    datalist = json_load(label_file)
    epochs = len(datalist) // chunk_size
    for epoch in range(begin_epoch, epochs):
        print(f"Epoch - {epoch} of {epochs}")
        dataset = MultiOutputsDataset(datalist[epoch * chunk_size: (epoch + 1) * chunk_size])
        if len(dataset) == 0:
            return
        dataloader = DataLoader(dataset, batch_size=forward_batch_size)
        rollout_buffer = LogitsRolloutBuffer()
        timer = Timer(len(dataloader), episode=10)
        for data in dataloader:
            timer.step()
            rollout_buffer.extend(
                buffer_collector.forward(data['instruction'], data['output'])
            )

        if local_rank == 0:
            rollout_buffer.save(save_dir, overwrite=(epoch == 0))
        set_barrier()


if __name__ == '__main__':
    fire.Fire(main)
