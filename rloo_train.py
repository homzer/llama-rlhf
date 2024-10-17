import gc
import os

import fire
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.modeling import get_parallel_model, get_parallel_verifier
from src.parallel.utils import setup_model_parallel, set_barrier
from src.ppo.buffer import CriticRolloutBuffer, RolloutBuffer, ActorRolloutBuffer, LogitsRolloutBuffer
from src.ppo.collector import CriticBufferCollector, DiversityActorBufferCollector, LogitsBufferCollector
from src.ppo.trainer import ParallelActorTrainerForCausalLM, ParallelCriticTrainerForCausalLM
from src.utils import masked_mean, json_load


def collect_actor_buffer(
        actor_model_type: str,
        actor_config_file: str,
        max_seq_len: int,
        actor_tokenizer_file: str,
        dtype: str,
        actor_ckpt_dir: str,
        epoch: int,
        actor_save_dir: str,
        use_chat_template: bool,
        dataset: JsonDataset,
        max_generate_batch_size: int,
        num_samples_per_prompt: int,
        diverse_prob: float
) -> ActorRolloutBuffer:
    actor, actor_tokenizer = get_parallel_model(
        model_type=actor_model_type,
        config_file=actor_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=actor_tokenizer_file,
        lora_rank=-1,
        dtype=dtype
    )
    actor.load(actor_ckpt_dir if epoch == 0 else os.path.join(actor_save_dir, f"epoch-{epoch}"))
    actor_buffer_collector = DiversityActorBufferCollector(
        actor=actor,
        tokenizer=actor_tokenizer,
        max_seq_len=max_seq_len,
        temperature=1.0,
        num_samples_per_prompt=num_samples_per_prompt,
        diverse_prob=diverse_prob,
    )
    actor_rollout_buffer = ActorRolloutBuffer()
    print('Actor buffer collecting ...')
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, actor_tokenizer)
    dataloader = DataLoader(dataset, batch_size=max_generate_batch_size)
    timer = Timer(len(dataloader))
    for data in dataloader:
        timer.step()
        actor_rollout_buffer.extend(actor_buffer_collector.forward(data['instruction']))
        print(actor_rollout_buffer.instructions[-1], '\n', actor_rollout_buffer.responses[-1])

    actor.cpu()
    del actor
    del actor_buffer_collector
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    return actor_rollout_buffer


def run(
        train_file: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_save_dir: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        verifier_config_file: str = None,
        verifier_tokenizer_file: str = None,
        policy_lora_rank: int = -1,
        policy_lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 1,
        max_forward_batch_size: int = 1,
        max_seq_len: int = 1024,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-6,
        num_samples_per_prompt: int = 4,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
):
    setup_model_parallel()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir
    verifier_config_file = verifier_config_file or verifier_ckpt_dir
    verifier_tokenizer_file = verifier_tokenizer_file or verifier_ckpt_dir

    datalist = json_load(train_file)
    chunk_size = chunk_size or len(datalist)
    epochs = len(datalist) // chunk_size
    for epoch in range(begin_epoch, epochs):
        print(f"Epoch - {epoch} of {epochs}")
        dataset = JsonDataset(f=datalist[epoch * chunk_size: (epoch + 1) * chunk_size])

        pass


if __name__ == '__main__':
    fire.Fire(run)
