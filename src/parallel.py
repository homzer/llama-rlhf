import collections
import os
import re
import sys

import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel, \
    get_model_parallel_rank, get_model_parallel_group, get_data_parallel_group, \
    get_data_parallel_rank, get_data_parallel_world_size, get_model_parallel_src_rank, \
    get_model_parallel_world_size
from torch.distributed import init_process_group
from torch.utils.data import DistributedSampler, DataLoader, Dataset

from src.utils import set_seed


class ParallelDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_data_parallel_world_size(),
            rank=get_data_parallel_rank(),
            shuffle=shuffle
        )
        super().__init__(dataset, batch_size=batch_size, sampler=sampler)


class ParallelDataWriter:
    def __init__(self, file: str, mode: str = 'w'):
        self.global_rank = int(os.environ.get("RANK"))
        self.model_parallel_src_rank = get_model_parallel_src_rank()
        self.model_parallel_world_size = get_model_parallel_world_size()
        self.worker_id = self.model_parallel_src_rank // self.model_parallel_world_size
        self.file = self.format_file(file)
        self.writer = open(self.file, mode=mode, encoding="utf-8")

    def format_file(self, file: str) -> str:
        match = re.search(r".+(\..+)$", file)
        if match:
            return re.sub(rf"{match.group(1)}$", f".worker.{self.worker_id}{match.group(1)}", file)
        return f"{file}.worker.{self.worker_id}"

    def __del__(self):
        self.writer.close()

    def write(self, s: str):
        if self.global_rank == self.model_parallel_src_rank:
            self.writer.write(s)
            self.writer.flush()
        set_model_parallel_barrier()


def get_data_parallel_src_rank() -> int:
    """Calculate the global rank corresponding to a local rank zero
    in the data parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_data_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


ParallelInfos = collections.namedtuple("ParallelInfos", [
    "global_rank",
    "local_rank",
    "world_size",
    "model_parallel_world_size",
    "model_parallel_rank",
    "model_parallel_src_rank",
    "data_parallel_world_size",
    "data_parallel_rank",
    "data_parallel_src_rank"
])


def setup_model_parallel(
        model_parallel_size: int = None, seed: int = None
) -> ParallelInfos:
    global_rank: int = int(os.environ.get("RANK"))
    local_rank: int = int(os.environ.get("LOCAL_RANK"))
    world_size: int = int(os.environ.get("WORLD_SIZE"))

    init_process_group("nccl")
    initialize_model_parallel(model_parallel_size or world_size)

    model_parallel_world_size: int = get_model_parallel_world_size()
    model_parallel_rank: int = get_model_parallel_rank()
    model_parallel_src_rank: int = get_model_parallel_src_rank()
    data_parallel_world_size: int = get_data_parallel_world_size()
    data_parallel_rank: int = get_data_parallel_rank()
    data_parallel_src_rank: int = get_data_parallel_src_rank()

    if global_rank != model_parallel_src_rank:
        sys.stdout = open(os.devnull, "w")

    torch.cuda.set_device(local_rank)
    # seed must be the same in all processes
    set_seed(seed or 1)

    return ParallelInfos(
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        model_parallel_world_size=model_parallel_world_size,
        model_parallel_rank=model_parallel_rank,
        model_parallel_src_rank=model_parallel_src_rank,
        data_parallel_world_size=data_parallel_world_size,
        data_parallel_rank=data_parallel_rank,
        data_parallel_src_rank=data_parallel_src_rank
    )


def set_barrier():
    """ make sure that all other processes cannot continue until reach this op. """
    torch.distributed.barrier()


def set_model_parallel_barrier():
    """ make sure that all other processes in model parallel group cannot continue until reach this op. """
    torch.distributed.barrier(get_model_parallel_group())


def set_data_parallel_barrier():
    """ make sure that all other processes in data parallel group cannot continue until reach this op. """
    torch.distributed.barrier(get_data_parallel_group())
