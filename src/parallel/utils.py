import collections
import os
import sys

import torch
from fairscale.nn.model_parallel.initialize import get_data_parallel_world_size, initialize_model_parallel, \
    get_model_parallel_world_size, get_model_parallel_rank, get_model_parallel_src_rank, get_data_parallel_rank, \
    get_model_parallel_group, get_data_parallel_group, get_pipeline_parallel_group
from torch.distributed import init_process_group

from src.utils import set_seed


def get_rank() -> int:
    """Return my global rank."""
    return int(os.environ.get("RANK"))


def get_local_rank() -> int:
    """Return my local rank."""
    return int(os.environ.get("LOCAL_RANK"))


def get_world_size() -> int:
    """Return the world size of the global group."""
    return int(os.environ.get("WORLD_SIZE"))


def get_data_parallel_src_rank() -> int:
    """Calculate the global rank corresponding to a local rank zero
    in the data parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_data_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_pipeline_parallel_rank() -> int:
    """Return my rank for the pipeline parallel group."""
    return torch.distributed.get_rank(group=get_pipeline_parallel_group())


def get_pipeline_parallel_world_size() -> int:
    """Return world size for the pipeline parallel group."""
    return torch.distributed.get_world_size(group=get_pipeline_parallel_group())


def get_pipeline_parallel_src_rank() -> int:
    """Calculate the global rank corresponding to a local rank zero
    in the pipeline parallel group."""
    global_rank = torch.distributed.get_rank()
    local_work_size = get_pipeline_parallel_world_size()
    return (global_rank // local_work_size) * local_work_size


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
        model_parallel_size: int = None, pipeline_parallel_size: int = 1, seed: int = None
) -> ParallelInfos:
    global_rank: int = int(os.environ.get("RANK"))
    local_rank: int = int(os.environ.get("LOCAL_RANK"))
    world_size: int = int(os.environ.get("WORLD_SIZE"))
    init_process_group("nccl")
    initialize_model_parallel(
        model_parallel_size_=model_parallel_size or (world_size // pipeline_parallel_size),
        pipeline_length=pipeline_parallel_size
    )

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


def set_pipeline_parallel_barrier():
    """ make sure that all other processes in pipeline parallel group cannot continue until reach this op. """
    torch.distributed.barrier(get_pipeline_parallel_group())
