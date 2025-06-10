# Parts of the code here are adapted from FairScale
import collections
import datetime
import os
import sys
from typing import Optional

import torch

from src.logger import Logger
from src.parallel.utils import ensure_divisibility
from src.utils import set_seed

_MODEL_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# Sequence parallel group that the current rank belongs to.
_SEQUENCE_PARALLEL_GROUP = None


def initialize_model_parallel(
        model_parallel_size_: int,
        sequence_parallel_size: int = 1,
        *,
        model_parallel_backend: Optional[str] = None,
        sequence_parallel_backend: Optional[str] = None,
        ddp_backend: Optional[str] = None
) -> None:
    """
    Initialize model data parallel groups.

    Arguments:
        model_parallel_size_: number of GPUs used to parallelize model.
        sequence_parallel_size: number of GPUs used to parallelize sequence.
        model_parallel_backend: backend used to parallelize model.
        sequence_parallel_backend: backend used to parallelize pipeline.
        ddp_backend: backend used to parallelize distributed data parallel.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7, and we
    use 2 GPUs to parallelize the model. The present function will
    create 4 model parallel groups and 2 data parallel groups as:
        4 model parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 data parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    model_parallel_size = int(min(model_parallel_size_, world_size))
    ensure_divisibility(world_size, model_parallel_size)
    ensure_divisibility(world_size, model_parallel_size * sequence_parallel_size)
    rank = torch.distributed.get_rank()

    data_parallel_size = int(world_size / (model_parallel_size * sequence_parallel_size))

    if torch.distributed.get_rank() == 0:
        print("> initializing model parallel with size {}".format(model_parallel_size_))
        print("> initializing ddp with size {}".format(data_parallel_size))
        print("> initializing sequence parallel with size {}".format(sequence_parallel_size))

    groups = torch.LongTensor(range(world_size)).reshape(
        data_parallel_size, sequence_parallel_size, model_parallel_size
    )

    found = torch.where(groups == rank)
    assert all(len(x) == 1 for x in found)
    found = [x[0] for x in found]

    # Build the data parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    for j in range(sequence_parallel_size):
        for k in range(model_parallel_size):
            group = torch.distributed.new_group(groups[:, j, k].tolist(), backend=ddp_backend)
            if j == found[1] and k == found[2]:
                _DATA_PARALLEL_GROUP = group

    # Build the model parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, "model parallel group is already initialized"
    for i in range(data_parallel_size):
        for j in range(sequence_parallel_size):
            group = torch.distributed.new_group(groups[i, j, :].tolist(), backend=model_parallel_backend)
            if i == found[0] and j == found[1]:
                _MODEL_PARALLEL_GROUP = group

    # Build the sequence parallel groups.
    global _SEQUENCE_PARALLEL_GROUP
    assert _SEQUENCE_PARALLEL_GROUP is None, "sequence parallel group is already initialized"
    for i in range(data_parallel_size):
        for k in range(model_parallel_size):
            group = torch.distributed.new_group(groups[i, :, k].tolist(), backend=sequence_parallel_backend)
            if i == found[0] and k == found[2]:
                _SEQUENCE_PARALLEL_GROUP = group


def model_parallel_is_initialized() -> bool:
    """Check if model and data parallel groups are initialized."""
    if _MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None or _SEQUENCE_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_data_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_sequence_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the sequence parallel group the caller rank belongs to."""
    assert _SEQUENCE_PARALLEL_GROUP is not None, "sequence parallel group is not initialized"
    return _SEQUENCE_PARALLEL_GROUP


ParallelInfos = collections.namedtuple("ParallelInfos", [
    "global_rank",
    "local_rank",
    "world_size",
    "model_parallel_world_size",
    "model_parallel_rank",
    "model_parallel_src_rank",
    "data_parallel_world_size",
    "data_parallel_rank",
    "data_parallel_src_rank",
    "sequence_parallel_world_size",
    "sequence_parallel_rank",
    "sequence_parallel_src_rank"
])


def setup_model_parallel(
        model_parallel_size: int = None, sequence_parallel_size: int = 1, seed: int = None, log_dir: str = None
) -> ParallelInfos:
    sequence_parallel_size = sequence_parallel_size or 1
    global_rank: int = int(os.environ.get("RANK"))
    local_rank: int = int(os.environ.get("LOCAL_RANK"))
    world_size: int = int(os.environ.get("WORLD_SIZE"))
    torch.distributed.init_process_group("nccl", timeout=datetime.timedelta(minutes=360))
    initialize_model_parallel(
        model_parallel_size_=model_parallel_size or (world_size // sequence_parallel_size),
        sequence_parallel_size=sequence_parallel_size
    )

    model_parallel_world_size: int = get_model_parallel_world_size()
    model_parallel_rank: int = get_model_parallel_rank()
    model_parallel_src_rank: int = get_model_parallel_src_rank()
    data_parallel_world_size: int = get_data_parallel_world_size()
    data_parallel_rank: int = get_data_parallel_rank()
    data_parallel_src_rank: int = get_data_parallel_src_rank()
    sequence_parallel_world_size: int = get_sequence_parallel_world_size()
    sequence_parallel_rank: int = get_sequence_parallel_rank()
    sequence_parallel_src_rank: int = get_sequence_parallel_src_rank()

    if data_parallel_src_rank != 0:
        sys.stdout = open(os.devnull, "w")
    if global_rank == 0 and log_dir is not None:
        sys.stdout = Logger(log_dir=log_dir)

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
        data_parallel_src_rank=data_parallel_src_rank,
        sequence_parallel_world_size=sequence_parallel_world_size,
        sequence_parallel_rank=sequence_parallel_rank,
        sequence_parallel_src_rank=sequence_parallel_src_rank
    )


def get_rank() -> int:
    """Return my global rank."""
    return int(os.environ.get("RANK"))


def get_local_rank() -> int:
    """Return my local rank."""
    return int(os.environ.get("LOCAL_RANK"))


def get_world_size() -> int:
    """Return the world size of the global group."""
    return int(os.environ.get("WORLD_SIZE"))


def get_data_parallel_rank() -> int:
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def get_model_parallel_src_rank() -> int:
    """Calculate the global rank corresponding to a local rank zero
    in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    mp_size = get_model_parallel_world_size()
    return (global_rank // mp_size) * mp_size


def get_data_parallel_world_size() -> int:
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_model_parallel_world_size() -> int:
    """Return world size for the model parallel group."""
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def get_model_parallel_rank() -> int:
    """Return my rank for the model parallel group."""
    return torch.distributed.get_rank(group=get_model_parallel_group())


def get_data_parallel_src_rank() -> int:
    """Calculate the global rank corresponding to a local rank zero
    in the data parallel group."""
    global_rank = torch.distributed.get_rank()
    mp_size = get_model_parallel_world_size()
    sp_size = get_sequence_parallel_world_size()
    return global_rank % (mp_size * sp_size)


def get_sequence_parallel_rank() -> int:
    """Return my rank for the sequence parallel group."""
    return torch.distributed.get_rank(group=get_sequence_parallel_group())


def get_sequence_parallel_world_size() -> int:
    """Return world size for the sequence parallel group."""
    return torch.distributed.get_world_size(group=get_sequence_parallel_group())


def get_sequence_parallel_src_rank() -> int:
    """Calculate the global rank corresponding to a local rank zero
    in the sequence parallel group."""
    global_rank = torch.distributed.get_rank()
    mp_size = get_model_parallel_world_size()
    sp_size = get_sequence_parallel_world_size()
    return global_rank % mp_size + (mp_size * sp_size) * (global_rank // (mp_size * sp_size))


def set_barrier():
    """ make sure that all other processes cannot continue until reach this op. """
    torch.distributed.barrier()


def set_model_parallel_barrier():
    """ make sure that all other processes in model parallel group cannot continue until reach this op. """
    torch.distributed.barrier(get_model_parallel_group())


def set_data_parallel_barrier():
    """ make sure that all other processes in data parallel group cannot continue until reach this op. """
    torch.distributed.barrier(get_data_parallel_group())


def set_sequence_parallel_barrier():
    """ make sure that all other processes in sequence parallel group cannot continue until reach this op. """
    torch.distributed.barrier(get_sequence_parallel_group())


def destroy_model_parallel() -> None:
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _SEQUENCE_PARALLEL_GROUP
    _SEQUENCE_PARALLEL_GROUP = None
