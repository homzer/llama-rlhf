import torch

from src.parallel.initialize import get_data_parallel_world_size, get_data_parallel_group, get_data_parallel_rank
from src.parallel.utils import divide_and_check_no_remainder


def gather_object_from_data_parallel_region(obj: list) -> list:
    """Gather object list and concat along the first dimension."""
    if get_data_parallel_world_size() == 1:  # Bypass the function if we are using only 1 GPU.
        return obj
    object_list = [[] for _ in range(get_data_parallel_world_size())]
    torch.distributed.all_gather_object(object_list, obj=obj, group=get_data_parallel_group())
    output = []
    for obj in object_list:
        output.extend(obj)
    return output


def scatter_object_to_data_parallel_region(obj: list) -> list:
    """Split the object list along its length size and keep the corresponding slice."""
    if get_data_parallel_world_size() == 1:  # Bypass the function if we are using only 1 GPU.
        return obj
    world_size = get_data_parallel_world_size()
    split_size = divide_and_check_no_remainder(len(obj), world_size)
    rank = get_data_parallel_rank()
    return obj[rank * split_size: (rank + 1) * split_size]