import torch

from src.parallel.initialize import get_data_parallel_world_size, get_data_parallel_group, get_data_parallel_rank
from src.parallel.utils import divide_and_check_no_remainder, split_tensor_along_first_dim


def gather_tensor_from_data_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """ Gather tensor and concat along the first dimension. """
    group = get_data_parallel_group()

    if get_data_parallel_world_size() == 1:
        return input_

    rank = get_data_parallel_rank()
    world_size = get_data_parallel_world_size()

    if input_.dim() == 0:  # Handle scalar
        input_ = input_.unsqueeze(dim=0)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)
    output = torch.cat(tensor_list, dim=0).contiguous()

    return output


def scatter_tensor_to_data_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """ Split the tensor along its first dimension and keep the corresponding slice """
    if get_data_parallel_world_size() == 1:
        return input_

    world_size = get_data_parallel_world_size()
    input_list = split_tensor_along_first_dim(input_, world_size)

    rank = get_data_parallel_rank()
    output = input_list[rank].contiguous()

    return output


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