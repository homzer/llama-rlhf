import torch

from src.parallel.initialize import get_sequence_parallel_group
from src.parallel.utils import split_tensor_along_second_dim


def _split(input_: torch.Tensor) -> torch.Tensor:
    """Split the tensor along its second dimension and keep the
    corresponding slice."""
    group = get_sequence_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Split along second dimension.
    world_size = torch.distributed.get_world_size(group=group)
    input_list = split_tensor_along_second_dim(input_, world_size)

    rank = torch.distributed.get_rank(group=group)
    output = input_list[rank].contiguous()

    return output


def _gather(input_: torch.Tensor) -> torch.Tensor:
    """Gather tensors and concatenate along the second dimension."""
    group = get_sequence_parallel_group()

    # Bypass the function if we are using only 1 GPU
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)

    output = torch.cat(tensor_list, dim=1).contiguous()

    return output


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate"""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _split(grad_output)


def scatter_to_sequence_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ScatterToSequenceParallelRegion.apply(input_)


def gather_from_sequence_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _GatherFromSequenceParallelRegion.apply(input_)
