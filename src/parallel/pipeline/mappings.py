import torch
import torch.distributed
from fairscale.nn.model_parallel.initialize import get_pipeline_parallel_group, get_pipeline_parallel_ranks

from src.parallel.utils import get_pipeline_parallel_rank


def _send_forward(input_: torch.Tensor) -> torch.Tensor:
    """ Send the input tensor  """
    group = get_pipeline_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Bypass the function if we are at the last pipe.
    world_size = torch.distributed.get_world_size(group=group)
    if get_pipeline_parallel_rank() == world_size - 1:
        return input_

    # Get the rank of the next pipe.
    dst = get_pipeline_parallel_ranks()[get_pipeline_parallel_rank() + 1]
    # Send.
    torch.distributed.send(input_, dst=dst, group=group)

    return input_


def _send_backward(input_: torch.Tensor) -> torch.Tensor:
    """ Send the input tensor  """
    group = get_pipeline_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Bypass the function if we are at the first pipe.
    if get_pipeline_parallel_rank() == 0:
        return input_

    # Get the rank of the previous pipe.
    dst = get_pipeline_parallel_ranks()[get_pipeline_parallel_rank() - 1]
    # Send.
    torch.distributed.send(input_, dst=dst, group=group)

    return input_


def _recv_forward(input_: torch.Tensor) -> torch.Tensor:
    group = get_pipeline_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Bypass the function if we are at the first pipe.
    if get_pipeline_parallel_rank() == 0:
        return input_

    # Get the rank of the previous pipe.
    src = get_pipeline_parallel_ranks()[get_pipeline_parallel_rank() - 1]
    output = torch.empty_like(input_)
    # Receive.
    torch.distributed.recv(output, src=src, group=group)

    return output


def _recv_backward(input_: torch.Tensor) -> torch.Tensor:
    group = get_pipeline_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Bypass the function if we are at the last pipe.
    world_size = torch.distributed.get_world_size(group=group)
    if get_pipeline_parallel_rank() == world_size - 1:
        return input_

    # Get the rank of the next pipe.
    src = get_pipeline_parallel_ranks()[get_pipeline_parallel_rank() + 1]
    output = torch.empty_like(input_)
    # Receive.
    torch.distributed.recv(output, src=src, group=group)
    return output


def _broadcast_forward(input_: torch.Tensor) -> torch.Tensor:
    group = get_pipeline_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Get the rank of the last pipe.
    src = get_pipeline_parallel_ranks()[-1]
    # Broadcast
    output = input_ if get_pipeline_parallel_rank() == src else torch.empty_like(input_)
    torch.distributed.broadcast(output, src=src, group=group)

    return output


def _broadcast_backward(input_: torch.Tensor) -> torch.Tensor:
    group = get_pipeline_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Get the rank of the first pipe.
    src = get_pipeline_parallel_ranks()[0]
    # Broadcast
    output = input_ if get_pipeline_parallel_rank() == src else torch.empty_like(input_)
    torch.distributed.broadcast(output, src=src, group=group)
    return output


class _SendToPipelineParallelRegion(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _send_forward(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _recv_backward(grad_output)


class _RecvFromPipelineParallelRegion(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        return _recv_forward(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _send_backward(grad_output)


class _BroadcastToPipelineParallelRegion(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        return _broadcast_forward(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ExcludeFromPipelineParallelRegion(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _broadcast_backward(grad_output)


def send_to_pipeline_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _SendToPipelineParallelRegion.apply(input_)


def recv_from_pipeline_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _RecvFromPipelineParallelRegion.apply(input_)


def broadcast_to_pipeline_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _BroadcastToPipelineParallelRegion.apply(input_)


def exclude_from_pipeline_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ExcludeFromPipelineParallelRegion.apply(input_)
