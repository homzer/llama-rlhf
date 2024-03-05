import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.model_parallel.mappings import (
    scatter_to_model_parallel_region,
    reduce_from_model_parallel_region,
    gather_from_model_parallel_region,
    copy_to_model_parallel_region
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class ColumnParallelLinearPartitioned(nn.Module):
    """ Warning: Partitioned size should be equal across all ranks """
    def __init__(self,
                 in_features: int,
                 output_size_per_partition: int,
                 bias: bool = True,
                 gather_output: bool = True,
                 init_method=nn.init.xavier_normal_):
        super().__init__()
        self.gather_output = gather_output
        self.weight = nn.Parameter(torch.Tensor(output_size_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size_per_partition))
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        init_method(self.weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output


class RowParallelLinearPartitioned(nn.Module):
    """ Warning: Partitioned size should be equal across all ranks """
    def __init__(self,
                 input_size_per_partition: int,
                 out_features: int,
                 bias: bool = True,
                 input_is_parallel: bool = False,
                 init_method=nn.init.xavier_normal_):
        super().__init__()
        self.input_is_parallel = input_is_parallel
        self.weight = nn.Parameter(torch.Tensor(out_features, input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        init_method(self.weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type:ignore
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output


class Clamp:
    def __init__(self, disable: bool = False):
        self.disable = disable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Clamp inf values to enable fp16 training.
        Will slow down speed, disable it when you don't need it.
        """
        if self.disable or not x.requires_grad:  # disable when inference
            return x
        if x.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(x).any(),
                torch.finfo(x.dtype).max - 1000,
                torch.finfo(x.dtype).max
            ).item()
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        return x
