import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.model_parallel import (
    gather_from_model_parallel_region,
    copy_to_model_parallel_region
)
from fairscale.nn.model_parallel.layers import ParallelEmbedding, ColumnParallelLinear
from fairscale.nn.model_parallel.mappings import (
    scatter_to_model_parallel_region,
    reduce_from_model_parallel_region
)

from src.modeling.llama import FeedForward
from src.modeling.llama_abstract import (
    AbstractAttention,
    AbstractTransformerBlock,
    AbstractLlama,
)
from src.modeling.modeling_acts import RMSNorm
from src.modeling.modeling_args import LoraLlamaArgs


class ColumnParallelLinear30B(nn.Module):
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


class RowParallelLinear30B(torch.nn.Module):
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


def get_partition_size(local_rank, layer_id):
    if (
            local_rank in [0, 2, 4, 6] and layer_id < 30
    ) or (
            local_rank in [1, 3, 5, 7] and layer_id >= 30
    ):
        return 896
    else:
        return 768


def get_n_local_heads(local_rank, layer_id):
    if (
            local_rank in [0, 2, 4, 6] and layer_id < 30
    ) or (
            local_rank in [1, 3, 5, 7] and layer_id >= 30
    ):
        return 7
    else:
        return 6


class Attention30B(AbstractAttention):
    def __init__(self, args: LoraLlamaArgs, layer_id: int):
        super().__init__(args)
        local_rank = fs_init.get_model_parallel_rank()
        self.n_local_heads = get_n_local_heads(local_rank, layer_id)

        self.wq = ColumnParallelLinear30B(
            args.dim,
            get_partition_size(local_rank, layer_id),
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear30B(
            args.dim,
            get_partition_size(local_rank, layer_id),
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear30B(
            args.dim,
            get_partition_size(local_rank, layer_id),
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear30B(
            get_partition_size(local_rank, layer_id),
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )


class TransformerBlock30B(AbstractTransformerBlock):
    def __init__(self, layer_id: int, args: LoraLlamaArgs):
        super().__init__(layer_id, args)
        self.attention = Attention30B(args, layer_id)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)


class Llama30B(AbstractLlama):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock30B(layer_id, args))

        self.tok_embeddings = ParallelEmbedding(
            args.vocab_size, args.dim, init_method=lambda x: x
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(
            args.dim, args.vocab_size, bias=False, init_method=lambda x: x
        )
