import fairscale.nn.model_parallel.initialize as fs_init
import torch.nn as nn
import torch.nn.init as init
from fairscale.nn.model_parallel.layers import ParallelEmbedding, ColumnParallelLinear

from src.modeling_30b import (
    get_n_local_heads,
    ColumnParallelLinear30B,
    get_partition_size,
    RowParallelLinear30B
)
from src.modeling_abstract import (
    AbstractLoraAttention,
    AbstractLoraTransformerBlock,
    AbstractLoraLlama,
    RMSNorm
)
from src.modeling_args import LoraModelArgs
from src.modeling_lora import LoraFeedForward


class LoraAttention30B(AbstractLoraAttention):
    def __init__(self, args: LoraModelArgs, layer_id: int):
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

        self.lora_a_wq = nn.Linear(
            args.dim,
            args.r,
            bias=False
        ).float()
        self.lora_b_wq = ColumnParallelLinear30B(
            args.r,
            get_partition_size(local_rank, layer_id),
            bias=False,
            gather_output=False,
            init_method=init.zeros_
        ).float()
        self.lora_a_wk = nn.Linear(
            args.dim,
            args.r,
            bias=False
        ).float()
        self.lora_b_wk = ColumnParallelLinear30B(
            args.r,
            get_partition_size(local_rank, layer_id),
            bias=False,
            gather_output=False,
            init_method=init.zeros_
        ).float()
        self.lora_a_wv = nn.Linear(
            args.dim,
            args.r,
            bias=False
        ).float()
        self.lora_b_wv = ColumnParallelLinear30B(
            args.r,
            get_partition_size(local_rank, layer_id),
            bias=False,
            gather_output=False,
            init_method=init.zeros_
        ).float()
        self.lora_a_wo = RowParallelLinear30B(
            get_partition_size(local_rank, layer_id),
            args.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_
        ).float()
        self.lora_b_wo = nn.Linear(
            args.r,
            args.dim,
            bias=False
        ).float()
        init.zeros_(self.lora_b_wo.weight)


class LoraTransformerBlock30B(AbstractLoraTransformerBlock):
    def __init__(self, layer_id: int, args: LoraModelArgs):
        super().__init__(layer_id, args)
        self.attention = LoraAttention30B(args, layer_id)
        self.feed_forward = LoraFeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)


class LoraLlama30B(AbstractLoraLlama):
    def __init__(self, args: LoraModelArgs):
        super().__init__(args)
        for layer_id in range(args.n_layers):
            self.layers.append(LoraTransformerBlock30B(layer_id, args))

        self.tok_embeddings = ParallelEmbedding(
            args.vocab_size, args.dim, init_method=lambda x: x
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(
            args.dim, args.vocab_size, bias=False, init_method=lambda x: x
        )

        self.lora_a_output = nn.Linear(
            args.dim,
            args.r,
            bias=False
        ).float()
        self.lora_b_output = ColumnParallelLinear(
            args.r,
            args.vocab_size,
            bias=False,
            gather_output=True,
            init_method=init.zeros_
        ).float()

        # Freeze parameters
        self._freeze()
