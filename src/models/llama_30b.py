import torch.nn as nn
import torch.nn.init as init

from src.models.llama import (
    Attention,
    TransformerBlock,
    Llama,
    LoraFeedForward,
    LoraLlama
)
from src.models.modeling_args import LoraLlamaArgs
from src.models.modeling_acts import ColumnParallelLinearPartitioned, RowParallelLinearPartitioned


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


class Attention30B(Attention):
    def __init__(self, args: LoraLlamaArgs, layer_id: int):
        super().__init__(args)
        self.layer_id = layer_id
        self.n_local_heads = get_n_local_heads(self.args.local_rank, layer_id)

    def init_weights(self):
        self.wq = ColumnParallelLinearPartitioned(
            self.args.dim,
            get_partition_size(self.args.local_rank, self.layer_id),
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinearPartitioned(
            self.args.dim,
            get_partition_size(self.args.local_rank, self.layer_id),
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinearPartitioned(
            self.args.dim,
            get_partition_size(self.args.local_rank, self.layer_id),
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinearPartitioned(
            get_partition_size(self.args.local_rank, self.layer_id),
            self.args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )


class TransformerBlock30B(TransformerBlock):
    def __init__(self, layer_id: int, args: LoraLlamaArgs):
        super().__init__(layer_id, args)
        self.attention = Attention30B(args, layer_id)


class Llama30B(Llama):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock30B(layer_id, args))


class LoraAttention30B(Attention30B):
    def __init__(self, args: LoraLlamaArgs, layer_id: int):
        super().__init__(args, layer_id)
        self.args = args

        self.lora_a_wq = None
        self.lora_b_wq = None
        self.lora_a_wk = None
        self.lora_b_wk = None
        self.lora_a_wv = None
        self.lora_b_wv = None
        self.lora_a_wo = None
        self.lora_b_wo = None

    def init_weights(self):
        super().init_weights()

        self.lora_a_wq = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_wq = ColumnParallelLinearPartitioned(
            self.args.r,
            get_partition_size(self.args.local_rank, self.layer_id),
            bias=False,
            gather_output=False,
            init_method=init.zeros_
        ).type(self.args.lora_dtype)
        self.lora_a_wk = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_wk = ColumnParallelLinearPartitioned(
            self.args.r,
            get_partition_size(self.args.local_rank, self.layer_id),
            bias=False,
            gather_output=False,
            init_method=init.zeros_
        ).type(self.args.lora_dtype)
        self.lora_a_wv = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_wv = ColumnParallelLinearPartitioned(
            self.args.r,
            get_partition_size(self.args.local_rank, self.layer_id),
            bias=False,
            gather_output=False,
            init_method=init.zeros_
        ).type(self.args.lora_dtype)
        self.lora_a_wo = RowParallelLinearPartitioned(
            get_partition_size(self.args.local_rank, self.layer_id),
            self.args.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_
        ).type(self.args.lora_dtype)
        self.lora_b_wo = nn.Linear(
            self.args.r,
            self.args.dim,
            bias=False
        ).type(self.args.lora_dtype)
        init.zeros_(self.lora_b_wo.weight)


class LoraTransformerBlock30B(TransformerBlock30B):
    def __init__(self, layer_id: int, args: LoraLlamaArgs):
        super().__init__(layer_id, args)
        self.attention = LoraAttention30B(args, layer_id)
        self.feed_forward = LoraFeedForward(args)


class LoraLlama30B(LoraLlama):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.args = args
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(LoraTransformerBlock30B(layer_id, args))
