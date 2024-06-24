from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding
)

from src.models.modeling import AttentionForCausalLM
from src.models.modeling_acts import RotaryEmbedding
from src.models.modeling_args import BaichuanArgs


class BaichuanAttention(AttentionForCausalLM):
    def __init__(self, args: BaichuanArgs):
        super().__init__(args.max_seq_len)
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        assert self.head_dim * self.num_heads == self.hidden_size

        self.W_pack = None  # TODO split W_pack into q_proj, k_proj, v_proj
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None
        self.rotary_emb = None

    def init_weights(self):
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.args.max_position_embeddings,
            base=10000
        )

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, seqlen, _ = x.size()
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        pass
