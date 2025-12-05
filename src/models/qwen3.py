from typing import Optional

import torch

from src.models.modeling import (
    AttentionForCausalLM
)
from src.models.modeling_acts import RMSNorm, RotaryEmbedding
from src.models.modeling_args import QwenArgs
from src.models.qwen import (
    Qwen,
    QwenHead,
    QwenTransformerBlock
)
from src.parallel.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear
)
from src.parallel.sequence_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region
)
from src.utils import compute_position_ids, apply_rotary_pos_emb_


class Qwen3Attention(AttentionForCausalLM):
    def __init__(self, args: QwenArgs):
        super().__init__(args.max_seq_len)
        self.args = args
        # self.head_dim = args.hidden_size // args.num_attention_heads
        self.head_dim = args.head_dim
        assert args.num_attention_heads % args.model_parallel_world_size == 0
        self.num_local_heads = args.num_attention_heads // args.model_parallel_world_size
        self.num_key_value_heads = args.num_key_value_heads
        assert self.num_key_value_heads % args.model_parallel_world_size == 0
        self.num_local_key_value_heads = self.num_key_value_heads // args.model_parallel_world_size
        self.n_rep = args.num_attention_heads // args.num_key_value_heads

        self.rotary_emb = None

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None
        self.q_proj_fn = lambda x: self.q_proj(x)
        self.k_proj_fn = lambda x: self.k_proj(x)
        self.v_proj_fn = lambda x: self.v_proj(x)
        self.o_proj_fn = lambda x: self.o_proj(x)
        self.q_norm = None
        self.k_norm = None

    def init_weights(self):
        self.q_proj = ColumnParallelLinear(
            self.args.hidden_size,
            self.args.num_attention_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            ).type(self.args.dtype)
        self.k_proj = ColumnParallelLinear(
            self.args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.v_proj = ColumnParallelLinear(
            self.args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            ).type(self.args.dtype)
        self.o_proj = RowParallelLinear(
            self.args.num_attention_heads * self.head_dim,
            self.args.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
            ).type(self.args.dtype)
        self.q_norm = RMSNorm(self.head_dim, eps=self.args.rms_norm_eps).type(self.args.dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=self.args.rms_norm_eps).type(self.args.dtype)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.args.max_seq_len,
            base=self.args.rope_theta,
        ).type(self.args.dtype)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, local_seq_len, _ = x.size()
        xq, xk, xv = self.q_proj_fn(x), self.k_proj_fn(x), self.v_proj_fn(x)

        xq = self.q_norm(xq.view(bsz, local_seq_len, self.num_local_heads, self.head_dim))
        xk = self.k_norm(xk.view(bsz, local_seq_len, self.num_local_key_value_heads, self.head_dim))
        xv = xv.view(bsz, local_seq_len, self.num_local_key_value_heads, self.head_dim)

        # Sequence Parallel Op.
        if not use_cache:  # Bypass the function if performing autoregressive generation.
            xk = gather_from_sequence_parallel_region(xk)
            xv = gather_from_sequence_parallel_region(xv)
        seq_len = xv.shape[1]

        position_ids = compute_position_ids(start_pos, seq_len).to(x.device)
        # Sequence Parallel Op.
        local_position_ids = position_ids
        if not use_cache:  # Bypass the function if performing autoregressive generation.
            local_position_ids = scatter_to_sequence_parallel_region(position_ids)

        cos, sin = self.rotary_emb.forward(xv.transpose(1, 2), seq_len=seq_len + start_pos)
        xq = apply_rotary_pos_emb_(xq.transpose(1, 2), cos, sin, local_position_ids)
        xk = apply_rotary_pos_emb_(xk.transpose(1, 2), cos, sin, position_ids)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)

        xk, xv = self.repeat_kv(xk, xv, self.n_rep)

        output = self.apply_attention(xq, xk, xv, mask)
        return self.o_proj_fn(output)


class Qwen3TransformerBlock(QwenTransformerBlock):
    def __init__(self, args: QwenArgs):
        super().__init__(args=args)
        self.self_attn = Qwen3Attention(args)


class Qwen3Head(QwenHead):
    def __init__(self, args: QwenArgs):
        super().__init__(args=args)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.num_hidden_layers):
            self.layers.append(Qwen3TransformerBlock(args))


class Qwen3(Qwen):
    def __init__(self, args: QwenArgs):
        super().__init__(args=args)
        self.model = Qwen3Head(args)
