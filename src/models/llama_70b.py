from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear
)

from src.models.llama import TransformerBlock, Llama, LoraLlama
from src.models.modeling import AttentionForCausalLM
from src.models.modeling_args import LlamaArgs, LoraLlamaArgs
from src.utils import apply_rotary_emb, clamp, apply_lora


class Attention70B(AttentionForCausalLM):
    def __init__(self, args: LlamaArgs):
        super().__init__(args.max_seq_len)
        self.args = args
        self.n_kv_heads = args.n_kv_heads
        self.n_local_heads = args.n_heads // args.world_size
        self.n_local_kv_heads = self.n_kv_heads // args.world_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = None
        self.wk = None
        self.wv = None
        self.wo = None

    def init_weights(self):
        self.wq = ColumnParallelLinear(
            self.args.dim,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            self.args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.wv = ColumnParallelLinear(
            self.args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            self.args.n_heads * self.head_dim,
            self.args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = clamp(self.wq(x)), clamp(self.wk(x)), clamp(self.wv(x))

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)

        xk = self.repeat_kv(xk)
        xv = self.repeat_kv(xv)

        output = self.apply_attention(xq, xk, xv, mask)
        return clamp(self.wo(output))

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        bs, seqlen, n_kv_heads, head_dim = x.shape
        if self.n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, seqlen, n_kv_heads, self.n_rep, head_dim)
            .reshape(bs, seqlen, n_kv_heads * self.n_rep, head_dim)
        )


class FeedForward70B(nn.Module):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        hidden_dim = int(2 * (4 * args.dim) / 3)
        hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.hidden_dim = hidden_dim
        self.dim = args.dim
        self.w1 = None
        self.w2 = None
        self.w3 = None

    def init_weights(self):
        self.w1 = ColumnParallelLinear(
            self.dim, self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            self.hidden_dim, self.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            self.dim, self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )

    def forward(self, x):
        return clamp(self.w2(clamp(F.silu(self.w1(x)) * self.w3(x))))


class TransformerBlock70B(TransformerBlock):
    def __init__(self, layer_id: int, args: LlamaArgs):
        super().__init__(layer_id, args)
        self.attention = Attention70B(args)
        self.feed_forward = FeedForward70B(args)


class Llama70B(Llama):
    def __init__(self, args: LlamaArgs):
        super().__init__(args)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock70B(layer_id, args))


class LoraAttention70B(Attention70B):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
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
        ).float()
        self.lora_b_wq = ColumnParallelLinear(
            self.args.r,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()
        self.lora_a_wk = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).float()
        self.lora_b_wk = ColumnParallelLinear(
            self.args.r,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()
        self.lora_a_wv = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).float()
        self.lora_b_wv = ColumnParallelLinear(
            self.args.r,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()
        self.lora_a_wo = RowParallelLinear(
            self.args.n_heads * self.head_dim,
            self.args.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).float()
        self.lora_b_wo = nn.Linear(
            self.args.r,
            self.args.dim,
            bias=False
        ).float()
        init.zeros_(self.lora_b_wo.weight)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, seqlen, _ = x.shape
        xq = clamp(self.wq(x) + apply_lora(x, self.lora_a_wq, self.lora_b_wq))
        xk = clamp(self.wk(x) + apply_lora(x, self.lora_a_wk, self.lora_b_wk))
        xv = clamp(self.wv(x) + apply_lora(x, self.lora_a_wv, self.lora_b_wv))

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)

        xk = self.repeat_kv(xk)
        xv = self.repeat_kv(xv)

        output = self.apply_attention(xq, xk, xv, mask)

        return clamp(self.wo(output) + apply_lora(output, self.lora_a_wo, self.lora_b_wo))


class LoraFeedForward70B(FeedForward70B):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.r = args.r

        self.lora_a_w1 = None
        self.lora_b_w1 = None
        self.lora_a_w2 = None
        self.lora_b_w2 = None
        self.lora_a_w3 = None
        self.lora_b_w3 = None

    def init_weights(self):
        super().init_weights()

        self.lora_a_w1 = nn.Linear(
            self.dim,
            self.r,
            bias=False
        ).float()
        self.lora_b_w1 = ColumnParallelLinear(
            self.r,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()
        self.lora_a_w2 = RowParallelLinear(
            self.hidden_dim,
            self.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).float()
        self.lora_b_w2 = nn.Linear(
            self.r,
            self.dim,
            bias=False
        ).float()
        init.zeros_(self.lora_b_w2.weight)
        self.lora_a_w3 = nn.Linear(
            self.dim,
            self.r,
            bias=False
        ).float()
        self.lora_b_w3 = ColumnParallelLinear(
            self.r,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()

    def forward(self, x):
        w1_x = self.w1(x) + apply_lora(x, self.lora_a_w1, self.lora_b_w1)
        w3_x = self.w3(x) + apply_lora(x, self.lora_a_w3, self.lora_b_w3)
        out = clamp(F.silu(w1_x) * w3_x)
        return clamp(self.w2(out) + apply_lora(out, self.lora_a_w2, self.lora_b_w2))


class LoraTransformerBlock70B(TransformerBlock70B):
    def __init__(self, layer_id: int, args: LoraLlamaArgs):
        super().__init__(layer_id, args)
        self.attention = LoraAttention70B(args)
        self.feed_forward = LoraFeedForward70B(args)


class LoraLlama70B(LoraLlama):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(LoraTransformerBlock70B(layer_id, args))
