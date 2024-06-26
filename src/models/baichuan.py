from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding
)

from src.checkpoint_baichuan import auto_split_huggingface_checkpoints
from src.models.modeling import AttentionForCausalLM, ParallelModelForCausalLM, CausalLMOutputs, ParallelVerifier, \
    VerifierOutputs
from src.models.modeling_acts import RotaryEmbedding, Clamp, RMSNorm
from src.models.modeling_args import BaichuanArgs
from src.utils import set_barrier, compute_position_ids, apply_rotary_pos_emb, logits_normalize


class BaichuanAttention(AttentionForCausalLM):
    def __init__(self, args: BaichuanArgs):
        super().__init__(args.max_seq_len)
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        assert args.num_attention_heads % args.world_size == 0
        self.num_local_heads = args.num_attention_heads // args.world_size
        self.head_dim = args.hidden_size // args.num_attention_heads

        assert self.head_dim * self.num_heads == self.hidden_size

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
        bsz, seq_len, _ = x.size()
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.num_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_local_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_local_heads, self.head_dim)

        cos, sin = self.rotary_emb.forward(xv.transpose(1, 2), seq_len=seq_len + start_pos)
        position_ids = compute_position_ids(start_pos, seq_len).to(x.device)
        xq, xk = apply_rotary_pos_emb(xq.transpose(1, 2), xk.transpose(1, 2), cos, sin, position_ids)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)

        output = self.apply_attention(xq, xk, xv, mask)
        return self.o_proj(output)


class BaichuanFeedForward(nn.Module):
    def __init__(self, args: BaichuanArgs):
        super().__init__()
        self.args = args

        self.gate_proj = None
        self.down_proj = None
        self.up_proj = None

    def init_weights(self):
        self.gate_proj = ColumnParallelLinear(
            self.args.hidden_size, self.args.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.down_proj = RowParallelLinear(
            self.args.intermediate_size, self.args.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.up_proj = ColumnParallelLinear(
            self.args.hidden_size, self.args.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class BaichuanTransformerBlock(nn.Module):
    def __init__(self, args: BaichuanArgs):
        super().__init__()
        self.args = args
        self.self_attn = BaichuanAttention(args)
        self.mlp = BaichuanFeedForward(args)
        self.clamp = Clamp(disable=not args.use_clamp)

        self.input_layernorm = None
        self.post_attention_layernorm = None

    def init_weights(self):
        self.self_attn.init_weights()
        self.mlp.init_weights()
        self.input_layernorm = RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps).type(self.args.dtype)
        self.post_attention_layernorm = RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps).type(self.args.dtype)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache
    ):
        h = x + self.self_attn.forward(self.input_layernorm(x), start_pos, mask, use_cache)
        h = self.clamp.forward(h)
        out = h + self.mlp.forward(self.post_attention_layernorm(h))
        out = self.clamp.forward(out)
        return out


class BaichuanHead(nn.Module):
    def __init__(self, args: BaichuanArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = None
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.num_hidden_layers):
            self.layers.append(BaichuanTransformerBlock(args))
        self.norm = None

    def init_weights(self):
        self.embed_tokens = ParallelEmbedding(
            self.args.vocab_size, self.args.hidden_size, init_method=lambda x: x
        ).type(self.args.dtype)
        for layer in self.layers:
            layer.init_weights()
        self.norm = RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps).type(self.args.dtype)

    def forward(self, tokens: torch.Tensor, start_pos=0, use_cache=False):
        tokens = tokens.to(next(self.parameters()).device)
        _bsz, seq_len = tokens.shape
        h = self.embed_tokens(tokens)

        mask = None
        if seq_len > 1:
            mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, mask, use_cache)
        return self.norm(h)


class Baichuan(ParallelModelForCausalLM):
    def __init__(self, args: BaichuanArgs):
        super().__init__(args.local_rank, args.world_size)
        self.args = args
        self.model = BaichuanHead(args)
        self.lm_head = None

    def init_weights(self):
        self.model.init_weights()
        self.lm_head = nn.Linear(  # TODO: For added tokens
            self.args.hidden_size, self.args.vocab_size, bias=False
        ).type(self.args.dtype)
        # self.lm_head = ColumnParallelLinear(
        #     self.args.hidden_size, self.args.vocab_size, bias=False, init_method=lambda x: x
        # ).type(self.args.dtype)

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ) -> CausalLMOutputs:
        h = self.model.forward(tokens, start_pos, use_cache)
        output = self.lm_head(h)
        return CausalLMOutputs(logits=logits_normalize(output), hidden_states=h)

    # Copied from llama_hf.LlamaHf.load
    def load(self, ckpt_dir: str, verbose: bool = True):
        checkpoints = sorted(Path(ckpt_dir).glob("consolidated.*.pth"))
        if len(checkpoints) == 0:  # splitting
            ckpt_dir = auto_split_huggingface_checkpoints(
                ckpt_dir, world_size=self.world_size, local_rank=self.local_rank, verbose=verbose
            )
            set_barrier()
        super().load(ckpt_dir, verbose=verbose, merge_lora=True)

    # Copied from llama_hf.LlamaHf.flush
    def flush(self):
        for i in range(self.args.num_hidden_layers):
            self.model.layers[i].self_attn.flush()
        set_barrier()


class BaichuanVerifier(ParallelVerifier):
    def __init__(self, args: BaichuanArgs):
        super().__init__(args.local_rank, args.world_size)
        self.args = args
        self.model = BaichuanHead(args)
        self.v_head = None

    def init_weights(self):
        self.model.init_weights()
        self.v_head = nn.Linear(
            self.args.hidden_size, 1, bias=False
        ).type(self.args.dtype)

    def forward(self, tokens: torch.Tensor) -> VerifierOutputs:
        h = self.model.forward(tokens)
        scores = self.v_head(h.type_as(self.v_head.weight)).squeeze(-1)  # [b, s]
        return VerifierOutputs(scores=scores)

    def load(self, ckpt_dir: str, verbose: bool = True):
        checkpoints = sorted(Path(ckpt_dir).glob("consolidated.*.pth"))
        if len(checkpoints) == 0:  # splitting
            ckpt_dir = auto_split_huggingface_checkpoints(
                ckpt_dir, world_size=self.world_size, local_rank=self.local_rank, verbose=verbose
            )
            set_barrier()
        super().load(ckpt_dir, verbose=verbose, merge_lora=True)

