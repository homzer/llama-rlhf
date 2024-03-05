import os
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

from src.checkpoint import splitting
from src.models.llama_hf import compute_position_ids, apply_rotary_pos_emb
from src.models.mistral import Attention, repeat_kv, FeedForward, TransformerBlock
from src.models.modeling import ParallelModelForCausalLM, CausalLMOutputs
from src.models.modeling_acts import RMSNorm
from src.models.modeling_args import MistralArgs
from src.utils import set_barrier


class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class AttentionHF(Attention):
    def __init__(self, args: MistralArgs):
        super().__init__(args)

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None

        self.rotary_emb = None

    def init_weights(self):
        self.q_proj = ColumnParallelLinear(
            self.args.dim,
            self.args.n_heads * self.args.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.k_proj = ColumnParallelLinear(
            self.args.dim,
            self.args.n_kv_heads * self.args.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.v_proj = ColumnParallelLinear(
            self.args.dim,
            self.args.n_kv_heads * self.args.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.o_proj = RowParallelLinear(
            self.args.n_heads * self.args.head_dim,
            self.args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        self.rotary_emb = MistralRotaryEmbedding(
            self.args.head_dim,
            max_position_embeddings=8192,  # TODO
            base=10000
        )

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: torch.Tensor = None,
            use_cache: bool = False
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.args.head_dim)

        cos, sin = self.rotary_emb.forward(xv.transpose(1, 2), seq_len=seqlen + start_pos)
        position_ids = compute_position_ids(start_pos, seqlen).to(x.device)
        xq, xk = apply_rotary_pos_emb(xq.transpose(1, 2), xk.transpose(1, 2), cos, sin, position_ids)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)
        else:
            xk, xv = repeat_kv(xk, xv, self.repeats)

        output = self.apply_attention(xq, xk, xv, mask[None, None, ...] if mask is not None else None)

        return self.o_proj(output)


class FeedForwardHF(FeedForward):
    def __init__(self, args: MistralArgs):
        super().__init__(args)

        self.gate_proj = None
        self.down_proj = None
        self.up_proj = None

    def init_weights(self):
        self.gate_proj = ColumnParallelLinear(
            self.args.dim,
            self.args.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.down_proj = RowParallelLinear(
            self.args.hidden_dim,
            self.args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        )
        self.up_proj = ColumnParallelLinear(
            self.args.dim,
            self.args.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )

    def forward(self, x) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlockHF(TransformerBlock):
    def __init__(self, args: MistralArgs):
        super().__init__(args)
        self.self_attn = AttentionHF(args)
        self.mlp = FeedForwardHF(args)

        self.input_layernorm = None
        self.post_attention_layernorm = None

    def init_weights(self):
        self.self_attn.init_weights()
        self.mlp.init_weights()
        self.input_layernorm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        self.post_attention_layernorm = RMSNorm(self.args.dim, eps=self.args.norm_eps)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            use_cache: bool
    ) -> torch.Tensor:
        h = x + self.self_attn.forward(self.input_layernorm(x), start_pos, freqs_cis, mask, use_cache)
        h = self.clamp.forward(h)
        out = h + self.mlp.forward(self.post_attention_layernorm(h))
        out = self.clamp.forward(out)
        return out


class MistralHeadHF(nn.Module):
    def __init__(self, args: MistralArgs):
        super().__init__()
        self.args = args

        self.embed_tokens = None
        self.layers = torch.nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlockHF(args))
        self.norm = None

    def init_weights(self):
        self.embed_tokens = ParallelEmbedding(
            self.args.vocab_size, self.args.dim, init_method=lambda x: x
        )
        for layer in self.layers:
            layer.init_weights()
        self.norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int,
            use_cache: bool
    ):
        tokens = tokens.to(next(self.parameters()).device)
        _bsz, seq_len = tokens.shape
        h = self.embed_tokens(tokens)

        mask = None
        if seq_len > 1:
            mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, None, mask, use_cache)
        return self.norm(h)


class MistralHF(ParallelModelForCausalLM):
    def __init__(self, args: MistralArgs):
        super().__init__(args.local_rank, args.world_size)
        self.args = args
        self.model = MistralHeadHF(args)
        self.lm_head = None

    def init_weights(self):
        self.model.init_weights()
        self.lm_head = ColumnParallelLinear(
            self.args.dim, self.args.vocab_size, bias=False, init_method=lambda x: x
        )

    def forward(self, tokens: torch.Tensor, start_pos=0, use_cache=False):
        h = self.model.forward(tokens, start_pos, use_cache)
        output = self.lm_head(h)
        return CausalLMOutputs(logits=output, hidden_states=h)

    def load(self, ckpt_dir: str, verbose: bool = True, **kwargs):
        checkpoints = sorted(Path(ckpt_dir).glob("consolidated.*.pth"))
        if len(checkpoints) != 0:  # normal loading
            super().load(ckpt_dir, verbose, **kwargs)
        else:  # splitting
            pl_ckpt_dir = os.path.join(ckpt_dir, str(self.world_size))
            if self.local_rank == 0 and not os.path.exists(pl_ckpt_dir):
                if verbose:
                    print(f'Parallel checkpoint dose not exist. Splitting into {pl_ckpt_dir} ...')
                if os.path.exists(os.path.join(ckpt_dir, "pytorch_model.bin")):
                    split_file = os.path.join(ckpt_dir, "pytorch_model.bin")
                else:
                    split_file = sorted(Path(ckpt_dir).glob("*.safetensors"))
                    if len(split_file) == 0:
                        raise FileNotFoundError("Can not find any checkpoint file")
                if 'num_added_tokens' in kwargs:
                    splitting(split_file, pl_ckpt_dir, n=self.world_size, num_added_tokens=kwargs['num_added_tokens'])
                else:
                    splitting(split_file, pl_ckpt_dir, n=self.world_size)
                if verbose:
                    print('Done!')
            set_barrier()
            super().load(pl_ckpt_dir, verbose, **kwargs)

    def flush(self):
        """ Clean cache in `Attention` module """
        for i in range(self.args.n_layers):
            self.model.layers[i].self_attn.flush()
        set_barrier()
