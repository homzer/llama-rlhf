import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding
)

from src.checkpoint import splitting
from src.models.modeling import ParallelModelForCausalLM, CausalLMOutputs, AttentionForCausalLM
from src.models.modeling_acts import RMSNorm, Clamp
from src.models.modeling_args import LlamaArgs, LoraLlamaArgs
from src.utils import set_barrier, apply_lora, logits_normalize


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def compute_position_ids(start_pos: int, seq_length: int):
    position_ids = torch.arange(
        start_pos, seq_length + start_pos, dtype=torch.long
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    return position_ids


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).type(self.args.lora_dtype).to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaAttentionHF(AttentionForCausalLM):
    def __init__(self, args: LlamaArgs):
        super().__init__(args.max_seq_len)
        self.args = args
        self.n_local_heads = args.n_heads // args.world_size
        self.head_dim = args.dim // args.n_heads

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None

        self.rotary_emb = None

    def init_weights(self):
        self.q_proj = ColumnParallelLinear(
            self.args.dim,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.k_proj = ColumnParallelLinear(
            self.args.dim,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.v_proj = ColumnParallelLinear(
            self.args.dim,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.o_proj = RowParallelLinear(
            self.args.n_heads * self.head_dim,
            self.args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_heads, self.head_dim)

        cos, sin = self.rotary_emb.forward(xv.transpose(1, 2), seq_len=seq_len + start_pos)
        position_ids = compute_position_ids(start_pos, seq_len).to(x.device)
        xq, xk = apply_rotary_pos_emb(xq.transpose(1, 2), xk.transpose(1, 2), cos, sin, position_ids)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)

        output = self.apply_attention(xq, xk, xv, mask)

        return self.o_proj(output)


class LlamaFeedForwardHF(nn.Module):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        hidden_dim = int(2 * (4 * args.dim) / 3)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.hidden_dim = hidden_dim
        self.dim = args.dim
        self.gate_proj = None
        self.down_proj = None
        self.up_proj = None

    def init_weights(self):
        self.gate_proj = ColumnParallelLinear(
            self.dim, self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.down_proj = RowParallelLinear(
            self.hidden_dim, self.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        )
        self.up_proj = ColumnParallelLinear(
            self.dim, self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaTransformerBlockHF(nn.Module):
    def __init__(self, layer_id: int, args: LlamaArgs):
        super().__init__()
        self.args = args
        self.self_attn = LlamaAttentionHF(args)
        self.mlp = LlamaFeedForwardHF(args)
        self.layer_id = layer_id
        self.clamp = Clamp(disable=not args.use_clamp)

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
            mask: Optional[torch.Tensor],
            use_cache
    ):
        h = x + self.self_attn.forward(self.input_layernorm(x), start_pos, mask, use_cache)
        h = self.clamp.forward(h)
        out = h + self.mlp.forward(self.post_attention_layernorm(h))
        out = self.clamp.forward(out)
        return out


class LlamaHeadHF(nn.Module):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        self.args = args

        self.embed_tokens = None
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(LlamaTransformerBlockHF(layer_id, args))
        self.norm = None

    def init_weights(self):
        self.embed_tokens = ParallelEmbedding(
            self.args.vocab_size, self.args.dim, init_method=lambda x: x
        )
        for layer in self.layers:
            layer.init_weights()
        self.norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)

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


class LlamaHF(ParallelModelForCausalLM):
    def __init__(self, args: LlamaArgs):
        super().__init__(args.local_rank, args.world_size)
        self.args = args
        self.model = LlamaHeadHF(args)
        self.lm_head = None

    def init_weights(self):
        self.model.init_weights()
        self.lm_head = ColumnParallelLinear(
            self.args.dim, self.args.vocab_size, bias=False, init_method=lambda x: x
        )

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos=0,
            use_cache=False
    ):
        h = self.model.forward(tokens, start_pos, use_cache)
        output = self.lm_head(h)
        return CausalLMOutputs(logits=logits_normalize(output), hidden_states=h)

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
                        split_file = sorted(Path(ckpt_dir).glob("pytorch_model*.bin"))
                        if len(split_file) == 0:
                            raise FileNotFoundError("Can not find any checkpoint file")
                splitting(split_file, pl_ckpt_dir, n=self.world_size)
                if verbose:
                    print('Done!')
            set_barrier()
            super().load(pl_ckpt_dir, verbose, **kwargs)

    def flush(self):
        """ Clean cache in `LlamaAttention` module """
        for i in range(self.args.n_layers):
            self.model.layers[i].self_attn.flush()
        set_barrier()


class LoraLlamaAttentionHF(LlamaAttentionHF):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.args = args

        self.lora_a_q_proj = None
        self.lora_b_q_proj = None
        self.lora_a_k_proj = None
        self.lora_b_k_proj = None
        self.lora_a_v_proj = None
        self.lora_b_v_proj = None
        self.lora_a_o_proj = None
        self.lora_b_o_proj = None

    def init_weights(self):
        super().init_weights()

        self.lora_a_q_proj = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_q_proj = ColumnParallelLinear(
            self.args.r,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_k_proj = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_k_proj = ColumnParallelLinear(
            self.args.r,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_v_proj = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_v_proj = ColumnParallelLinear(
            self.args.r,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_o_proj = RowParallelLinear(
            self.args.n_heads * self.head_dim,
            self.args.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).type(self.args.lora_dtype)
        self.lora_b_o_proj = nn.Linear(
            self.args.r,
            self.args.dim,
            bias=False
        ).type(self.args.lora_dtype)
        init.zeros_(self.lora_b_wo.weight)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x) + apply_lora(x, self.lora_a_q_proj, self.lora_b_q_proj)
        xk = self.k_proj(x) + apply_lora(x, self.lora_a_k_proj, self.lora_b_k_proj)
        xv = self.v_proj(x) + apply_lora(x, self.lora_a_v_proj, self.lora_b_v_proj)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_heads, self.head_dim)

        cos, sin = self.rotary_emb.forward(xv.transpose(1, 2), seq_len=seq_len + start_pos)
        position_ids = compute_position_ids(start_pos, seq_len).to(x.device)
        xq, xk = apply_rotary_pos_emb(xq.transpose(1, 2), xk.transpose(1, 2), cos, sin, position_ids)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)

        output = self.apply_attention(xq, xk, xv, mask)

        return self.o_proj(output) + apply_lora(output, self.lora_a_o_proj, self.lora_b_o_proj)


class LoraLlamaFeedForwardHF(LlamaFeedForwardHF):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.args = args
        self.r = args.r

        self.lora_a_gate_proj = None
        self.lora_b_gate_proj = None
        self.lora_a_down_proj = None
        self.lora_b_down_proj = None
        self.lora_a_up_proj = None
        self.lora_b_up_proj = None

    def init_weights(self):
        super().init_weights()

        self.lora_a_gate_proj = nn.Linear(
            self.dim,
            self.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_gate_proj = ColumnParallelLinear(
            self.r,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_down_proj = RowParallelLinear(
            self.hidden_dim,
            self.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).type(self.args.lora_dtype)
        self.lora_b_down_proj = nn.Linear(
            self.r,
            self.dim,
            bias=False
        ).type(self.args.lora_dtype)
        init.zeros_(self.lora_b_w2.weight)
        self.lora_a_up_proj = nn.Linear(
            self.dim,
            self.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_up_proj = ColumnParallelLinear(
            self.r,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)

    def forward(self, x):
        w1_x = self.gate_proj(x) + apply_lora(x, self.lora_a_gate_proj, self.lora_b_gate_proj)
        w3_x = self.up_proj(x) + apply_lora(x, self.lora_a_up_proj, self.lora_b_up_proj)
        out = F.silu(w1_x) * w3_x
        return self.down_proj(out) + apply_lora(out, self.lora_a_down_proj, self.lora_b_down_proj)


class LoraLlamaTransformerBlockHF(LlamaTransformerBlockHF):
    def __init__(self, layer_id: int, args: LoraLlamaArgs):
        super().__init__(layer_id, args)
        self.self_attn = LoraLlamaAttentionHF(args)
        self.mlp = LoraLlamaFeedForwardHF(args)


class LoraLlamaHeadHF(LlamaHeadHF):
    def __init__(self, args: LlamaArgs):
        super().__init__(args)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(LlamaTransformerBlockHF(layer_id, args))


class LoraLlamaHF(LlamaHF):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.args = args
        self.model = LoraLlamaHeadHF(args)
        self.lora_a_lm_head = None
        self.lora_b_lm_head = None

    def init_weights(self):
        super().init_weights()

        self.lora_a_lm_head = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_lm_head = ColumnParallelLinear(
            self.args.r,
            self.args.vocab_size,
            bias=False,
            gather_output=True,
            init_method=init.zeros_
        ).type(self.args.lora_dtype)

        # Freeze parameters
        self._freeze()

    def forward(self, tokens: torch.Tensor, start_pos=0, use_cache=False):
        h = self.model.forward(tokens, start_pos, use_cache)
        output = self.lm_head(h) + apply_lora(h, self.lora_a_lm_head, self.lora_b_lm_head)
        return CausalLMOutputs(logits=output, hidden_states=h)

    def _freeze(self):
        """ Freeze all parameters but lora ones. """
        frozen_names = []
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad_(False)
                frozen_names.append(name)
