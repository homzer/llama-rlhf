import math
from typing import Optional

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils as utils
from src.modeling.modeling import ParallelModelForCausalLM, CausalLMOutputs
from src.modeling.modeling_args import LlamaArgs, LoraLlamaArgs


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
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
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


class AbstractAttentionHF(nn.Module):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        self.args = args
        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None

        self.cache_k = None
        self.cache_v = None
        self.rotary_emb = None

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
            if self.cache_k is None:
                self.cache_k = torch.zeros(
                    (bsz, self.args.max_seq_len, self.n_local_heads, self.head_dim)
                ).cuda()
            if self.cache_v is None:
                self.cache_v = torch.zeros(
                    (bsz, self.args.max_seq_len, self.n_local_heads, self.head_dim)
                ).cuda()

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos: start_pos + seq_len] = xk
            self.cache_v[:bsz, start_pos: start_pos + seq_len] = xv

            keys = self.cache_k[:bsz, : start_pos + seq_len]
            values = self.cache_v[:bsz, : start_pos + seq_len]
        else:
            keys = xk
            values = xv
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seq_len, -1)

        return self.o_proj(output)

    def flush(self):
        """ Clean self.cache for next inference. """
        self.cache_v = None
        self.cache_k = None


class AbstractFeedForwardHF(nn.Module):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        hidden_dim = int(2 * (4 * args.dim) / 3)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.hidden_dim = hidden_dim
        self.dim = args.dim
        self.gate_proj = None
        self.down_proj = None
        self.up_proj = None

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class AbstractTransformerBlockHF(nn.Module):
    def __init__(self, layer_id: int, args: LlamaArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.self_attn = AbstractAttentionHF(args)
        self.mlp = AbstractFeedForwardHF(args)
        self.layer_id = layer_id
        self.input_layernorm = None
        self.post_attention_layernorm = None

    def forward(self,
                x: torch.Tensor,
                start_pos: int,
                mask: Optional[torch.Tensor],
                use_cache):
        h = x + self.self_attn.forward(self.input_layernorm(x), start_pos, mask, use_cache)
        out = h + self.mlp.forward(self.post_attention_layernorm(h))
        return out


class AbstractBasicLLaMAHF(nn.Module):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        self.params = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.embed_tokens = None

        self.layers = torch.nn.ModuleList()
        self.norm = None

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


class AbstractLlamaHF(ParallelModelForCausalLM):
    def __init__(self, args: LlamaArgs):
        super().__init__(args.local_rank, args.world_size)
        self.params = args
        self.model = AbstractBasicLLaMAHF(args)
        self.lm_head = None

    def forward(self, tokens: torch.Tensor, start_pos=0, use_cache=False):
        h = self.model.forward(tokens, start_pos, use_cache)
        output = self.lm_head(h)
        return CausalLMOutputs(logits=output.float(), hidden_states=h)

    def flush(self):
        """ Clean cache in `Attention` module """
        for i in range(self.params.n_layers):
            self.model.layers[i].self_attn.flush()
        utils.barrier()


class AbstractLoraAttentionHF(AbstractAttentionHF):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)

        self.lora_a_q_proj = None
        self.lora_b_q_proj = None
        self.lora_a_k_proj = None
        self.lora_b_k_proj = None
        self.lora_a_v_proj = None
        self.lora_b_v_proj = None
        self.lora_a_o_proj = None
        self.lora_b_o_proj = None

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x) + self.lora_b_q_proj(self.lora_a_q_proj(x.float())).to(x.dtype)
        xk = self.k_proj(x) + self.lora_b_k_proj(self.lora_a_k_proj(x.float())).to(x.dtype)
        xv = self.v_proj(x) + self.lora_b_v_proj(self.lora_a_v_proj(x.float())).to(x.dtype)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_heads, self.head_dim)

        cos, sin = self.rotary_emb.forward(xv.transpose(1, 2), seq_len=seq_len + start_pos)
        position_ids = compute_position_ids(start_pos, seq_len).to(x.device)
        xq, xk = apply_rotary_pos_emb(xq.transpose(1, 2), xk.transpose(1, 2), cos, sin, position_ids)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        if use_cache:
            if self.cache_k is None:
                self.cache_k = torch.zeros(
                    (bsz, self.args.max_seq_len, self.n_local_heads, self.head_dim)
                ).cuda()
            if self.cache_v is None:
                self.cache_v = torch.zeros(
                    (bsz, self.args.max_seq_len, self.n_local_heads, self.head_dim)
                ).cuda()

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos: start_pos + seq_len] = xk
            self.cache_v[:bsz, start_pos: start_pos + seq_len] = xv

            keys = self.cache_k[:bsz, : start_pos + seq_len]
            values = self.cache_v[:bsz, : start_pos + seq_len]
        else:
            keys = xk
            values = xv
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seq_len, -1)

        return self.o_proj(output) + self.lora_b_o_proj(self.lora_a_o_proj(output.float())).to(output.dtype)


class AbstractLoraFeedForwardHF(AbstractFeedForwardHF):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.r = args.r

        self.lora_a_gate_proj = None
        self.lora_b_gate_proj = None
        self.lora_a_down_proj = None
        self.lora_b_down_proj = None
        self.lora_a_up_proj = None
        self.lora_b_up_proj = None

    def forward(self, x):
        w1_x = self.gate_proj(x) + self.lora_b_gate_proj(self.lora_a_gate_proj(x.float())).to(x.dtype)
        w3_x = self.up_proj(x) + self.lora_b_up_proj(self.lora_a_up_proj(x.float())).to(x.dtype)
        out = F.silu(w1_x) * w3_x
        return self.down_proj(out) + self.lora_b_down_proj(self.lora_a_down_proj(out.float())).to(out.dtype)


class AbstractLoraTransformerBlockHF(AbstractTransformerBlockHF):
    def __init__(self, layer_id: int, args: LoraLlamaArgs):
        super().__init__(layer_id, args)
        self.self_attn = AbstractLoraAttentionHF(args)
        self.mlp = AbstractLoraFeedForwardHF(args)


class AbstractLoraBasicLLaMAHF(AbstractBasicLLaMAHF):
    def __init__(self, args: LlamaArgs):
        super().__init__(args)


class AbstractLoraLlamaHF(AbstractLlamaHF):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.model = AbstractLoraBasicLLaMAHF(args)
        self.lora_a_lm_head = None
        self.lora_b_lm_head = None

    def forward(self, tokens: torch.Tensor, start_pos=0, use_cache=False):
        h = self.model.forward(tokens, start_pos, use_cache)
        output = self.lm_head(h) + self.lora_b_lm_head(self.lora_a_lm_head(h.float())).to(h.dtype)
        return CausalLMOutputs(logits=output.float(), hidden_states=h)

    def _freeze(self):
        """ Freeze all parameters but lora ones. """
        frozen_names = []
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad_(False)
                frozen_names.append(name)
