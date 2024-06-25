""" For validation only. """

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modeling import AttentionForCausalLM, ModelForCausalLM, CausalLMOutputs
from src.models.modeling_acts import RotaryEmbedding, Clamp, RMSNorm
from src.models.modeling_args import QwenArgs
from src.utils import compute_position_ids, apply_rotary_pos_emb, logits_normalize


class QwenAttention(AttentionForCausalLM):
    def __init__(self, args: QwenArgs):
        super().__init__(args.max_seq_len)
        self.args = args
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.n_rep = args.num_attention_heads // args.num_key_value_heads

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.args.max_position_embeddings,
            base=self.args.rope_theta,
        ).type(self.args.dtype)

        self.q_proj = nn.Linear(self.args.hidden_size, self.args.hidden_size, bias=True).type(self.args.dtype)
        self.k_proj = nn.Linear(self.args.hidden_size, self.num_key_value_heads * self.head_dim, bias=True).type(self.args.dtype)
        self.v_proj = nn.Linear(self.args.hidden_size, self.num_key_value_heads * self.head_dim, bias=True).type(self.args.dtype)
        self.o_proj = nn.Linear(self.args.hidden_size, self.args.hidden_size, bias=False).type(self.args.dtype)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, seq_len, _ = x.size()
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.args.num_attention_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        cos, sin = self.rotary_emb.forward(xv.transpose(1, 2), seq_len=seq_len + start_pos)
        position_ids = compute_position_ids(start_pos, seq_len).to(x.device)
        xq, xk = apply_rotary_pos_emb(xq.transpose(1, 2), xk.transpose(1, 2), cos, sin, position_ids)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)

        xk = self.repeat_kv(xk)
        xv = self.repeat_kv(xv)

        output = self.apply_attention(xq, xk, xv, mask)
        return self.o_proj(output)

    # Copied from src.models.llama_70B.LlamaAttention70B.repeat_kv
    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        bs, seqlen, n_kv_heads, head_dim = x.shape
        if self.n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, seqlen, n_kv_heads, self.n_rep, head_dim)
            .reshape(bs, seqlen, n_kv_heads * self.n_rep, head_dim)
        )


class QwenFeedForward(nn.Module):
    def __init__(self, args: QwenArgs):
        super().__init__()
        self.args = args

        self.gate_proj = nn.Linear(self.args.hidden_size, self.args.intermediate_size, bias=False).type(self.args.dtype)
        self.down_proj = nn.Linear(self.args.intermediate_size, self.args.hidden_size, bias=False).type(self.args.dtype)
        self.up_proj = nn.Linear(self.args.hidden_size, self.args.intermediate_size, bias=False).type(self.args.dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenTransformerBlock(nn.Module):
    def __init__(self, args: QwenArgs):
        super().__init__()
        self.args = args
        self.self_attn = QwenAttention(args)
        self.mlp = QwenFeedForward(args)
        self.clamp = Clamp(disable=not args.use_clamp)

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


class QwenHead(nn.Module):
    def __init__(self, args: QwenArgs):
        super().__init__()
        self.args = args

        self.embed_tokens = nn.Embedding(self.args.vocab_size, self.args.hidden_size).type(self.args.dtype)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.num_hidden_layers):
            self.layers.append(QwenTransformerBlock(args))
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


class Qwen(ModelForCausalLM):
    def __init__(self, args: QwenArgs):
        super().__init__()
        self.args = args
        self.model = QwenHead(args)
        self.lm_head = nn.Linear(self.args.hidden_size, self.args.vocab_size, bias=False).type(self.args.dtype)

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ) -> CausalLMOutputs:
        h = self.model.forward(tokens, start_pos, use_cache)
        output = self.lm_head(h)
        return CausalLMOutputs(logits=logits_normalize(output), hidden_states=h)

    def flush(self):
        for i in range(self.args.num_hidden_layers):
            self.model.layers[i].self_attn.flush()
