from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.checkpoint import CheckpointForGemma2
from src.models.modeling import (
    ParallelModelForCausalLM,
    CausalLMOutputs,
    AttentionForCausalLM
)
from src.models.modeling_acts import Gemma2RMSNorm, LogitsNormalize, RotaryEmbedding
from src.models.modeling_args import Gemma2Args
from src.parallel.initialize import set_model_parallel_barrier
from src.parallel.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding
)
from src.parallel.sequence_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region
)
from src.utils import compute_position_ids, apply_rotary_pos_emb_


class Gemma2Attention(AttentionForCausalLM):
    def __init__(self, args: Gemma2Args):
        super().__init__(args.max_seq_len)
        self.args = args
        self.head_dim = args.head_dim
        self.num_key_value_groups = args.num_attention_heads // args.num_key_value_heads
        self.num_local_heads = args.num_attention_heads // args.model_parallel_world_size
        self.num_local_key_value_heads = args.num_key_value_heads // args.model_parallel_world_size
        self.scaling = args.query_pre_attn_scalar ** -0.5

        self.rotary_emb = None
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None
        self.q_proj_fn = lambda x: self.q_proj(x)
        self.k_proj_fn = lambda x: self.k_proj(x)
        self.v_proj_fn = lambda x: self.v_proj(x)
        self.o_proj_fn = lambda x: self.o_proj(x)

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
            self.args.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.v_proj = ColumnParallelLinear(
            self.args.hidden_size,
            self.args.num_key_value_heads * self.head_dim,
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

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.args.max_position_embeddings,
            base=self.args.rope_theta,
        ).type(self.args.dtype)

    def apply_eager_attention(self, query, key, value, attention_mask, scaling, softcap):
        bsz, seq_len, n_heads, head_dim = query.shape
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if scaling is None:
            scaling = self.head_dim ** -0.5

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

        if softcap is not None:
            attn_weights = attn_weights / softcap
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * softcap

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask  # (bs, n_local_heads, slen, cache_len + slen)
        if attn_weights.dtype == torch.float16:
            scores = F.softmax(attn_weights.float(), dim=-1).type_as(query.dtype)
        else:
            scores = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(scores, value)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return output

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, local_seq_len, _ = x.size()
        xq, xk, xv = self.q_proj_fn(x), self.k_proj_fn(x), self.v_proj_fn(x)

        xq = xq.view(bsz, local_seq_len, self.num_local_heads, self.head_dim)
        xk = xk.view(bsz, local_seq_len, self.num_local_key_value_heads, self.head_dim)
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

        xk, xv = self.repeat_kv(xk, xv, self.num_key_value_groups)

        output = self.apply_eager_attention(xq, xk, xv, mask, self.scaling, self.args.attn_logit_softcapping)
        return self.o_proj_fn(output)


class Gemma2FeedForward(nn.Module):
    def __init__(self, args: Gemma2Args):
        super().__init__()
        self.args = args

        self.gate_proj = None
        self.down_proj = None
        self.up_proj = None
        self.gate_proj_fn = lambda x: self.gate_proj(x)
        self.down_proj_fn = lambda x: self.down_proj(x)
        self.up_proj_fn = lambda x: self.up_proj(x)

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
        return self.down_proj_fn(F.gelu(self.gate_proj_fn(x), approximate="tanh") * self.up_proj_fn(x))


class Gemma2TransformerBlock(nn.Module):
    def __init__(self, args: Gemma2Args):
        super().__init__()
        self.args = args
        self.self_attn = Gemma2Attention(args)
        self.mlp = Gemma2FeedForward(args)
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.pre_feedforward_layernorm = None
        self.post_feedforward_layernorm = None

    def init_weights(self):
        self.self_attn.init_weights()
        self.mlp.init_weights()
        self.input_layernorm = Gemma2RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps).type(self.args.dtype)
        self.post_attention_layernorm = Gemma2RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps).type(self.args.dtype)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps).type(self.args.dtype)
        self.post_feedforward_layernorm = Gemma2RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps).type(self.args.dtype)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache
    ):
        h = x + self.post_attention_layernorm(
            self.self_attn.forward(
                self.input_layernorm(x), start_pos, mask, use_cache
            )
        )
        out = h + self.post_feedforward_layernorm(
            self.mlp.forward(
                self.pre_feedforward_layernorm(h)
            )
        )
        return out


class Gemma2Head(nn.Module):
    def __init__(self, args: Gemma2Args):
        super().__init__()
        self.args = args

        self.embed_tokens = None
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.num_hidden_layers):
            self.layers.append(Gemma2TransformerBlock(args))
        self.norm = None

    def init_weights(self):
        self.embed_tokens = ParallelEmbedding(
            self.args.vocab_size, self.args.hidden_size, init_method=lambda x: x
        ).type(self.args.dtype)
        for layer in self.layers:
            layer.init_weights()
        self.norm = Gemma2RMSNorm(self.args.hidden_size, eps=self.args.rms_norm_eps).type(self.args.dtype)

    def forward(self, tokens: torch.Tensor, start_pos=0, use_cache=False):
        tokens = tokens.to(next(self.parameters()).device)
        _bsz, seq_len = tokens.shape

        # Sequence Parallel Op.
        if not use_cache:  # Bypass the function if performing autoregressive generation.
            tokens = scatter_to_sequence_parallel_region(tokens)
        h = self.embed_tokens(tokens)
        # normalized
        h = h * torch.tensor(self.args.hidden_size ** 0.5, dtype=h.dtype)

        mask = None
        if seq_len > 1:
            mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

            # Sequence Parallel Op.
            if not use_cache:  # Bypass the function if performing autoregressive generation.
                mask = scatter_to_sequence_parallel_region(mask.transpose(1, 2)).transpose(1, 2)

        for layer in self.layers:
            h = layer(h, start_pos, mask, use_cache)
        return self.norm(h)


class Gemma2(ParallelModelForCausalLM):
    def __init__(self, args: Gemma2Args):
        super().__init__()
        self.args = args
        self.model = Gemma2Head(args)
        self.lm_head = None
        self.lm_head_fn = lambda x: self.lm_head(x)
        self.logits_norm = LogitsNormalize(enable=self.args.use_logits_normalize)
        self.checkpoint = CheckpointForGemma2()

    def init_weights(self):
        self.model.init_weights()
        self.lm_head = ColumnParallelLinear(
            self.args.hidden_size, self.args.vocab_size, bias=False, init_method=lambda x: x
        ).type(self.args.dtype)

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ) -> CausalLMOutputs:
        h = self.model.forward(tokens, start_pos, use_cache)
        logits = self.lm_head_fn(h)
        if self.args.final_logit_softcapping is not None:
            logits = logits / self.args.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.args.final_logit_softcapping

        # Sequence Parallel Op.
        if not use_cache:  # Bypass the function if performing autoregressive generation.
            logits = gather_from_sequence_parallel_region(logits)

        return CausalLMOutputs(logits=self.logits_norm.forward(logits), hidden_states=h)

    # Copied from llama_hf.LlamaHf.load
    def load(self, ckpt_dir: str, verbose: bool = True, **kwargs):
        ckpt_dir = self.checkpoint.auto_split_or_merge_checkpoints(
            ckpt_dir=ckpt_dir,
            model_parallel_world_size=self.model_parallel_world_size,
            global_rank=self.global_rank
        )
        merge_lora = kwargs.get("merge_lora", True)
        super().load(ckpt_dir, verbose=verbose, merge_lora=merge_lora)

    # Copied from llama_hf.LlamaHf.flush
    def flush(self):
        for i in range(self.args.num_hidden_layers):
            self.model.layers[i].self_attn.flush()
        set_model_parallel_barrier()
