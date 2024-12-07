from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding
)

from src.checkpoint import CheckpointForQwen
from src.models.modeling import ParallelModelForCausalLM, CausalLMOutputs, AttentionForCausalLM, ParallelVerifier, \
    VerifierOutputs
from src.models.modeling_acts import Clamp, RMSNorm, RotaryEmbedding, LogitsNormalize
from src.models.modeling_args import QwenArgs, LoraQwenArgs
from src.utils import compute_position_ids, apply_rotary_pos_emb, apply_lora
from src.parallel.utils import set_model_parallel_barrier


class QwenAttention(AttentionForCausalLM):
    def __init__(self, args: QwenArgs):
        super().__init__(args.max_seq_len)
        self.args = args
        self.head_dim = args.hidden_size // args.num_attention_heads
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

    def init_weights(self):
        self.q_proj = ColumnParallelLinear(
            self.args.hidden_size,
            self.args.num_attention_heads * self.head_dim,
            bias=True,
            gather_output=False,
            init_method=lambda x: x,
        ).type(self.args.dtype)
        self.k_proj = ColumnParallelLinear(
            self.args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.v_proj = ColumnParallelLinear(
            self.args.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
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
        xk = xk.view(bsz, seq_len, self.num_local_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_local_key_value_heads, self.head_dim)

        cos, sin = self.rotary_emb.forward(xv.transpose(1, 2), seq_len=seq_len + start_pos)
        position_ids = compute_position_ids(start_pos, seq_len).to(x.device)
        xq, xk = apply_rotary_pos_emb(xq.transpose(1, 2), xk.transpose(1, 2), cos, sin, position_ids)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)

        xk, xv = self.repeat_kv(xk, xv, self.n_rep)

        output = self.apply_attention(xq, xk, xv, mask)
        return self.o_proj(output)


class QwenFeedForward(nn.Module):
    def __init__(self, args: QwenArgs):
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


class QwenTransformerBlock(nn.Module):
    def __init__(self, args: QwenArgs):
        super().__init__()
        self.args = args
        self.self_attn = QwenAttention(args)
        self.mlp = QwenFeedForward(args)
        self.clamp = Clamp(enable=args.use_clamp)

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


class QwenHead(nn.Module):
    def __init__(self, args: QwenArgs):
        super().__init__()
        self.args = args

        self.embed_tokens = None
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.num_hidden_layers):
            self.layers.append(QwenTransformerBlock(args))
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


class Qwen(ParallelModelForCausalLM):
    def __init__(self, args: QwenArgs):
        super().__init__()
        self.args = args
        self.model = QwenHead(args)
        self.lm_head = None
        self.logits_norm = LogitsNormalize(enable=self.args.use_logits_normalize)
        self.checkpoint = CheckpointForQwen()

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
        output = self.lm_head(h)
        return CausalLMOutputs(logits=self.logits_norm.forward(output), hidden_states=h)

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


class LoraQwenAttention(QwenAttention):
    def __init__(self, args: LoraQwenArgs):
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
    
    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, seq_len, _ = x.size()
        xq = self.q_proj(x) + apply_lora(x, self.lora_a_q_proj, self.lora_b_q_proj)
        xk = self.k_proj(x) + apply_lora(x, self.lora_a_k_proj, self.lora_b_k_proj)
        xv = self.v_proj(x) + apply_lora(x, self.lora_a_v_proj, self.lora_b_v_proj)

        xq = xq.view(bsz, seq_len, self.num_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_local_key_value_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_local_key_value_heads, self.head_dim)

        cos, sin = self.rotary_emb.forward(xv.transpose(1, 2), seq_len=seq_len + start_pos)
        position_ids = compute_position_ids(start_pos, seq_len).to(x.device)
        xq, xk = apply_rotary_pos_emb(xq.transpose(1, 2), xk.transpose(1, 2), cos, sin, position_ids)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)

        # xk = self.repeat_kv(xk)
        # xv = self.repeat_kv(xv)
        xk, xv = self.repeat_kv(xk, xv, self.n_rep)

        output = self.apply_attention(xq, xk, xv, mask)
        return self.o_proj(output) + apply_lora(output, self.lora_a_o_proj, self.lora_b_o_proj)
    
    def init_weights(self):
        super().init_weights()
        
        self.lora_a_q_proj = nn.Linear(
            self.args.hidden_size,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_q_proj = ColumnParallelLinear(
            self.args.r,
            self.args.num_attention_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_
        ).type(self.args.lora_dtype)
        self.lora_a_k_proj = nn.Linear(
            self.args.hidden_size,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_k_proj = ColumnParallelLinear(
            self.args.r,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_
        ).type(self.args.lora_dtype)
        self.lora_a_v_proj = nn.Linear(
            self.args.hidden_size,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_v_proj = ColumnParallelLinear(
            self.args.r,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_
        ).type(self.args.lora_dtype)
        self.lora_a_o_proj = RowParallelLinear(
            self.args.num_attention_heads * self.head_dim,
            self.args.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_
        ).type(self.args.lora_dtype)
        self.lora_b_o_proj = nn.Linear(
            self.args.r,
            self.args.hidden_size,
            bias=False
        ).type(self.args.lora_dtype)
        init.zeros_(self.lora_b_o_proj.weight)


class LoraQwenFeedForward(QwenFeedForward):
    def __init__(self, args: LoraQwenArgs):
        super().__init__(args)
        self.args = args

        self.lora_a_gate_proj = None
        self.lora_b_gate_proj = None
        self.lora_a_down_proj = None
        self.lora_b_down_proj = None
        self.lora_a_up_proj = None
        self.lora_b_up_proj = None

    def forward(self, x):
        x1 = self.gate_proj(x) + apply_lora(x, self.lora_a_gate_proj, self.lora_b_gate_proj)
        x3 = self.up_proj(x) + apply_lora(x, self.lora_a_up_proj, self.lora_b_up_proj)
        out = F.silu(x1) * x3
        return self.down_proj(out) + apply_lora(out, self.lora_a_down_proj, self.lora_b_down_proj)

    def init_weights(self):
        super().init_weights()
        self.lora_a_gate_proj = nn.Linear(
            self.args.hidden_size,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_gate_proj = ColumnParallelLinear(
            self.args.r,
            self.args.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=init.zeros_
        ).type(self.args.lora_dtype)
        self.lora_a_down_proj = RowParallelLinear(
            self.args.intermediate_size,
            self.args.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_
        ).type(self.args.lora_dtype)
        self.lora_b_down_proj = nn.Linear(
            self.args.r,
            self.args.hidden_size,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_a_up_proj = nn.Linear(
            self.args.hidden_size,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_up_proj = ColumnParallelLinear(
            self.args.r,
            self.args.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=init.zeros_
        ).type(self.args.lora_dtype)


class LoraQwenTransformerBlock(QwenTransformerBlock):
    def __init__(self, args: LoraQwenArgs):
        super().__init__(args)
        self.args = args
        self.self_attn = LoraQwenAttention(args)
        self.mlp = LoraQwenFeedForward(args)


class LoraQwenHead(QwenHead):
    def __init__(self, args: LoraQwenArgs):
        super().__init__(args)
        self.args = args
        self.layers = nn.ModuleList()
        for layer_id in range(args.num_hidden_layers):
            self.layers.append(LoraQwenTransformerBlock(args))


class LoraQwen(Qwen):
    def __init__(self, args: LoraQwenArgs):
        super().__init__(args)
        self.args = args
        self.model = LoraQwenHead(args)
        self.lora_a_lm_head = None
        self.lora_b_lm_head = None

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ) -> CausalLMOutputs:
        h = self.model.forward(tokens, start_pos, use_cache)
        output = self.lm_head(h) + apply_lora(h, self.lora_a_lm_head, self.lora_b_lm_head)
        return CausalLMOutputs(logits=self.logits_norm.forward(output), hidden_states=h)

    def init_weights(self):
        super().init_weights()

        self.lora_a_lm_head = nn.Linear(
            self.args.hidden_size,
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

        self._freeze()

    def load(self, ckpt_dir: str, verbose: bool = True, merge_lora: bool = False):
        super().load(ckpt_dir=ckpt_dir, verbose=verbose, merge_lora=merge_lora)

    def _freeze(self):
        """ Freeze all parameters but lora ones. """
        frozen_names = []
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad_(False)
                frozen_names.append(name)


class QwenVerifier(ParallelVerifier):
    def __init__(self, args: QwenArgs):
        super().__init__()
        self.args = args
        self.model = QwenHead(args)
        self.v_head = None
        self.checkpoint = CheckpointForQwen()

    def init_weights(self):
        self.model.init_weights()
        self.v_head = nn.Linear(
            self.args.hidden_size, 1, bias=False
        ).type(self.args.dtype)

    def forward(self, tokens: torch.Tensor) -> VerifierOutputs:
        h = self.model.forward(tokens)
        scores = self.v_head(h.type_as(self.v_head.weight)).squeeze(-1)  # [b, s]
        return VerifierOutputs(scores=scores)

    # Copied from llama_hf.LlamaHf.load
    def load(self, ckpt_dir: str, verbose: bool = True, **kwargs):
        ckpt_dir = self.checkpoint.auto_split_or_merge_checkpoints(
            ckpt_dir=ckpt_dir,
            model_parallel_world_size=self.model_parallel_world_size,
            global_rank=self.global_rank
        )
        merge_lora = kwargs.get("merge_lora", True)
        super().load(ckpt_dir, verbose=verbose, merge_lora=merge_lora)


class LoraQwenVerifier(QwenVerifier):
    def __init__(self, args: LoraQwenArgs):
        super().__init__(args)
        self.args = args
        self.model = LoraQwenHead(args)

    def init_weights(self):
        super().init_weights()
        self._freeze()

    def _freeze(self):
        """ Freeze all parameters but lora ones. """
        frozen_names = []
        for name, param in self.named_parameters():
            if 'lora' not in name and 'v_head' not in name:
                param.requires_grad_(False)
                frozen_names.append(name)

    def load(self, ckpt_dir: str, verbose: bool = True, merge_lora: bool = False):
        super().load(ckpt_dir=ckpt_dir, verbose=verbose, merge_lora=merge_lora)
