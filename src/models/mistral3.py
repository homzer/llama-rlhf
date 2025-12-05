import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.checkpoint import CheckpointForMinistral3, CheckpointForMistral3
from src.models.modeling import (
    ParallelModelForCausalLM,
    CausalLMOutputs,
    AttentionForCausalLM,
    ParallelVerifier, VerifierOutputs)
from src.models.modeling_acts import Clamp, RMSNorm, LogitsNormalize
from src.models.modeling_args import Mistral3Args
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


def _compute_yarn_parameters(
        rope_theta: float,
        head_dim: int,
        factor: float,
        mscale: float,
        mscale_all_dim: float,
        max_position_embeddings: int,
        device: torch.device,
        partial_rotary_factor: float = 1.0,
        attention_factor: float = None,
        original_max_position_embeddings: int = None,
        beta_fast: float = 32.,
        beta_slow: float = 1.,
        truncate: bool = True
):
    base = rope_theta
    dim = int(head_dim * partial_rotary_factor)

    if original_max_position_embeddings is not None:
        factor = max_position_embeddings / original_max_position_embeddings
    else:
        original_max_position_embeddings = max_position_embeddings

    def get_mscale(scale, mscale_=1.):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale_ * math.log(scale) + 1.0

    if attention_factor is None:
        if mscale and mscale_all_dim:
            attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
        else:
            attention_factor = get_mscale(factor)

    def find_correction_dim(num_rotations, dim_, base_, max_position_embeddings_):
        """Inverse dimension formula to find the dimension based on the number of rotations"""
        return (dim_ * math.log(max_position_embeddings_ / (num_rotations * 2 * math.pi))) / (2 * math.log(base_))

    def find_correction_range(low_rot, high_rot, dim_, base_, max_position_embeddings_, truncate_):
        """Find dimension range bounds based on rotations"""
        low_ = find_correction_dim(low_rot, dim_, base_, max_position_embeddings_)
        high_ = find_correction_dim(high_rot, dim_, base_, max_position_embeddings_)
        if truncate_:
            low_ = math.floor(low_)
            high_ = math.ceil(high_)
        return max(low_, 0), min(high_, dim_ - 1)

    def linear_ramp_factor(min_, max_, dim_):
        if min_ == max_:
            max_ += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim_, dtype=torch.float32) - min_) / (max_ - min_)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    pos_freqs = base ** (torch.arange(0, dim, 2).to(device=device, dtype=torch.float) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate)

    # Get n-dimensional rotational scaling corrected for extrapolation
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(device=device, dtype=torch.float)
    inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
            + inv_freq_extrapolation * inv_freq_extrapolation_factor
    )
    return inv_freq, attention_factor


def _get_llama_4_attn_scale(position_ids: torch.Tensor, beta: float, max_position_embeddings: int) -> torch.Tensor:
    scaling = 1 + beta * torch.log(1 + torch.floor(position_ids / max_position_embeddings))
    return scaling.unsqueeze(-1)


class Ministral3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, args: Mistral3Args, device=None):
        super().__init__()
        self.max_seq_len_cached = args.text_config_max_position_embeddings
        self.original_max_seq_len = args.text_config_max_position_embeddings

        self.args = args

        inv_freq, self.attention_scaling = _compute_yarn_parameters(
            rope_theta=args.text_config_rope_parameters_rope_theta,
            head_dim=args.text_config_head_dim,
            factor=args.text_config_rope_parameters_factor,
            mscale=args.text_config_rope_parameters_mscale,
            mscale_all_dim=args.text_config_rope_parameters_mscale_all_dim,
            max_position_embeddings=args.text_config_max_position_embeddings,
            device=device,
            original_max_position_embeddings=args.text_config_rope_parameters_original_max_position_embeddings,
            beta_fast=args.text_config_rope_parameters_beta_fast,
            beta_slow=args.text_config_rope_parameters_beta_slow
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @torch.no_grad()
    def forward(self, x: torch.Tensor, seq_len: int):
        position_ids = compute_position_ids(0, seq_len).to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(x)[None, :, :, :], sin.to(x)[None, :, :, :]


class Ministral3Attention(AttentionForCausalLM):
    def __init__(self, args: Mistral3Args):
        super().__init__(args.max_seq_len)
        self.args = args
        self.head_dim = args.text_config_head_dim
        self.num_local_heads = args.text_config_num_attention_heads // args.model_parallel_world_size
        self.num_key_value_heads = args.text_config_num_key_value_heads
        assert self.num_key_value_heads % args.model_parallel_world_size == 0
        self.num_local_key_value_heads = self.num_key_value_heads // args.model_parallel_world_size
        self.n_rep = args.text_config_num_attention_heads // args.text_config_num_key_value_heads

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
            self.args.text_config_hidden_size,
            self.args.text_config_num_attention_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            ).type(self.args.dtype)
        self.k_proj = ColumnParallelLinear(
            self.args.text_config_hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.v_proj = ColumnParallelLinear(
            self.args.text_config_hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            ).type(self.args.dtype)
        self.o_proj = RowParallelLinear(
            self.args.text_config_num_attention_heads * self.head_dim,
            self.args.text_config_hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
            ).type(self.args.dtype)

        self.rotary_emb = Ministral3RotaryEmbedding(args=self.args)

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

        cos, sin = self.rotary_emb.forward(x, seq_len=start_pos + seq_len)
        xq = apply_rotary_pos_emb_(xq.transpose(1, 2), cos, sin, local_position_ids)
        xk = apply_rotary_pos_emb_(xk.transpose(1, 2), cos, sin, position_ids)
        xq = xq * _get_llama_4_attn_scale(
            position_ids=local_position_ids,
            beta=self.args.text_config_rope_parameters_llama_4_scaling_beta,
            max_position_embeddings=self.args.text_config_rope_parameters_original_max_position_embeddings
        ).to(xq)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)

        xk, xv = self.repeat_kv(xk, xv, self.n_rep)

        output = self.apply_attention(xq, xk, xv, mask)
        return self.o_proj_fn(output)


class Ministral3FeedForward(nn.Module):
    def __init__(self, args: Mistral3Args):
        super().__init__()
        self.args = args

        self.intermediate_size = args.text_config_intermediate_size

        self.gate_proj = None
        self.down_proj = None
        self.up_proj = None
        self.gate_proj_fn = lambda x: self.gate_proj(x)
        self.down_proj_fn = lambda x: self.down_proj(x)
        self.up_proj_fn = lambda x: self.up_proj(x)

    def init_weights(self):
        self.gate_proj = ColumnParallelLinear(
            self.args.text_config_hidden_size, self.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.down_proj = RowParallelLinear(
            self.intermediate_size, self.args.text_config_hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.up_proj = ColumnParallelLinear(
            self.args.text_config_hidden_size, self.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)

    def forward(self, x):
        return self.down_proj_fn(F.silu(self.gate_proj_fn(x)) * self.up_proj_fn(x))


class Ministral3TransformerBlock(nn.Module):
    def __init__(self, args: Mistral3Args):
        super().__init__()
        self.args = args
        self.self_attn = Ministral3Attention(args)
        self.mlp = Ministral3FeedForward(args)
        self.clamp = Clamp(enable=args.use_clamp)

        self.input_layernorm = None
        self.post_attention_layernorm = None

    def init_weights(self):
        self.self_attn.init_weights()
        self.mlp.init_weights()
        self.input_layernorm = RMSNorm(
            self.args.text_config_hidden_size, eps=self.args.text_config_rms_norm_eps
        ).type(self.args.dtype)
        self.post_attention_layernorm = RMSNorm(
            self.args.text_config_hidden_size, eps=self.args.text_config_rms_norm_eps
        ).type(self.args.dtype)

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


class Ministral3Head(nn.Module):
    def __init__(self, args: Mistral3Args):
        super().__init__()
        self.args = args

        self.embed_tokens = None
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.text_config_num_hidden_layers):
            self.layers.append(Ministral3TransformerBlock(args))
        self.norm = None

    def init_weights(self):
        self.embed_tokens = ParallelEmbedding(
            self.args.text_config_vocab_size, self.args.text_config_hidden_size, init_method=lambda x: x
        ).type(self.args.dtype)
        for layer in self.layers:
            layer.init_weights()
        self.norm = RMSNorm(
            self.args.text_config_hidden_size, eps=self.args.text_config_rms_norm_eps
        ).type(self.args.dtype)

    def forward(self, tokens: torch.Tensor, start_pos=0, use_cache=False):
        tokens = tokens.to(next(self.parameters()).device)
        _bsz, seq_len = tokens.shape

        # Sequence Parallel Op.
        if not use_cache:  # Bypass the function if performing autoregressive generation.
            tokens = scatter_to_sequence_parallel_region(tokens)
        h = self.embed_tokens(tokens)

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


class Ministral3(ParallelModelForCausalLM):
    def __init__(self, args: Mistral3Args):
        super().__init__()
        self.args = args
        self.model = Ministral3Head(args)
        self.lm_head = None
        self.lm_head_fn = lambda x: self.lm_head(x)
        self.logits_norm = LogitsNormalize(enable=self.args.use_logits_normalize)
        self.checkpoint = CheckpointForMinistral3()

    def init_weights(self):
        self.model.init_weights()
        self.lm_head = ColumnParallelLinear(
            self.args.text_config_hidden_size, self.args.text_config_vocab_size, bias=False, init_method=lambda x: x
        ).type(self.args.dtype)

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ) -> CausalLMOutputs:
        h = self.model.forward(tokens, start_pos, use_cache)
        output = self.lm_head_fn(h)

        # Sequence Parallel Op.
        if not use_cache:  # Bypass the function if performing autoregressive generation.
            output = gather_from_sequence_parallel_region(output)

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

    def flush(self):
        for i in range(self.args.text_config_num_hidden_layers):
            self.model.layers[i].self_attn.flush()
        set_model_parallel_barrier()


class Mistral3(ParallelModelForCausalLM):
    def __init__(self, args: Mistral3Args):
        super().__init__()
        self.args = args
        self.language_model = Ministral3(args)
        self.checkpoint = CheckpointForMistral3()

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ) -> CausalLMOutputs:
        return self.language_model.forward(tokens, start_pos, use_cache)

    def init_weights(self):
        self.language_model.init_weights()

    def load(self, ckpt_dir: str, verbose: bool = True, **kwargs):
        ckpt_dir = self.checkpoint.auto_split_or_merge_checkpoints(
            ckpt_dir=ckpt_dir,
            model_parallel_world_size=self.model_parallel_world_size,
            global_rank=self.global_rank
        )
        merge_lora = kwargs.get("merge_lora", True)
        super().load(ckpt_dir, verbose=verbose, merge_lora=merge_lora)

    def flush(self):
        self.language_model.flush()


class Ministral3Verifier(ParallelVerifier):
    def __init__(self, args: Mistral3Args):
        super().__init__()
        self.args = args
        self.model = Ministral3Head(args)
        self.v_head = None
        self.v_head_fn = lambda x: self.v_head(x)
        self.checkpoint = CheckpointForMinistral3()

    def init_weights(self):
        self.model.init_weights()
        self.v_head = nn.Linear(
            self.args.text_config_hidden_size, 1, bias=False
        ).type(self.args.dtype)

    def forward(self, tokens: torch.Tensor) -> VerifierOutputs:
        h = self.model.forward(tokens)
        scores = self.v_head_fn(h).squeeze(-1)  # [b, s]

        # Sequence parallel op.
        scores = gather_from_sequence_parallel_region(scores)

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


class Mistral3Verifier(ParallelVerifier):
    def __init__(self, args: Mistral3Args):
        super().__init__()
        self.args = args
        self.language_model = Ministral3Verifier(args)

    def init_weights(self):
        self.language_model.init_weights()

    def forward(self, tokens: torch.Tensor) -> VerifierOutputs:
        return self.language_model.forward(tokens)

    def load(self, ckpt_dir: str, verbose: bool = True, **kwargs):
        self.language_model.load(ckpt_dir, verbose, **kwargs)
