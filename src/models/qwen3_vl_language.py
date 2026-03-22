import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modeling import AttentionForCausalLM
from src.models.modeling_acts import RMSNorm
from src.models.modeling_args import Qwen3VLArgs
from src.parallel.model_parallel.layers import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding
from src.utils import rotate_half


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3LanguageRotaryEmbedding(nn.Module):
    def __init__(self, args: Qwen3VLArgs):
        super().__init__()
        self.args = args
        self.max_seq_len_cached = args.text_config_max_position_embeddings
        self.original_max_seq_len = args.text_config_max_position_embeddings

        base = args.text_config_rope_theta
        dim = args.text_config_head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section = args.text_config_rope_scaling_mrope_section


    @torch.no_grad()
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :]  # shape (3, bs, 1, positions)

        freqs = (inv_freq_expanded @ position_ids_expanded.to(inv_freq_expanded)).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        return cos.to(x), sin.to(x)

    @staticmethod
    def apply_interleaved_mrope(freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


class Qwen3LanguageAttention(AttentionForCausalLM):
    def __init__(self, args: Qwen3VLArgs):
        super().__init__(args.max_seq_len)
        self.args = args
        self.head_dim = args.text_config_head_dim
        self.num_key_value_groups = args.text_config_num_attention_heads // args.text_config_num_key_value_heads
        assert args.text_config_num_attention_heads % args.model_parallel_world_size == 0
        self.num_local_heads = args.text_config_num_attention_heads // args.model_parallel_world_size
        assert self.args.text_config_num_key_value_heads % args.model_parallel_world_size == 0
        self.num_local_key_value_heads = self.args.text_config_num_key_value_heads // args.model_parallel_world_size

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
            self.args.text_config_hidden_size,
            self.head_dim * self.args.text_config_num_attention_heads,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.k_proj = ColumnParallelLinear(
            self.args.text_config_hidden_size,
            self.head_dim * self.args.text_config_num_key_value_heads,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.v_proj = ColumnParallelLinear(
            self.args.text_config_hidden_size,
            self.head_dim * self.args.text_config_num_key_value_heads,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.o_proj = RowParallelLinear(
            self.head_dim * self.args.text_config_num_attention_heads,
            self.args.text_config_hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        ).type(self.args.dtype)
        self.q_norm = RMSNorm(self.head_dim, eps=self.args.text_config_rms_norm_eps).type(self.args.dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=self.args.text_config_rms_norm_eps).type(self.args.dtype)

    def forward(
            self,
            x: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            start_pos: int,
            mask: torch.Tensor = None,
            use_cache=False
    ) -> torch.Tensor:
        bsz, seq_len = x.shape[0], x.shape[1]
        xq, xk, xv = self.q_proj_fn(x), self.k_proj_fn(x), self.v_proj_fn(x)
        xq = self.q_norm(xq.view(bsz, seq_len, self.num_local_heads, self.head_dim))
        xk = self.k_norm(xk.view(bsz, seq_len, self.num_local_key_value_heads, self.head_dim))
        xv = xv.view(bsz, seq_len, self.num_local_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq.transpose(1, 2), xk.transpose(1, 2), cos, sin)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)
        xk, xv = self.repeat_kv(xk, xv, self.num_key_value_groups)
        output = self.apply_attention(xq, xk, xv, mask)
        return self.o_proj_fn(output)


class Qwen3LanguageMLP(nn.Module):
    def __init__(self, args: Qwen3VLArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.text_config_hidden_size
        self.intermediate_size = args.text_config_intermediate_size

        self.gate_proj = None
        self.down_proj = None
        self.up_proj = None
        self.gate_proj_fn = lambda x: self.gate_proj(x)
        self.down_proj_fn = lambda x: self.down_proj(x)
        self.up_proj_fn = lambda x: self.up_proj(x)

    def init_weights(self):
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size, self.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.down_proj = RowParallelLinear(
            self.intermediate_size, self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.up_proj = ColumnParallelLinear(
            self.hidden_size, self.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)

    def forward(self, x) -> torch.Tensor:
        return self.down_proj_fn(F.silu(self.gate_proj_fn(x)) * self.up_proj_fn(x))


class Qwen3LanguageTransformerBlock(nn.Module):
    def __init__(self, args: Qwen3VLArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.text_config_hidden_size
        self.rms_norm_eps = args.text_config_rms_norm_eps
        self.self_attn = Qwen3LanguageAttention(args)
        self.mlp = Qwen3LanguageMLP(args)
        self.input_layernorm = None
        self.post_attention_layernorm = None

    def init_weights(self):
        self.self_attn.init_weights()
        self.mlp.init_weights()
        self.input_layernorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps).type(self.args.dtype)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps).type(self.args.dtype)

    def forward(
            self,
            x: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            start_pos: int,
            mask: torch.Tensor = None,
            use_cache=False
    ) -> torch.Tensor:
        h = x + self.self_attn.forward(
            self.input_layernorm(x),
            position_embeddings=position_embeddings,
            start_pos=start_pos,
            mask=mask,
            use_cache=use_cache
        )
        h = h + self.mlp.forward(self.post_attention_layernorm(h))
        return h


class Qwen3LanguageModel(nn.Module):
    def __init__(self, args: Qwen3VLArgs):
        super().__init__()
        self.args = args

        self.embed_tokens = None
        self.layers = nn.ModuleList(
            [Qwen3LanguageTransformerBlock(args) for _ in range(args.text_config_num_hidden_layers)]
        )
        self.norm = None
        self.rotary_emb = None

    def init_weights(self):
        self.embed_tokens = ParallelEmbedding(
            self.args.text_config_vocab_size, self.args.text_config_hidden_size, init_method=lambda x: x
        ).type(self.args.dtype)
        for layer in self.layers:
            layer.init_weights()
        self.norm = RMSNorm(
            self.args.text_config_hidden_size, eps=self.args.text_config_rms_norm_eps
        ).type(self.args.dtype)
        self.rotary_emb = Qwen3LanguageRotaryEmbedding(self.args)

    @staticmethod
    def _deepstack_process(
            hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def forward(
            self,
            inputs_embeds: torch.Tensor,  # [batch_size, seq_len, hidden_size]
            inputs_masks: torch.Tensor,  # [batch_size, max_seq_len]
            start_pos: int,
            use_cache: bool,
            position_ids: torch.Tensor,
            deepstack_visual_embeds: list[torch.Tensor] = None,
            visual_pos_masks: torch.Tensor = None
    ) -> torch.Tensor:
        bsz, seq_len = inputs_embeds.shape[:2]

        masks = torch.full((bsz, 1, seq_len, start_pos + seq_len), float("-inf"), device=inputs_embeds.device)
        masks = torch.triu(masks, diagonal=start_pos + 1)
        for i in range(bsz):
            prefix_len = torch.nonzero(inputs_masks[i])[0][0].item()  # for prefix padding
            masks[i][..., : prefix_len] = float("-inf")

        # mask = None
        # if seq_len > 1:
        #     mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=inputs_embeds.device)
        #     mask = torch.triu(mask, diagonal=start_pos + 1).type_as(inputs_embeds)

        # if position_ids is None:
        #     position_ids = torch.arange(start_pos, start_pos + seq_len)
        #     position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        h = inputs_embeds
        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, position_embeddings, start_pos, masks, use_cache)

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                h = self._deepstack_process(
                    hidden_states=h,
                    visual_pos_masks=visual_pos_masks,
                    visual_embeds=deepstack_visual_embeds[layer_idx],
                )
        return self.norm(h)
