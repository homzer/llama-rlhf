import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modeling_acts import RMSNorm
from src.models.modeling_args import QwenVLArgs
from src.models.qwen import QwenTransformerBlock
from src.parallel.model_parallel.layers import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding
from src.utils import rotate_half


def apply_rotary_pos_emb_vision(
        q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """ See transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb() """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class QwenVisionPatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = 14, temporal_patch_size: int = 2, in_channels: int = 3, embed_dim: int = 1152):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class QwenVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        seq = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class QwenVisionAttention(nn.Module):
    def __init__(self, args: QwenVLArgs):
        super().__init__()
        self.args = args
        self.dim = args.vision_config_hidden_size
        self.num_heads = args.vision_config_num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = None  # parallel column splitting and interleaving
        self.proj = None
        self.qkv_fn = lambda x: self.qkv(x)
        self.proj_fn = lambda x: self.proj(x)

    def init_weights(self):
        self.qkv = ColumnParallelLinear(
            self.dim,
            self.dim * 3,
            bias=True,
            gather_output=False,
            init_method=lambda x: x,
        ).type(self.args.dtype)
        self.proj = RowParallelLinear(
            self.dim,
            self.dim,
            bias=True,
            input_is_parallel=True,
            init_method=lambda x: x,
        ).type(self.args.dtype)

    # Copied from src.models.modeling.AttentionForCausalLM.apply_attention
    @staticmethod
    def apply_attention(xq, xk, xv, mask=None) -> torch.Tensor:
        bsz, seq_len, n_heads, head_dim = xq.shape
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        if scores.dtype == torch.float16:
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        else:
            scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, xv)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return output

    # Copied from src.models.modeling.AttentionForCausalLM.repeat_kv
    @staticmethod
    def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The tensors go from (batch_size, seq_len, num_key_value_heads, head_dim) to
        (batch_size, seq_len, num_attention_heads, head_dim)
        """
        keys = torch.repeat_interleave(keys, repeats=repeats, dim=2)
        values = torch.repeat_interleave(values, repeats=repeats, dim=2)
        return keys, values


    def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] = None
    ):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv_fn(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # [1, seq_len, num_heads, head_dim]
        query_states = query_states.unsqueeze(0)
        key_states = key_states.unsqueeze(0)
        value_states = value_states.unsqueeze(0)

        # Process each chunk separately
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=1) for tensor in (query_states, key_states, value_states)
        ]
        attn_output = torch.cat([
            self.apply_attention(q, *self.repeat_kv(k, v, self.num_key_value_groups)) for q, k, v in zip(*splits)
        ], dim=1).reshape(seq_length, -1).contiguous()
        return self.proj_fn(attn_output)


class QwenVisionMLP(nn.Module):
    def __init__(self, args: QwenVLArgs, bias: bool = False):
        super().__init__()
        self.args = args
        self.bias = bias
        self.hidden_size = args.vision_config_hidden_size
        self.intermediate_size = args.vision_config_intermediate_size
        self.gate_proj = None
        self.down_proj = None
        self.up_proj = None
        self.gate_proj_fn = lambda x: self.gate_proj(x)
        self.down_proj_fn = lambda x: self.down_proj(x)
        self.up_proj_fn = lambda x: self.up_proj(x)

    def init_weights(self):
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size, self.intermediate_size,
            bias=self.bias,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.down_proj = RowParallelLinear(
            self.intermediate_size, self.hidden_size,
            bias=self.bias,
            input_is_parallel=True,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.up_proj = ColumnParallelLinear(
            self.hidden_size, self.intermediate_size,
            bias=self.bias,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)

    def forward(self, x):
        return self.down_proj_fn(F.silu(self.gate_proj_fn(x)) * self.up_proj_fn(x))


class QwenVisionBlock(nn.Module):
    def __init__(self, args: QwenVLArgs):
        super().__init__()
        self.args = args
        self.norm1 = None
        self.norm2 = None
        self.attn = QwenVisionAttention(args)
        self.mlp = QwenVisionMLP(args, bias=True)

    def init_weights(self):
        self.attn.init_weights()
        self.mlp.init_weights()
        self.norm1 = RMSNorm(self.args.vision_config_hidden_size, eps=1e-6).type(self.args.dtype)
        self.norm2 = RMSNorm(self.args.vision_config_hidden_size, eps=1e-6).type(self.args.dtype)

    def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            position_embeddings: torch.Tensor = None
    ):
        hidden_states = hidden_states + self.attn.forward(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class QwenVisionPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = RMSNorm(self.context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.dim)
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


class QwenVisionModel(nn.Module):
    def __init__(self, args: QwenVLArgs):
        super().__init__()
        self.args = args
        self.spatial_merge_size = args.vision_config_spatial_merge_size
        self.patch_size = args.vision_config_patch_size
        self.fullatt_block_indexes = args.vision_config_fullatt_block_indexes
        self.window_size = args.vision_config_window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.head_dim = args.vision_config_hidden_size // args.vision_config_num_heads

        self.patch_embed = None
        self.rotary_pos_emb = None
        self.blocks = nn.ModuleList([QwenVisionBlock(args) for _ in range(args.vision_config_depth)])
        self.merger = None

    def init_weights(self):
        self.patch_embed = QwenVisionPatchEmbedding(
            patch_size=self.args.vision_config_patch_size,
            temporal_patch_size=self.args.vision_config_temporal_patch_size,
            in_channels=self.args.vision_config_in_chans,
            embed_dim=self.args.vision_config_hidden_size
        ).type(self.args.dtype)
        self.rotary_pos_emb = QwenVisionRotaryEmbedding(self.head_dim // 2).type(self.args.dtype)
        for block in self.blocks:
            block.init_weights()
        self.merger = QwenVisionPatchMerger(
            dim=self.args.vision_config_out_hidden_size,
            context_dim=self.args.vision_config_hidden_size,
            spatial_merge_size=self.args.vision_config_spatial_merge_size
        ).type(self.args.dtype)


    # Copied from modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel
    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw.tolist():
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
                )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
                )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    # Copied from modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel
    def get_window_index(self, grid_thw):
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size
        grid_thw_list = grid_thw.tolist()

        for grid_t, grid_h, grid_w in grid_thw_list:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
                )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += grid_t * llm_grid_h * llm_grid_w
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        hidden_states = hidden_states.to(next(self.parameters()).device)
        grid_thw = grid_thw.to(next(self.parameters()).device)

        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(cu_window_seqlens).to(grid_thw)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0
        ).to(grid_thw)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_num, block in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = block(hidden_states, cu_seqlens_now, position_embeddings)

        merged_hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        merged_hidden_states = merged_hidden_states[reverse_indices, :]

        return merged_hidden_states


class QwenLanguageModel(nn.Module):
    def __init__(self, args: QwenVLArgs):
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

    def forward(self, text_embeds: torch.Tensor, start_pos: int, use_cache: bool):
        seq_len = text_embeds.shape[1]

        mask = None
        if seq_len > 1:
            mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=text_embeds.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(text_embeds)

        for layer in self.layers:
            text_embeds = layer(text_embeds, start_pos, mask, use_cache)
        return self.norm(text_embeds)


class QwenVLHead(nn.Module):
    def __init__(self, args: QwenVLArgs):
        super().__init__()
        self.args = args
        self.visual = QwenVisionModel(args)
        self.language_model = QwenLanguageModel(args)
        self.rope_deltas = None

    @staticmethod
    def get_vision_position_ids(
            start_position: int,
            grid_thw: list | torch.Tensor,
            temp_merge_size: int = 1,
            spatial_merge_size: int = 1,
            time_interval: int = 1,
            device: str | torch.device | None = None,
    ):
        """
        Compute 3D positional indices for vision tokens derived from a single image or video input.

        The positions are generated from the input grid defined by temporal (T), height (H), and
        width (W) dimensions. Temporal and spatial dimensions can be downscaled according to the
        merge sizes used in the vision backbone. The resulting positions are offset by `start_position`.

        Args:
            start_position (`int`):
                Offset added to all computed positional indices.
            grid_thw (`Sequence[int]` or `torch.Tensor` of shape `(3,)`):
                The (T, H, W) grid representing the feature layout of the current image or video after patch embedding.
            temp_merge_size (`int`, *optional*):
                Factor by which the temporal dimension is reduced in the backbone. The temporal grid size is divided
                by this value. Defaults to 1.
            spatial_merge_size (`int`, *optional*):
                Factor by which the spatial dimensions (H and W) are reduced in the backbone. Both H and W are divided
                by this value. Defaults to 1.
            time_interval (`int`, *optional*):
                Spacing factor applied between consecutive temporal position indices.Defaults to 1.
            device (`str` or `torch.device`, *optional*):
                Device on which the resulting tensor is allocated. If `None`, uses the current default device.

        Returns:
            torch.LongTensor of shape (3, sequence_length):
                Positional indices for temporal, height, and width dimensions,
                flattened into sequence form and offset by `start_position`.
        """
        llm_grid_t, llm_grid_h, llm_grid_w = (
            grid_thw[0].item() // temp_merge_size,
            grid_thw[1].item() // spatial_merge_size,
            grid_thw[2].item() // spatial_merge_size,
        )

        image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
        position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
            llm_grid_h * llm_grid_t
        )
        position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
            llm_grid_w * llm_grid_t
        )
        position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
        position_temporal = position_temporal * time_interval
        vision_position_ids = torch.stack([position_temporal, position_height, position_width], dim=0)

        return vision_position_ids

    def get_rope_index(
            self,
            input_ids: torch.Tensor,
            mm_token_type_ids: torch.Tensor,
            image_grid_thw: torch.Tensor | None = None,
            video_grid_thw: torch.Tensor | None = None,
            second_per_grid_ts: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's sizes. The utility expects a `vision + text`
        sequence and will error out otherwise. For pure text sequence, please rely on model's auto-inferred
        position ids. In a mixed vision + text sequence, vision tokens use 3D RoPE (temporal, height, width)
        while text tokens use standard 1D RoPE.

        Example:
            Temporal patches: 3; Height patches: 2; Width patches: 2
            Each vision input results in (temporal x height × width) positions. Here: 3 x 2 × 2 = 12 positions total.

            Temporal position IDs are spaced by:
                `interval = tokens_per_second * temporal_patch_size / fps`

                If fps = 1; tokens_per_second = 25; temporal_patch_size = 2, temporal IDs increase by 50 for each temporal patch:
                `[0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]`

            Height IDs repeat per row: `[0, 0, 1, 1, ...]`
            Width IDs alternate per column: `[0, 1, 0, 1, ...]`
            Text tokens follow standard 1D RoPE and the position IDs grow consequently with a step of `1`

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`):
                Token type ids matching each modality to a different value in the input sequence, i.e. text (0), image (1), video (2).
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.args.vision_config_spatial_merge_size
        tokens_per_second = self.args.vision_config_tokens_per_second

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }
        second_per_grid_ts = (
            iter(second_per_grid_ts) if second_per_grid_ts is not None else iter([1] * input_ids.shape[1])
        )
        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                # text == 0
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                # image == 1, video == 2
                else:
                    grid_thw = next(grid_iters[modality_type])
                    time_interval = tokens_per_second * int(next(second_per_grid_ts))
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, 1, spatial_merge_size, time_interval, device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def compute_3d_position_ids(
            self,
            input_ids: torch.Tensor | None,
            image_grid_thw: torch.Tensor | None,
            video_grid_thw: torch.Tensor | None,
            attention_mask: torch.Tensor | None,
            past_key_values: torch.Tensor | None,
            second_per_grid_ts: torch.Tensor | None = None,
            mm_token_type_ids: torch.IntTensor | None = None,
    ) -> torch.Tensor | None:
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        can_compute_mrope = (
                input_ids is not None
                and mm_token_type_ids is not None
                and (image_grid_thw is not None or video_grid_thw is not None)
        )

        if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                second_per_grid_ts=second_per_grid_ts,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
        # Use pre-calculated rope-deltas to infer correct 3D position ids
        elif self.rope_deltas is not None:
            batch_size, seq_length = input_ids.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1)
            else:
                position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            position_ids = position_ids + delta.to(device=position_ids.device)
        else:
            # Can't build correct 3D positions. Let the model infer it from `cache_position`
            position_ids = None
        return position_ids

    def get_video_features(self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor):
        video_embeds = self.visual.forward(pixel_values_videos, grid_thw=video_grid_thw)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return torch.split(video_embeds, split_sizes)

    def get_image_features(self, pixel_values_images: torch.Tensor, image_grid_thw: torch.Tensor):
        image_embeds = self.visual.forward(pixel_values_images, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return torch.split(image_embeds, split_sizes)

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False,
            pixel_values_images: torch.Tensor = None,
            pixel_values_videos: torch.Tensor = None,
            image_grid_thw: torch.Tensor = None,
            video_grid_thw: torch.Tensor = None,
            second_per_grid_ts: torch.Tensor = None,
            mm_token_type_ids: torch.Tensor = None
    ):
        tokens = tokens.to(next(self.language_model.parameters()).device)

        text_embeds = self.language_model.embed_tokens(tokens)

        if pixel_values_images is not None:
            image_embeds = self.get_image_features(pixel_values_images, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(text_embeds)
            image_masks = tokens == self.args.image_token_id
            image_masks = image_masks.unsqueeze(-1).expand_as(text_embeds).to(text_embeds.device)
            text_embeds = text_embeds.masked_scatter(image_masks, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(text_embeds)
            video_masks = tokens == self.args.video_token_id
            video_masks = video_masks.unsqueeze(-1).expand_as(text_embeds).to(text_embeds.device)
            text_embeds = text_embeds.masked_scatter(video_masks, video_embeds)

        position_ids = self.compute_3d_position_ids(
            input_ids=tokens,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )  # TODO
        return self.language_model.forward(text_embeds, start_pos, use_cache)
