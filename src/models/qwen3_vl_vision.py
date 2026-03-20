import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import GELUTanh

from src.models.modeling_args import Qwen3VLArgs
from src.parallel.model_parallel.layers import ColumnParallelLinear, RowParallelLinear
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


class Qwen3VisionPatchEmbedding(nn.Module):
    def __init__(self, args: Qwen3VLArgs):
        super().__init__()
        self.args = args
        self.patch_size = args.vision_config_patch_size
        self.temporal_patch_size = args.vision_config_temporal_patch_size
        self.in_channels = args.vision_config_in_channels
        self.embed_dim = args.vision_config_hidden_size

        self.kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim, kernel_size=self.kernel_size, stride=self.kernel_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return freqs


class Qwen3VisionPatchMerger(nn.Module):
    def __init__(self, args: Qwen3VLArgs, use_post_shuffle_norm=True) -> None:
        super().__init__()
        self.hidden_size = args.vision_config_hidden_size * (args.vision_config_spatial_merge_size ** 2)
        self.use_post_shuffle_norm = use_post_shuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_post_shuffle_norm else args.vision_config_hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, args.vision_config_out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_post_shuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class Qwen3VisionAttention(nn.Module):
    def __init__(self, args: Qwen3VLArgs):
        super().__init__()
        self.args = args
        self.dim = args.vision_config_hidden_size
        self.num_heads = args.vision_config_num_heads
        self.head_dim = self.dim // self.num_heads
        assert args.vision_config_num_heads % args.model_parallel_world_size == 0
        self.num_local_heads = args.vision_config_num_heads // args.model_parallel_world_size
        self.num_key_value_groups = 1
        self.qkv = None  # parallel column splitting and interleaving
        self.proj = None

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
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_local_heads, -1).permute(1, 0, 2, 3).unbind(0)
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
        return self.proj(attn_output)


class Qwen3VisionMLP(nn.Module):
    def __init__(self, args: Qwen3VLArgs):
        super().__init__()
        self.args = args
        self.hidden_size = args.vision_config_hidden_size
        self.intermediate_size = args.vision_config_intermediate_size
        self.linear_fc1 = None
        self.linear_fc2 = None
        self.act_fn = GELUTanh()

    def init_weights(self):
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=True,
            gather_output=False,
            init_method=lambda x: x,
        ).type(self.args.dtype)
        self.linear_fc2 = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=True,
            input_is_parallel=True,
            init_method=lambda x: x,
        ).type(self.args.dtype)

    def forward(self, x: torch.Tensor):
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3VisionBlock(nn.Module):
    def __init__(self, args: Qwen3VLArgs):
        super().__init__()
        self.args = args
        self.norm1 = None
        self.norm2 = None
        self.attn = Qwen3VisionAttention(args)
        self.mlp = Qwen3VisionMLP(args)

    def init_weights(self):
        self.attn.init_weights()
        self.mlp.init_weights()
        self.norm1 = nn.LayerNorm(self.args.vision_config_hidden_size, eps=1e-6).type(self.args.dtype)
        self.norm2 = nn.LayerNorm(self.args.vision_config_hidden_size, eps=1e-6).type(self.args.dtype)

    def forward(
            self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            position_embeddings: torch.Tensor = None
    ):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VisionModel(nn.Module):
    def __init__(self, args: Qwen3VLArgs):
        super().__init__()
        self.args = args
        self.spatial_merge_size = args.vision_config_spatial_merge_size
        self.patch_size = args.vision_config_patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.num_grid_per_side = int(args.vision_config_num_position_embeddings ** 0.5)
        self.head_dim = args.vision_config_hidden_size // args.vision_config_num_heads
        self.deepstack_visual_indexes = args.vision_config_deepstack_visual_indexes

        self.patch_embed = None
        self.pos_embed = None
        self.rotary_pos_emb = None
        self.blocks = nn.ModuleList([Qwen3VisionBlock(args) for _ in range(args.vision_config_depth)])
        self.merger = None
        self.deepstack_merger_list = None

    def init_weight(self):
        self.patch_embed = Qwen3VisionPatchEmbedding(self.args).type(self.args.dtype)
        self.pos_embed = nn.Embedding(
            self.args.vision_config_num_position_embeddings, self.args.vision_config_hidden_size
        ).type(self.args.dtype)
        self.rotary_pos_emb = Qwen3VisionRotaryEmbedding(self.head_dim // 2).type(self.args.dtype)
        for block in self.blocks:
            block.init_weights()
        self.merger = Qwen3VisionPatchMerger(self.args, use_post_shuffle_norm=False).type(self.args.dtype)
        self.deepstack_merger_list = nn.ModuleList(
            [Qwen3VisionPatchMerger(self.args) for _ in range(len(self.deepstack_visual_indexes))]
        ).type(self.args.dtype)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.args.vision_config_spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        hidden_states = hidden_states.to(next(self.parameters()))
        grid_thw = grid_thw.to(next(self.parameters()).device)

        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)

        Outputs = collections.namedtuple("Outputs", ["pooler_output", "deepstack_features"])
        return Outputs(pooler_output=merged_hidden_states, deepstack_features=deepstack_feature_lists)
