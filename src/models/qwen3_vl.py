import collections
import itertools

import torch
import torch.nn as nn

from src.checkpoint import CheckpointForQwen3VL
from src.models import ParallelModule
from src.models.modeling_acts import LogitsNormalize
from src.models.modeling_args import Qwen3VLArgs
from src.models.qwen3_vl_language import Qwen3LanguageModel
from src.models.qwen3_vl_vision import Qwen3VisionModel
from src.parallel.initialize import set_model_parallel_barrier
from src.parallel.model_parallel.layers import ColumnParallelLinear


class Qwen3VLHead(nn.Module):
    def __init__(self, args: Qwen3VLArgs):
        super().__init__()
        self.args = args
        self.visual = Qwen3VisionModel(args)
        self.language_model = Qwen3LanguageModel(args)
        self.spatial_merge_size = args.vision_config_spatial_merge_size
        self.image_token_id = args.image_token_id
        self.video_token_id = args.video_token_id
        self.rope_deltas = None

    def init_weights(self):
        self.visual.init_weight()
        self.language_model.init_weights()

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
            grid_thw[0] // temp_merge_size,
            grid_thw[1] // spatial_merge_size,
            grid_thw[2] // spatial_merge_size,
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
            input_masks: torch.Tensor,
            image_grid_thw: torch.Tensor | None = None,
            video_grid_thw: torch.Tensor | None = None,
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
            input_masks (`torch.Tensor` of shape `(batch_size, max_sequence_length)`):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.


        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        # Matching each modality to a different value in the input sequence, i.e. text (0), image (1), video (2).
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        mm_token_type_ids[input_ids == self.video_token_id] = 2

        input_masks = input_masks[:, : input_ids.shape[1]]

        mrope_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx in range(input_ids.shape[0]):
            cur_input_masks = input_masks[batch_idx].bool()
            cur_input_ids = input_ids[batch_idx][cur_input_masks]
            input_token_type = mm_token_type_ids[batch_idx][cur_input_masks]

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
                    grid_thw = next(grid_iters[modality_type]).tolist()
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, 1, self.spatial_merge_size, device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // self.spatial_merge_size
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[:, batch_idx, cur_input_masks] = llm_positions.to(position_ids.device)
            # subtracting cur_input_masks.nonzero()[0][0] for prefix padding purpose.
            mrope_deltas.append(llm_positions.max() + 1 - len(cur_input_ids) - cur_input_masks.nonzero()[0][0])
        mrope_deltas = torch.tensor(mrope_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_deltas

    def get_image_features(self, pixel_values_images: torch.Tensor, image_grid_thw: torch.Tensor):
        vision_output = self.visual.forward(pixel_values_images, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        pooler_output = torch.split(vision_output.pooler_output, split_sizes)
        Outputs = collections.namedtuple("Outputs", ["pooler_output", "deepstack_features"])
        return Outputs(pooler_output=pooler_output, deepstack_features=vision_output.deepstack_features)

    def get_video_features(self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor):
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def compute_3d_position_ids(
            self,
            input_ids: torch.Tensor,
            input_masks: torch.Tensor,
            start_pos: int,
            image_grid_thw: torch.Tensor | None = None,
            video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if start_pos == 0:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids=input_ids,
                input_masks=input_masks,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )
            self.rope_deltas = rope_deltas
        # Use pre-calculated rope-deltas to infer correct 3D position ids
        else:
            position_ids = torch.arange(start_pos, start_pos + input_ids.shape[1])
            position_ids = position_ids.view(1, 1, -1).expand(3, input_ids.shape[0], -1)
            position_ids = position_ids.to(self.rope_deltas.device) + self.rope_deltas.view(1, -1, 1)
        return position_ids

    def forward(
            self,
            input_ids: torch.Tensor,
            input_masks: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False,
            pixel_values_images: torch.Tensor = None,
            pixel_values_videos: torch.Tensor = None,
            image_grid_thw: torch.Tensor = None,
            video_grid_thw: torch.Tensor = None,
    ):
        input_ids = input_ids.to(next(self.parameters()).device)
        input_masks = input_masks.to(next(self.parameters()).device)
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if start_pos == 0:
            image_masks, video_masks = None, None
            deepstack_image_embeds, deepstack_video_embeds = None, None
            if pixel_values_images is not None:
                image_outputs = self.get_image_features(pixel_values_images, image_grid_thw)
                image_embeds = image_outputs.pooler_output
                deepstack_image_embeds = image_outputs.deepstack_features
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds)
                image_masks = input_ids == self.args.image_token_id
                image_masks = image_masks.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(image_masks, image_embeds)
            if pixel_values_videos is not None:
                video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = video_outputs.pooler_output
                deepstack_video_embeds = video_outputs.deepstack_features
                video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds)
                video_masks = input_ids == self.args.video_token_id
                video_masks = video_masks.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
                inputs_embeds = inputs_embeds.masked_scatter(video_masks, video_embeds)

            if image_masks is not None and video_masks is not None:
                image_masks = image_masks[..., 0]
                video_masks = video_masks[..., 0]
                visual_pos_masks = image_masks | video_masks
                deepstack_visual_embeds = []
                image_mask_joint = image_masks[visual_pos_masks]
                video_mask_joint = video_masks[visual_pos_masks]
                for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                    embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)
            elif image_masks is not None:
                image_masks = image_masks[..., 0]
                visual_pos_masks = image_masks
                deepstack_visual_embeds = deepstack_image_embeds
            elif video_masks is not None:
                video_masks = video_masks[..., 0]
                visual_pos_masks = video_masks
                deepstack_visual_embeds = deepstack_video_embeds

        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            input_masks=input_masks,
            start_pos=start_pos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        return self.language_model.forward(
            inputs_embeds=inputs_embeds,
            inputs_masks=input_masks,
            start_pos=start_pos,
            use_cache=use_cache,
            position_ids=position_ids,
            deepstack_visual_embeds=deepstack_visual_embeds,
            visual_pos_masks=visual_pos_masks
        )


class Qwen3VL(ParallelModule):
    def __init__(self, args: Qwen3VLArgs):
        super().__init__()
        self.args = args
        self.global_rank = args.global_rank
        self.model_parallel_world_size = args.model_parallel_world_size

        self.model = Qwen3VLHead(args)
        self.lm_head = None
        self.lm_head_fn = lambda x: self.lm_head(x)
        self.logits_norm = LogitsNormalize(enable=self.args.use_logits_normalize)
        self.checkpoint = CheckpointForQwen3VL()

    def init_weights(self):
        self.model.init_weights()
        self.lm_head = ColumnParallelLinear(
            self.args.text_config_hidden_size,
            self.args.text_config_vocab_size,
            bias=False,
            init_method=lambda x: x
        ).type(self.args.dtype)

    def forward(
            self,
            input_ids: torch.Tensor,
            input_masks: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False,
            pixel_values_images: torch.Tensor = None,
            pixel_values_videos: torch.Tensor = None,
            image_grid_thw: torch.Tensor = None,
            video_grid_thw: torch.Tensor = None,
    ) -> torch.Tensor:
        h = self.model.forward(
            input_ids=input_ids,
            input_masks=input_masks,
            start_pos=start_pos,
            use_cache=use_cache,
            pixel_values_images=pixel_values_images,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        logits = self.lm_head_fn(h)
        return self.logits_norm.forward(logits)

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
            self.model.language_model.layers[i].self_attn.flush()
        set_model_parallel_barrier()
