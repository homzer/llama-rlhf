import collections
from typing import List

import torch

from src.generator import sampling_strategy, get_output_masks
from src.models.modeling import ParallelModelForVisualLM
from src.tokenizers.processor import Processor


class GeneratorForVisualLM:
    def __init__(
            self,
            model: ParallelModelForVisualLM,
            processor: Processor,
            max_seq_len: int,
            temperature: float = 0.0,
            top_p: float = 0.95
    ):
        self.model = model
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.top_p = top_p

    def prepare_for_generation(self, texts: List[str], images: List[str] = None, videos: List[str] = None):
        # prefix padding
        processor_outputs = self.processor.apply_chat_template(texts=texts, images=images, videos=videos)
        bsz = len(processor_outputs.input_ids)
        start_pos = max([len(t) for t in processor_outputs.input_ids])
        assert start_pos < self.max_seq_len  # TODO
        tokens = torch.full((bsz, self.max_seq_len), self.processor.pad_id).long()
        for k, t in enumerate(processor_outputs.input_ids):
            tokens[k, start_pos - len(t): start_pos] = torch.tensor(t).long()
        input_masks = tokens != self.processor.pad_id
        Outputs = collections.namedtuple("Outputs", [
            "tokens", "input_masks", "start_pos", "pixel_values_images", "pixel_values_videos",
            "image_grid_thw", "video_grid_thw"
        ])
        return Outputs(
            tokens=tokens,
            input_masks=input_masks,
            start_pos=start_pos,
            pixel_values_images=processor_outputs.pixel_values_images,
            pixel_values_videos=processor_outputs.pixel_values_videos,
            image_grid_thw=processor_outputs.image_grid_thw,
            video_grid_thw=processor_outputs.video_grid_thw,
        )

    def sampling(self, logits: torch.Tensor) -> torch.Tensor:
        return sampling_strategy(logits, self.temperature, self.top_p)


    def model_forward(
            self,
            tokens: torch.Tensor,
            input_masks: torch.Tensor,
            start_pos: int,
            pixel_values_images: torch.Tensor = None,
            pixel_values_videos: torch.Tensor = None,
            image_grid_thw: torch.Tensor = None,
            video_grid_thw: torch.Tensor = None,
    ):
        bsz = tokens.shape[0]
        prev_pos = 0
        tokens = tokens.to(self.model.device()).clone()
        input_masks = input_masks.to(self.model.device())
        unfinished_sequences = torch.ones(size=[bsz], dtype=torch.long, device=self.model.device())
        tokens_logits = torch.zeros(tokens.shape)
        tokens_logprobs = torch.zeros(tokens.shape)
        for cur_pos in range(start_pos, self.max_seq_len):
            with torch.no_grad():
                logits = self.model.forward(
                    input_ids=tokens[:, prev_pos: cur_pos],
                    input_masks=input_masks,
                    start_pos=prev_pos,
                    use_cache=True,
                    pixel_values_images=pixel_values_images,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                )
                next_tokens = self.sampling(logits)
                tokens_logits = tokens_logits.to(logits)
                tokens_logprobs = tokens_logprobs.to(logits)
                tokens_logits[:, prev_pos: cur_pos] = torch.gather(
                    logits, dim=-1, index=next_tokens.unsqueeze(-1)
                ).squeeze(-1)
                tokens_logprobs[:, prev_pos: cur_pos] = torch.gather(
                    torch.log_softmax(logits, dim=-1), dim=-1, index=next_tokens.unsqueeze(-1)
                ).squeeze(-1)
                next_token = next_tokens[:, -1].reshape(-1)
                next_token = torch.where(
                    input_masks[:, cur_pos], tokens[:, cur_pos], next_token
                )
                tokens[:, cur_pos] = next_token
                prev_pos = cur_pos
                unfinished_sequences = unfinished_sequences * (
                    torch.any(torch.stack([next_token != self.processor.eos_id, input_masks[:, cur_pos]]), dim=0)
                ).long()
                if unfinished_sequences.max() == 0:
                    break

        self.model.flush()
        Outputs = collections.namedtuple("Outputs", ['tokens', 'tokens_logits', 'tokens_logprobs'])
        return Outputs(tokens=tokens, tokens_logits=tokens_logits, tokens_logprobs=tokens_logprobs)

    def get_output_masks(self, tokens, input_masks):
        return get_output_masks(
            tokens=tokens,
            input_masks=input_masks,
            eos_id=self.processor.eos_id
        )

    def decode_response(self, tokens, output_masks):
        responses = []
        # shift right
        shifted_output_masks = torch.full_like(output_masks, fill_value=False)
        shifted_output_masks[:, 1:] = output_masks[:, :-1]
        for t, m in zip(tokens, shifted_output_masks):
            responses.append(self.processor.decode(t[m].tolist()))
        return responses

    def forward(self, texts: List[str], images: List[str] = None, videos: List[str] = None) -> List[str]:
        self.model.eval()
        prep_outputs = self.prepare_for_generation(texts, images, videos)
        forward_outputs = self.model_forward(
            tokens=prep_outputs.tokens,
            input_masks=prep_outputs.input_masks,
            start_pos=prep_outputs.start_pos,
            pixel_values_images=prep_outputs.pixel_values_images,
            pixel_values_videos=prep_outputs.pixel_values_videos,
            image_grid_thw=prep_outputs.image_grid_thw,
            video_grid_thw=prep_outputs.video_grid_thw
        )
        output_masks = self.get_output_masks(forward_outputs.tokens, prep_outputs.input_masks)
        responses = self.decode_response(forward_outputs.tokens, output_masks)
        return responses
