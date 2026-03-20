import collections
from typing import List

from transformers import Qwen3VLProcessor as Qwen3VLProcessorHf


ProcessorOutputs = collections.namedtuple(
    "ProcessorOutputs", [
        "input_ids", "pixel_values_images", "image_grid_thw", "pixel_values_videos", "video_grid_thw"
    ]
)


class Qwen3VLProcessor:
    def __init__(self, model_dir: str):
        self.model = Qwen3VLProcessorHf.from_pretrained(model_dir)
        self.bos_id = self.model.tokenizer.bos_token_id or self.model.tokenizer.convert_tokens_to_ids('<|im_start|>')
        self.eos_id=self.model.tokenizer.eos_token_id
        self.pad_id=self.model.tokenizer.pad_token_id
        self.image_token_id = self.model.image_token_id
        self.video_token_id = self.model.video_token_id

    def apply_chat_template(self, messages: list) -> ProcessorOutputs:
        """
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "..."},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": "..."},
                        {"type": "text", "text": "Describe this video."},
                    ],
                },
            ],
        ]
        """
        outputs = self.model.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_mm_token_type_ids=False
        )
        return ProcessorOutputs(
            input_ids=outputs["input_ids"],
            pixel_values_images=outputs["pixel_values"],
            image_grid_thw=outputs["image_grid_thw"],
            pixel_values_videos=outputs["pixel_values_videos"],
            video_grid_thw=outputs["video_grid_thw"]
        )

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        encode = self.model.tokenizer.encode(s)
        if bos and (len(encode) == 0 or encode[0] != self.bos_id):
            encode.insert(0, self.bos_id)
        if eos and (len(encode) == 0 or encode[-1] != self.eos_id):
            encode.append(self.eos_id)
        return encode

    def decode(self, t: List[int]) -> str:
        return self.model.tokenizer.decode(t, skip_special_tokens=True)
