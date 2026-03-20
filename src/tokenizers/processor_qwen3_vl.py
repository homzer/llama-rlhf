import collections

from transformers import Qwen3VLProcessor as Qwen3VLProcessorHf


ProcessorOutputs = collections.namedtuple(
    "ProcessorOutputs", [
        "input_ids", "mm_token_type_ids", "pixel_values_images", "image_grid_thw", "pixel_values_videos", "video_grid_thw"
    ]
)


class Qwen3VLProcessor:
    def __init__(self, model_dir: str, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.model = Qwen3VLProcessorHf.from_pretrained(model_dir)

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
            padding='max_length',
            max_length=self.max_seq_len,
            return_tensors="pt"
        )
        return ProcessorOutputs(
            input_ids=outputs["input_ids"],
            mm_token_type_ids=outputs["mm_token_type_ids"],
            pixel_values_images=outputs["pixel_values"],
            image_grid_thw=outputs["image_grid_thw"],
            pixel_values_videos=outputs["pixel_values_videos"],
            video_grid_thw=outputs["video_grid_thw"]
        )
