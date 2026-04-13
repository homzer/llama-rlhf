import collections
from typing import List

ProcessorOutputs = collections.namedtuple(
    "ProcessorOutputs", [
        "input_ids", "pixel_values_images", "image_grid_thw", "pixel_values_videos", "video_grid_thw"
    ]
)


class Processor:
    def __init__(self, vocab_size, bos_id, eos_id, pad_id):
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.system_prompt = None

    def apply_chat_template(
            self, texts: List[str], images: List[str] = None, videos: List[str] = None
    ) -> ProcessorOutputs:
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
        raise NotImplementedError

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        raise NotImplementedError

    def decode(self, t: List[int]) -> str:
        raise NotImplementedError


class AutoProcessor:
    _registry = {}

    @classmethod
    def register(cls, name):
        def wrapper(processor_cls):
            cls._registry[name] = processor_cls
            return processor_cls
        return wrapper

    @classmethod
    def from_pretrained(cls, model_type: str, tokenizer_file: str) -> Processor:
        return cls._registry[model_type](tokenizer_file)
