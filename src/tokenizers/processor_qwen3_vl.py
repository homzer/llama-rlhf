from typing import List

from transformers import Qwen3VLProcessor as Qwen3VLProcessorHf

from src.tokenizers.processor import Processor, ProcessorOutputs, AutoProcessor


@AutoProcessor.register("qwen3-vl")
class Qwen3VLProcessor(Processor):
    def __init__(self, model_dir: str):
        self.model = Qwen3VLProcessorHf.from_pretrained(model_dir)
        super().__init__(
            vocab_size=self.model.tokenizer.vocab_size,
            bos_id=self.model.tokenizer.bos_token_id or self.model.tokenizer.convert_tokens_to_ids('<|im_start|>'),
            eos_id=self.model.tokenizer.eos_token_id,
            pad_id=self.model.tokenizer.pad_token_id
        )
        self.image_token_id = self.model.image_token_id
        self.video_token_id = self.model.video_token_id

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
        messages = []
        if images is None:
            images = [None] * len(texts)
        if videos is None:
            videos = [None] * len(texts)
        for i, text in enumerate(texts):
            message = []
            if self.system_prompt is not None:
                message.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
            user_content = []
            if images[i] is not None and len(images[i]) > 0:
                user_content.append({"type": "image", "image": images[i]})
            if videos[i] is not None and len(videos[i]) > 0:
                user_content.append({"type": "video", "video": videos[i]})
            user_content.append({"type": "text", "text": texts[i]})
            message.append({"role": "user", "content": user_content})
            messages.append(message)
        outputs = self.model.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_mm_token_type_ids=False
        )
        return ProcessorOutputs(
            input_ids=outputs.get("input_ids"),
            pixel_values_images=outputs.get("pixel_values", None),
            image_grid_thw=outputs.get("image_grid_thw", None),
            pixel_values_videos=outputs.get("pixel_values_videos", None),
            video_grid_thw=outputs.get("video_grid_thw", None)
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
