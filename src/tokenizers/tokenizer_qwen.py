from typing import List

from transformers import AutoTokenizer as AutoTokenizerHf

from src.tokenizers import Tokenizer
from src.tokenizers.tokenizer import AutoTokenizer


@AutoTokenizer.register("qwen")
@AutoTokenizer.register("qwen3")
class QwenTokenizer(Tokenizer):
    def __init__(self, model_dir: str):
        self.model = AutoTokenizerHf.from_pretrained(model_dir)
        super().__init__(
            vocab_size=self.model.vocab_size,
            bos_id=self.model.bos_token_id or self.model.convert_tokens_to_ids('<|im_start|>'),
            eos_id=self.model.eos_token_id,
            pad_id=self.model.pad_token_id
        )
        self.system_prompt = None

    def apply_chat_template(self, messages: List[dict]) -> str:
        """
        :param messages: [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "greetings!"}]
        :return:
        """
        if self.system_prompt is not None:
            if messages[0]["role"] == "system":
                messages[0]["content"] = self.system_prompt
            else:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
        return self.model.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        encode = self.model.encode(s)
        if bos and (len(encode) == 0 or encode[0] != self.bos_id):
            encode.insert(0, self.bos_id)
        if eos and (len(encode) == 0 or encode[-1] != self.eos_id):
            encode.append(self.eos_id)
        return encode

    def decode(self, t: List[int]) -> str:
        return self.model.decode(t, skip_special_tokens=True)

    def save(self, save_dir: str):
        self.model.save_pretrained(save_dir)
