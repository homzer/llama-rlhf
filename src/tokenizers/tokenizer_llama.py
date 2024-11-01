import os
import re
from typing import List

from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer

from src.tokenizers.tokenizer import Tokenizer


class LlamaTokenizer(Tokenizer):
    B_SYS: str = "<<SYS>>\n"
    E_SYS: str = "\n<</SYS>>\n\n"
    B_INST: str = "[INST]"
    E_INST: str = "[/INST]"

    def __init__(self, model_file: str):
        if not model_file.endswith("tokenizer.model"):
            model_file = os.path.join(model_file, "tokenizer.model")
        self.model_file = model_file
        self.model = SentencePieceProcessor()
        self.model.Init(model_file=model_file)
        super().__init__(
            vocab_size=self.model.vocab_size(),
            bos_id=self.model.bos_id(),
            eos_id=self.model.eos_id(),
            pad_id=0 if self.model.pad_id() == -1 else self.model.pad_id()
        )
        assert self.vocab_size == self.model.GetPieceSize()

    def apply_chat_template(self, messages: List[dict]) -> str:
        """
        :param messages: [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "greetings!"}]
        :return:
        """
        if messages[0]["role"] == "system":
            messages = [{
                "role": messages[1]["role"],
                "content": self.B_SYS + messages[0]["content"] + self.E_SYS + messages[1]["content"]
            }] + messages[2:]
        assert all([msg["role"] == "user" for msg in messages[::2]]) and all(
            [msg["role"] == "assistant" for msg in messages[1::2]]
        ), "Only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' (u/a/u/a/u...)."
        s = ""
        for message in messages:
            if message["role"] == "user":
                s += f"<s>{self.B_INST} {(message['content']).strip()} {self.E_INST}"
            elif message["role"] == "assistant":
                s += f" {(message['content']).strip()} </s>"
            else:
                raise ValueError(message["role"])
        return s

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        encode = []
        if s.startswith("<s>"):
            encode.append(self.bos_id)
            s = re.sub(r"^<s>", "", s)
        dialogs = s.split('<s>')
        for i, dialog in enumerate(dialogs):
            enc = []
            if dialog.endswith("</s>"):
                enc.append(self.eos_id)
                dialog = re.sub(r"</s>$", "", dialog)
            enc = [*self.model.Encode(dialog), *enc] if i == 0 else [self.bos_id, *self.model.Encode(dialog), *enc]
            encode.extend(enc)
        if bos and (len(encode) == 0 or encode[0] != self.bos_id):
            encode = [self.bos_id, *encode]
        if eos and (len(encode) == 0 or encode[-1] != self.eos_id):
            encode = [*encode, self.eos_id]
        return encode

    def decode(self, t: List[int]) -> str:
        return self.model.Decode(t)

    def tokenize(self, s: str) -> List[str]:
        return self.model.Encode(s, out_type=str)

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        os.system(f"cp {self.model_file} {os.path.join(save_dir, 'tokenizer.model')}")


class LlamaTokenizerHf(Tokenizer):
    def __init__(self, model_dir: str):
        self.model = AutoTokenizer.from_pretrained(model_dir)
        super().__init__(
            vocab_size=len(self.model.get_vocab()),
            bos_id=self.model.bos_token_id,
            eos_id=self.model.eos_token_id,
            pad_id=self.model.pad_token_id or 0
        )

    def apply_chat_template(self, messages: List[dict]) -> str:
        """
        :param messages: [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "greetings!"}]
        :return:
        """
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
