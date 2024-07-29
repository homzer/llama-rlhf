import os
from typing import List

from sentencepiece import SentencePieceProcessor

from src.tokenizers.tokenizer import Tokenizer


class LlamaTokenizer(Tokenizer):
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

    def apply_chat_template(self, messages: List[dict]) -> str:  # TODO
        """
        :param messages: [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "greetings!"}]
        :return:
        """
        s = ""
        user_template = "\n\nHuman: "
        assistant_template = "\n\nAssistant: "
        for message in messages:
            if message['role'] == 'user':
                s += user_template + message['content']
            elif message['role'] == 'assistant':
                s += assistant_template + message['content']
            else:
                raise ValueError(message['role'])
        s += assistant_template
        return s

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        return self.model.Encode(s, add_bos=bos, add_eos=eos)

    def decode(self, t: List[int]) -> str:
        return self.model.Decode(t)

    def tokenize(self, s: str, bos: bool = False, eos: bool = False):
        return self.model.Encode(s, out_type=str, add_bos=bos, add_eos=eos)

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        os.system(f"cp {self.model_file} {os.path.join(save_dir, 'tokenizer.model')}")
