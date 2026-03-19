from typing import List


class Tokenizer:
    def __init__(self, vocab_size, bos_id, eos_id, pad_id):
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    def apply_chat_template(self, messages: List[dict]) -> str:
        """
        :param messages: [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "greetings!"}]
        :return: str
        """
        raise NotImplementedError

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        raise NotImplementedError

    def decode(self, t: List[int]) -> str:
        raise NotImplementedError

    def save(self, save_dir: str):
        raise NotImplementedError


class AutoTokenizer:
    _registry = {}

    @classmethod
    def register(cls, name):
        def wrapper(tokenizer_cls):
            cls._registry[name] = tokenizer_cls
            return tokenizer_cls
        return wrapper

    @classmethod
    def from_pretrained(cls, model_type: str, tokenizer_file: str) -> Tokenizer:
        return cls._registry[model_type](tokenizer_file)
