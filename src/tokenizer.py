from sentencepiece import SentencePieceProcessor
from transformers import GPT2Tokenizer as BaseGPT2Tokenizer
from typing import List
import os


class Tokenizer:
    def __init__(self, vocab_size, bos_id, eos_id, pad_id):
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        raise NotImplementedError

    def decode(self, t: List[int]) -> str:
        raise NotImplementedError


class LlamaTokenizer(Tokenizer):
    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), model_path
        self.model = SentencePieceProcessor()
        self.model.Init(model_file=model_path)
        super().__init__(
            vocab_size=self.model.vocab_size(),
            bos_id=self.model.bos_id(),
            eos_id=self.model.eos_id(),
            pad_id=0 if self.model.pad_id() == -1 else self.model.pad_id()
        )
        assert self.vocab_size == self.model.GetPieceSize()

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        return self.model.Encode(s, add_bos=bos, add_eos=eos)

    def decode(self, t: List[int]) -> str:
        return self.model.Decode(t)

    def tokenize(self, s: str, bos: bool = False, eos: bool = False):
        return self.model.Encode(s, out_type=str, add_bos=bos, add_eos=eos)

    def detokenize(self, t: List[int]) -> List[str]:
        return [self.id2piece(i) for i in t]

    def id2piece(self, idx: int) -> str:
        return self.model.IdToPiece(idx)

    def piece2id(self, s: str) -> int:
        return self.model.PieceToId(s)

    def whitespace_id(self):
        return self.piece2id('▁')


class GPT2Tokenizer(Tokenizer):
    def __init__(self, model_path: str):
        self.model = BaseGPT2Tokenizer.from_pretrained(model_path)
        super().__init__(
            vocab_size=self.model.vocab_size,
            bos_id=self.model.bos_token_id,
            eos_id=self.model.eos_token_id,
            pad_id=self.model.pad_token_id if self.model.pad_token_id else self.model.bos_token_id
        )

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        t = []
        if bos:
            t = [self.bos_id]
        t.extend(self.model.encode(s))
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: List[int]) -> str:
        return self.model.decode(t, skip_special_tokens=True)
