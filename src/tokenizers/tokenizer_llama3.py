import os
from pathlib import Path
from typing import List, Dict, cast, Iterator

import tiktoken
from tiktoken.load import load_tiktoken_bpe
from transformers import AutoTokenizer

from src.tokenizers.tokenizer import Tokenizer

PAT_STR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
TIKTOKEN_MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACES_CHARS = 25_000


def _split_whitespaces_or_non_whitespaces(
        s: str, max_consecutive_slice_len: int
) -> Iterator[str]:
    """
    Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
    consecutive whitespaces or consecutive non-whitespaces.
    """
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    slice_start = 0

    for i in range(len(s)):
        is_now_space = s[i].isspace()

        if current_slice_is_space ^ is_now_space:
            current_slice_len = 1
            current_slice_is_space = is_now_space
        else:
            current_slice_len += 1
            if current_slice_len > max_consecutive_slice_len:
                yield s[slice_start:i]
                slice_start = i
                current_slice_len = 1
    yield s[slice_start:]


class Llama3Tokenizer(Tokenizer):
    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 256
    pat_str = PAT_STR

    def __init__(self, model_file: str):
        if not model_file.endswith("tokenizer.model"):
            model_file = os.path.join(model_file, "tokenizer.model")
        assert os.path.isfile(model_file), model_file
        self.model_file = model_file
        mergeable_ranks = load_tiktoken_bpe(model_file)
        num_base_tokens = len(mergeable_ranks)
        self.begin_of_text = "<|begin_of_text|>"
        self.end_of_text = "<|end_of_text|>"
        self.start_header = "<|start_header_id|>"
        self.end_header = "<|end_header_id|>"
        self.end_of_turn = "<|eot_id|>"
        special_tokens = [
                             self.begin_of_text,
                             self.end_of_text,
                             "<|reserved_special_token_0|>",
                             "<|reserved_special_token_1|>",
                             "<|reserved_special_token_2|>",
                             "<|reserved_special_token_3|>",
                             self.start_header,
                             self.end_header,
                             "<|reserved_special_token_4|>",
                             self.end_of_turn,  # end of turn
                         ] + [
                             f"<|reserved_special_token_{i}|>"
                             for i in range(5, self.num_reserved_special_tokens - 5)
                         ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_file).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.vocab_size: int = self.model.n_vocab
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens[self.begin_of_text]
        self.eos_id: int = self.special_tokens[self.end_of_text]
        self.pad_id: int = self.eos_id
        self.eot_id: int = self.special_tokens[self.end_of_turn]
        # This is very nasty! We got two different stop tokens. Don't know which to use.
        # When we apply chat template, `end_of_text` is set to be the stop token,
        # while `end_of_turn` is set when we not apply chat template.
        self.stop_tokens = {
            self.special_tokens[self.end_of_text],
            self.special_tokens[self.end_of_turn]
        }
        self.allowed_special = {
            self.begin_of_text,
            self.end_of_text,
            self.start_header,
            self.end_header,
            self.end_of_turn
        }
        self.skip_tokens = [self.begin_of_text, self.end_of_text, self.end_of_turn]
        self.skip_tokens_ids = {self.special_tokens[token] for token in self.skip_tokens}
        super().__init__(self.vocab_size, self.bos_id, self.eos_id, self.pad_id)

    def apply_chat_template(self, messages: List[dict]) -> str:
        """
        :param messages: [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "greetings!"}]
        :return:
        """
        # sys_str = "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
        # if messages[0]['role'] != "system":
        #     messages.insert(0, {"role": "system", "content": sys_str})
        # else:
        #     messages[0]["content"] = sys_str + messages[0]["content"]
        s = f"{self.begin_of_text}"
        for message in messages:
            s += f"{self.start_header}{message['role']}{self.end_header}\n\n{message['content']}{self.end_of_turn}"
        if messages[-1]["role"] != 'assistant':
            s += f"{self.start_header}assistant{self.end_header}\n\n"
        return s

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert isinstance(s, str)

        substrs = (substr for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
                   for substr in _split_whitespaces_or_non_whitespaces(
            s[i: i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
        ))
        t: List[int] = []
        for substr in substrs:
            t.extend(self.model.encode(substr, allowed_special=self.allowed_special, disallowed_special=()))
        if bos and (len(t) == 0 or t[0] != self.bos_id):
            t.insert(0, self.bos_id)
        if eos and (len(t) == 0 or t[-1] != self.eos_id):
            t.append(self.eos_id)
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        t = cast(List[int], t)
        # Skip special tokens
        t = [x for x in t if x not in self.skip_tokens_ids]
        return self.model.decode(t)

    def tokenize(self, s: str) -> List[str]:
        return [self.model.decode([token]) for token in self.encode(s)]

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        os.system(f"cp {self.model_file} {os.path.join(save_dir, 'tokenizer.model')}")


# TODO: check for correctness
class Llama3TokenizerHf(Tokenizer):
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
