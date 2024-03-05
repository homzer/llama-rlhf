import json
from dataclasses import dataclass

HF_CONFIG_MAP = {
    "hidden_size": "dim",
    "num_attention_heads": "n_heads",
    "num_hidden_layers": "n_layers",
    "rms_norm_eps": "norm_eps",
}


@dataclass
class Args:
    use_clamp: bool = False

    def _set_attribute(self, name, value):
        try:
            if getattr(self, name, None) is None:
                setattr(self, name, value)
        except AttributeError as err:
            print(f"Can't set `{name}` with value `{value}` for {self}")
            raise err

    def show(self):
        param_str = '\n'.join(['%30s = %s' % (k, v) for k, v in sorted(vars(self).items())])
        print('%30s   %s\n%s\n%s\n' % ('ATTRIBUTE', 'VALUE', '_' * 60, param_str))

    def from_json(self, filename: str):
        with open(filename, 'r', encoding='utf-8') as reader:
            config_dict = json.load(reader)
        for key, value in config_dict.items():
            if not hasattr(self, key):
                continue
            self._set_attribute(key, value)
        return self


@dataclass
class GPT2Args(Args):
    max_seq_len: int
    attn_pdrop: int = None
    embd_pdrop: int = None
    layer_norm_epsilon: float = None
    n_ctx: int = None
    n_embd: int = None
    n_head: int = None
    n_layer: int = None
    n_positions: int = None
    resid_pdrop: float = None
    vocab_size: int = None
    activation_function: str = None


@dataclass
class LlamaArgs(Args):
    max_seq_len: int
    local_rank: int
    world_size: int

    dim: int = None
    n_layers: int = None
    n_heads: int = None
    vocab_size: int = None  # defined later by tokenizer
    multiple_of: int = None  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = None

    # for 70b
    ffn_dim_multiplier: float = None
    n_kv_heads: int = None


@dataclass
class LoraLlamaArgs(LlamaArgs):
    r: int = None  # Rank of lora


@dataclass
class MistralArgs(Args):
    max_seq_len: int
    local_rank: int
    world_size: int

    dim: int = None
    n_layers: int = None
    head_dim: int = None
    hidden_dim: int = None
    n_heads: int = None
    n_kv_heads: int = None
    norm_eps: float = None
    vocab_size: int = None

    # For rotary embeddings. If not set, will be infered from sliding window.
    rope_theta: float = None
    # If this is set, use sliding window attention rotating cache.
    sliding_window: int = None
    # If this is set, we will use MoE layers instead of dense layers.
    moe = None


@dataclass
class OpenChatArgs(MistralArgs):
    pass


@dataclass
class LoraMistralArgs(MistralArgs):
    r: int = None
