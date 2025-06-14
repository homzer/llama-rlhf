import json
import os
from dataclasses import dataclass

import transformers
from src.parallel.initialize import (
    get_model_parallel_world_size,
    get_model_parallel_rank,
    get_model_parallel_src_rank,
    get_data_parallel_world_size,
    get_data_parallel_rank,
)

from src.utils import get_torch_dtype
from src.parallel.initialize import get_data_parallel_src_rank

HF_CONFIG_MAP = {
    "hidden_size": "dim",
    "num_attention_heads": "n_heads",
    "num_hidden_layers": "n_layers",
    "rms_norm_eps": "norm_eps",
}


class Args:
    def __post_init__(self):
        dtype = getattr(self, 'dtype', None)
        if dtype is not None:
            setattr(self, 'dtype', get_torch_dtype(dtype))
        lora_dtype = getattr(self, 'lora_dtype', None)
        if lora_dtype is not None:
            setattr(self, 'lora_dtype', get_torch_dtype(lora_dtype))
        if getattr(self, 'global_rank', None) is None:
            setattr(self, 'global_rank', int(os.environ.get("RANK")))
        if getattr(self, 'local_rank', None) is None:
            setattr(self, 'local_rank', int(os.environ.get("LOCAL_RANK")))
        if getattr(self, 'world_size', None) is None:
            setattr(self, 'world_size', int(os.environ.get("WORLD_SIZE")))
        if getattr(self, 'model_parallel_world_size', None) is None:
            setattr(self, 'model_parallel_world_size', get_model_parallel_world_size())
        if getattr(self, 'model_parallel_rank', None) is None:
            setattr(self, 'model_parallel_rank', get_model_parallel_rank())
        if getattr(self, 'model_parallel_src_rank', None) is None:
            setattr(self, 'model_parallel_src_rank', get_model_parallel_src_rank())
        if getattr(self, 'data_parallel_world_size', None) is None:
            setattr(self, 'data_parallel_world_size', get_data_parallel_world_size())
        if getattr(self, 'data_parallel_rank', None) is None:
            setattr(self, 'data_parallel_rank', get_data_parallel_rank())
        if getattr(self, 'data_parallel_src_rank', None) is None:
            setattr(self, 'data_parallel_src_rank', get_data_parallel_src_rank())

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
class BaseArgs(Args):
    max_seq_len: int
    use_clamp: bool = False
    use_logits_normalize: bool = True


@dataclass
class BaseParallelArgs(Args):
    max_seq_len: int

    global_rank: int = None
    local_rank: int = None
    world_size: int = None
    model_parallel_world_size: int = None
    model_parallel_rank: int = None
    model_parallel_src_rank: int = None
    data_parallel_world_size: int = None
    data_parallel_rank: int = None
    data_parallel_src_rank: int = None

    dtype: str = "float16"

    use_clamp: bool = False
    use_logits_normalize: bool = True


@dataclass
class GPT2Args(BaseArgs):
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
class LlamaArgs(BaseParallelArgs):
    dim: int = None
    n_layers: int = None
    n_heads: int = None
    vocab_size: int = None  # defined later by tokenizer
    multiple_of: int = None  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = None
    rope_theta: float = 10000.0

    # for 70b
    ffn_dim_multiplier: float = None
    n_kv_heads: int = None

    def from_json(self, filename: str):
        if not filename.endswith(".json"):
            filename = os.path.join(filename, "params.json")
        return super().from_json(filename)


@dataclass
class Llama3Args(LlamaArgs):
    rope_theta: float = None


@dataclass
class LoraLlamaArgs(LlamaArgs):
    r: int = None  # Rank of lora
    lora_dtype: str = "float32"


@dataclass
class LlamaArgsHf(BaseParallelArgs):
    hidden_size: int = None
    num_hidden_layers: int = None
    intermediate_size: int = None
    max_position_embeddings: int = None
    num_attention_heads: int = None
    num_key_value_heads: int = None
    rms_norm_eps: float = None
    vocab_size: int = None

    def from_json(self, filename: str):
        if not filename.endswith(".json"):
            filename = os.path.join(filename, "config.json")
        return super().from_json(filename)


@dataclass
class LoraLlamaArgsHf(LlamaArgsHf):
    r: int = None
    lora_dtype: str = "float32"


@dataclass
class MistralArgs(BaseParallelArgs):
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

    def from_json(self, filename: str):
        if not filename.endswith(".json"):
            filename = os.path.join(filename, "params.json")
        return super().from_json(filename)


@dataclass
class MistralArgsHf(BaseParallelArgs):
    hidden_size: int = None
    num_hidden_layers: int = None
    intermediate_size: int = None
    max_position_embeddings: int = None
    num_attention_heads: int = None
    num_key_value_heads: int = None
    rms_norm_eps: float = None
    vocab_size: int = None

    # For rotary embeddings. If not set, will be infered from sliding window.
    rope_theta: int = None
    # If this is set, use sliding window attention rotating cache.
    sliding_window: int = None
    # If this is set, we will use MoE layers instead of dense layers.
    moe = None

    def from_json(self, filename: str):
        if not filename.endswith(".json"):
            filename = os.path.join(filename, "config.json")
        return super().from_json(filename)


@dataclass
class LoraMistralArgs(MistralArgs):
    r: int = None
    lora_dtype: str = "float32"


@dataclass
class LoraMistralArgsHf(MistralArgsHf):
    r: int = None
    lora_dtype: str = "float32"


@dataclass
class QwenArgs(BaseParallelArgs):
    hidden_size: int = None
    intermediate_size: int = None
    max_position_embeddings: int = None
    max_window_layers: int = None
    num_attention_heads: int = None
    num_hidden_layers: int = None
    num_key_value_heads: int = None
    rms_norm_eps: float = None
    rope_theta: int = None
    sliding_window: int = None
    tie_word_embeddings: bool = None
    use_sliding_window: bool = None
    vocab_size: int = None

    def from_json(self, filename: str):
        if not filename.endswith(".json"):
            filename = os.path.join(filename, "config.json")
        return super().from_json(filename)


@dataclass
class LoraQwenArgs(QwenArgs):
    r: int = None  # Rank of lora
    lora_dtype: str = "float32"


@dataclass
class Gemma2Args(BaseParallelArgs):
    attn_logit_softcapping: float = None
    final_logit_softcapping: float = None
    head_dim: int = None
    hidden_size: int = None
    intermediate_size: int = None
    max_position_embeddings: int = None
    num_attention_heads: int = None
    num_hidden_layers: int = None
    num_key_value_heads: int = None
    query_pre_attn_scalar: int = None
    rms_norm_eps: float = None
    rope_theta: float = None
    vocab_size: int = None

    def from_json(self, filename: str):
        if not filename.endswith(".json"):
            filename = os.path.join(filename, "config.json")
        return super().from_json(filename)


@dataclass
class BaichuanArgs(BaseParallelArgs):
    vocab_size: int = None
    hidden_size: int = None
    intermediate_size: int = None
    num_hidden_layers: int = None
    num_attention_heads: int = None
    hidden_act: str = None
    max_position_embeddings: int = None
    initializer_range: int = None
    rms_norm_eps: float = None
    use_cache: bool = None

    def from_json(self, filename: str):
        if not filename.endswith(".json"):
            filename = os.path.join(filename, "config.json")
        return super().from_json(filename)


@dataclass
class LoraBaichuanArgs(BaichuanArgs):
    r: int = None  # Rank of lora
    lora_dtype: str = "float32"


@dataclass
class InternLMArgs(BaseParallelArgs):
    hidden_size: int = None
    intermediate_size: int = None
    max_position_embeddings: int = None
    max_window_layers: int = None
    num_attention_heads: int = None
    num_hidden_layers: int = None
    num_key_value_heads: int = None
    rms_norm_eps: float = None
    rope_theta: int = None
    sliding_window: int = None
    tie_word_embeddings: bool = None
    use_sliding_window: bool = None
    vocab_size: int = None

    def from_json(self, filename: str):
        if not filename.endswith(".json"):
            filename = os.path.join(filename, "config.json")
        return super().from_json(filename)


class MistralMoeArgsHf(MistralArgsHf):
    num_local_experts: int
    num_experts_per_tok: int


class T5Config(transformers.T5Config):
    def __init__(self, max_input_len=128, max_output_len=384, **kwargs):
        super().__init__(**kwargs)
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len


class LoraT5Config(T5Config):
    def __init__(self, r=16, **kwargs):
        super().__init__(**kwargs)
        self.r = r
        self.lora_dtype = "float32"
