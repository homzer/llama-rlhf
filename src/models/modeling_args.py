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
        def unpack(v, k):
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    unpack(v_, "_".join([k, k_]))
            elif hasattr(self, k):
                self._set_attribute(k, v)

        with open(filename, 'r', encoding='utf-8') as reader:
            config_dict = json.load(reader)
        for key, value in config_dict.items():
            unpack(value, key)
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


class AutoArgs:
    _registry = {}

    @classmethod
    def register(cls, name):
        def wrapper(args_cls):
            cls._registry[name] = args_cls
            return args_cls
        return wrapper

    @classmethod
    def from_pretrained(
            cls,
            model_type: str,
            config_file: str,
            max_seq_len: int,
            dtype: str = 'bfloat16',
            lora_rank: int = -1,
            lora_dtype: str = 'bfloat16',
            use_logits_normalize: bool = True
    ) -> BaseParallelArgs:
        kwargs = dict(
            max_seq_len=max_seq_len,
            dtype=dtype,
            use_clamp=False,
            use_logits_normalize=use_logits_normalize
        )
        if lora_rank > 0:
            kwargs["r"] = lora_rank
            kwargs["lora_dtype"] = lora_dtype
            args = cls._registry[f"lora-{model_type}"](**kwargs).from_json(config_file)
        else:
            args = cls._registry[model_type](**kwargs).from_json(config_file)
        return args


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
@AutoArgs.register("llama")
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
@AutoArgs.register("llama3")
class Llama3Args(LlamaArgs):
    rope_theta: float = None


@dataclass
@AutoArgs.register("lora-llama")
class LoraLlamaArgs(LlamaArgs):
    r: int = None  # Rank of lora
    lora_dtype: str = "float32"


@dataclass
@AutoArgs.register("llama-hf")
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
@AutoArgs.register("lora-llama-hf")
class LoraLlamaArgsHf(LlamaArgsHf):
    r: int = None
    lora_dtype: str = "float32"


@dataclass
@AutoArgs.register("mistral")
class MistralArgs(BaseParallelArgs):
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
@AutoArgs.register("mistral3")
class Mistral3Args(BaseParallelArgs):
    text_config_hidden_size: int = None
    text_config_intermediate_size: int = None
    text_config_max_position_embeddings: int = None
    text_config_head_dim: int = None
    text_config_num_attention_heads: int = None
    text_config_num_hidden_layers: int = None
    text_config_num_key_value_heads: int = None
    text_config_rms_norm_eps: float = None
    text_config_vocab_size: int = None
    text_config_rope_parameters_beta_fast: float = None
    text_config_rope_parameters_beta_slow: float = None
    text_config_rope_parameters_factor: float = None
    text_config_rope_parameters_mscale: float = None
    text_config_rope_parameters_mscale_all_dim: float = None
    text_config_rope_parameters_rope_theta: float = None
    text_config_rope_parameters_llama_4_scaling_beta: float = None
    text_config_rope_parameters_original_max_position_embeddings: int = None

    def from_json(self, filename: str):
        if not filename.endswith(".json"):
            filename = os.path.join(filename, "config.json")
        return super().from_json(filename)


@dataclass
@AutoArgs.register("gemma3")
class Gemma3Args(BaseParallelArgs):
    text_config_hidden_size: int = None
    text_config_intermediate_size: int = None
    text_config_num_hidden_layers: int = None
    text_config_num_attention_heads: int = 8
    text_config_num_key_value_heads: int = 4
    text_config_head_dim: int = 256
    text_config_rope_scaling_factor: float = None

    def from_json(self, filename: str):
        if not filename.endswith(".json"):
            filename = os.path.join(filename, "config.json")
        return super().from_json(filename)


@dataclass
@AutoArgs.register("qwen")
@AutoArgs.register("qwen3")
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

    # For Qwen3
    head_dim: int = None

    def from_json(self, filename: str):
        if not filename.endswith(".json"):
            filename = os.path.join(filename, "config.json")
        return super().from_json(filename)


@dataclass
@AutoArgs.register("qwen-vl")
class QwenVLArgs(QwenArgs):
    image_token_id: int = None
    video_token_id: int = None
    vision_config_depth: int = 32
    vision_config_hidden_size: int = None
    vision_config_intermediate_size: int = None
    vision_config_num_heads: int = None
    vision_config_in_chans: int = None
    vision_config_out_hidden_size: int = None
    vision_config_patch_size: int = None
    vision_config_spatial_merge_size: int = None
    vision_config_spatial_patch_size: int = None
    vision_config_window_size: int = None
    vision_config_fullatt_block_indexes: list = None
    vision_config_tokens_per_second: int = None
    vision_config_temporal_patch_size: int = None
    rope_scaling_mrope_section: list = None


@dataclass
@AutoArgs.register("lora-qwen")
@AutoArgs.register("lora-qwen3")
class LoraQwenArgs(QwenArgs):
    r: int = None  # Rank of lora
    lora_dtype: str = "float32"


@dataclass
@AutoArgs.register("qwen-moe")
class QwenMoeArgs(QwenArgs):
    moe_intermediate_size: int = None
    norm_topk_prob: bool = True
    num_experts: int = None
    num_experts_per_tok: int = None


@dataclass
@AutoArgs.register("gemma2")
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
@AutoArgs.register("baichuan")
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
@AutoArgs.register("lora-baichuan")
class LoraBaichuanArgs(BaichuanArgs):
    r: int = None  # Rank of lora
    lora_dtype: str = "float32"


@dataclass
@AutoArgs.register("internlm")
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


@dataclass
@AutoArgs.register("internlm3")
class InternLM3Args(BaseParallelArgs):
    head_dim: int = None
    hidden_size: int = None
    intermediate_size: int = None
    max_position_embeddings: int = None
    num_attention_heads: int = None
    num_hidden_layers: int = None
    num_key_value_heads: int = None
    rms_norm_eps: float = None
    rope_scaling_factor: float = None
    rope_theta: float = None
    vocab_size: int = None

    def from_json(self, filename: str):
        if not filename.endswith(".json"):
            filename = os.path.join(filename, "config.json")
        return super().from_json(filename)


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
