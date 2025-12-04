from src.models import (
    ParallelModule,
    LoraLlama,
    LoraLlamaVerifier,
    Mistral,
    Llama,
    LlamaVerifier,
    Qwen,
    QwenVerifier,
    Llama3,
    LoraLlama3,
    LoraQwen,
    LoraQwenVerifier,
    Baichuan,
    BaichuanVerifier,
    LoraBaichuanVerifier,
    LoraBaichuan,
    Llama3Verifier,
    LoraLlama3Verifier,
    LlamaHf,
    LoraLlamaHf,
    InternLM,
    Llama3Hf,
    Qwen3,
    Gemma2,
    InternLM3,
    Qwen3Moe,
    Mistral3,
    Mistral3Verifier
)
from src.models.modeling_args import (
    LlamaArgs,
    MistralArgs,
    LoraLlamaArgs,
    QwenArgs,
    LoraQwenArgs,
    BaichuanArgs,
    LoraBaichuanArgs,
    LoraLlamaArgsHf,
    LlamaArgsHf,
    InternLMArgs,
    Llama3Args,
    Gemma2Args,
    InternLM3Args,
    QwenMoeArgs,
    Mistral3Args
)
from src.tokenizers import (
    Tokenizer,
    LlamaTokenizer,
    Llama3Tokenizer,
    MistralTokenizer,
    QwenTokenizer,
    BaichuanTokenizer,
    LlamaTokenizerHf,
    InternLMTokenizer,
    InternLM3Tokenizer,
    Llama3TokenizerHf,
    GemmaTokenizer,
    Mistral3Tokenizer
)


ARGS = {
    "llama": LlamaArgs,
    "lora-llama": LoraLlamaArgs,
    "llama-hf": LlamaArgsHf,
    "lora-llama-hf": LoraLlamaArgsHf,
    "llama3": Llama3Args,
    "lora-llama3": LoraLlamaArgs,
    "llama3-hf": LlamaArgsHf,
    "mistral": MistralArgs,
    "qwen": QwenArgs,
    "qwen3": QwenArgs,
    "qwen3moe": QwenMoeArgs,
    "lora-qwen": LoraQwenArgs,
    "baichuan": BaichuanArgs,
    "lora-baichuan": LoraBaichuanArgs,
    "internlm": InternLMArgs,
    "gemma2": Gemma2Args,
    "internlm3": InternLM3Args,
    "mistral3": Mistral3Args
}


MODELS = {
    "llama": Llama,
    "lora-llama": LoraLlama,
    "llama3": Llama3,
    "lora-llama3": LoraLlama3,
    "llama3-hf": Llama3Hf,
    "llama-hf": LlamaHf,
    "lora-llama-hf": LoraLlamaHf,
    "mistral": Mistral,
    "qwen": Qwen,
    "qwen3": Qwen3,
    "qwen3moe": Qwen3Moe,
    "lora-qwen": LoraQwen,
    "baichuan": Baichuan,
    "lora-baichuan": LoraBaichuan,
    "internlm": InternLM,
    "gemma2": Gemma2,
    "internlm3": InternLM3,
    "mistral3": Mistral3
}

VERIFIERS = {
    "llama": LlamaVerifier,
    "lora-llama": LoraLlamaVerifier,
    "llama3": Llama3Verifier,
    "lora-llama3": LoraLlama3Verifier,
    "qwen": QwenVerifier,
    "lora-qwen": LoraQwenVerifier,
    "baichuan": BaichuanVerifier,
    "lora-baichuan": LoraBaichuanVerifier,
    "mistral3": Mistral3Verifier
}

TOKENIZERS = {
    "llama": LlamaTokenizer,
    "llama-hf": LlamaTokenizerHf,
    "llama3": Llama3Tokenizer,
    "llama3-hf": Llama3TokenizerHf,
    "mistral": MistralTokenizer,
    "qwen": QwenTokenizer,
    "qwen3": QwenTokenizer,
    "qwen3moe": QwenTokenizer,
    "baichuan": BaichuanTokenizer,
    "internlm": InternLMTokenizer,
    "internlm3": InternLM3Tokenizer,
    "gemma2": GemmaTokenizer,
    "mistral3": Mistral3Tokenizer
}


def get_parallel_model(
        model_type: str,
        config_file: str,
        max_seq_len: int,
        tokenizer_file: str,
        lora_rank: int = -1,
        dtype: str = 'bfloat16',  # float16 might be NaN
        lora_dtype: str = 'float32',
        use_clamp: bool = False,
        use_logits_normalize: bool = True
) -> (ParallelModule, Tokenizer):
    kwargs = dict(
        max_seq_len=max_seq_len,
        dtype=dtype,
        use_clamp=use_clamp,
        use_logits_normalize=use_logits_normalize
    )
    if lora_rank > 0:
        kwargs["r"] = lora_rank
        kwargs["lora_dtype"] = lora_dtype
        args = ARGS["lora-" + model_type](**kwargs).from_json(config_file)
        model = MODELS["lora-" + model_type](args)
    else:
        args = ARGS[model_type](**kwargs).from_json(config_file)
        model = MODELS[model_type](args)
    tokenizer = TOKENIZERS[model_type](tokenizer_file)
    model.init_weights()
    return model, tokenizer


def get_parallel_verifier(
        model_type: str,
        config_file: str,
        max_seq_len: int,
        tokenizer_file: str,
        lora_rank: int = -1,
        dtype: str = 'bfloat16',
        lora_dtype: str = 'float32',
        use_clamp: bool = False,
        use_logits_normalize: bool = True
) -> (ParallelModule, Tokenizer):
    kwargs = dict(
        max_seq_len=max_seq_len,
        dtype=dtype,
        use_clamp=use_clamp,
        use_logits_normalize=use_logits_normalize
    )
    if lora_rank > 0:
        kwargs["r"] = lora_rank
        kwargs["lora_dtype"] = lora_dtype
        args = ARGS["lora-" + model_type](**kwargs).from_json(config_file)
        model = VERIFIERS["lora-" + model_type](args)
    else:
        args = ARGS[model_type](**kwargs).from_json(config_file)
        model = VERIFIERS[model_type](args)
    tokenizer = TOKENIZERS[model_type](tokenizer_file)
    model.init_weights()
    return model, tokenizer
