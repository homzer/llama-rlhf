from src.models import (
    ParallelModule,
    LoraMistral,
    LoraLlama,
    LoraLlamaVerifier,
    Mistral,
    MistralHf,
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
    LoraMistralHf,
    LlamaHf,
    LoraLlamaHf,
    InternLM
)
from src.models.modeling_args import (
    LlamaArgs,
    MistralArgs,
    LoraLlamaArgs,
    LoraMistralArgs,
    QwenArgs,
    MistralArgsHf,
    LoraMistralArgsHf,
    LoraQwenArgs,
    BaichuanArgs,
    LoraBaichuanArgs,
    LoraLlamaArgsHf,
    LlamaArgsHf,
    InternLMArgs
)
from src.tokenizers import (
    Tokenizer,
    LlamaTokenizer,
    Llama3Tokenizer,
    MistralTokenizer,
    QwenTokenizer,
    BaichuanTokenizer,
    LlamaTokenizerHf,
    InternLMTokenizer
)


ARGS = {
    "llama": LlamaArgs,
    "lora-llama": LoraLlamaArgs,
    "llama-hf": LlamaArgsHf,
    "lora-llama-hf": LoraLlamaArgsHf,
    "llama3": LlamaArgs,
    "lora-llama3": LoraLlamaArgs,
    "mistral": MistralArgs,
    "lora-mistral": LoraMistralArgs,
    "mistral-hf": MistralArgsHf,
    "lora-mistral-hf": LoraMistralArgsHf,
    "qwen": QwenArgs,
    "lora-qwen": LoraQwenArgs,
    "baichuan": BaichuanArgs,
    "lora-baichuan": LoraBaichuanArgs,
    "internlm": InternLMArgs
}


MODELS = {
    "llama": Llama,
    "lora-llama": LoraLlama,
    "llama3": Llama3,
    "lora-llama3": LoraLlama3,
    "llama-hf": LlamaHf,
    "lora-llama-hf": LoraLlamaHf,
    "mistral": Mistral,
    "lora-mistral": LoraMistral,
    "mistral-hf": MistralHf,
    "lora-mistral-hf": LoraMistralHf,
    "qwen": Qwen,
    "lora-qwen": LoraQwen,
    "baichuan": Baichuan,
    "lora-baichuan": LoraBaichuan,
    "internlm": InternLM
}

VERIFIERS = {
    "llama": LlamaVerifier,
    "lora-llama": LoraLlamaVerifier,
    "llama3": Llama3Verifier,
    "lora-llama3": LoraLlama3Verifier,
    "qwen": QwenVerifier,
    "lora-qwen": LoraQwenVerifier,
    "baichuan": BaichuanVerifier,
    "lora-baichuan": LoraBaichuanVerifier
}

TOKENIZERS = {
    "llama": LlamaTokenizer,
    "llama-hf": LlamaTokenizerHf,
    "llama3": Llama3Tokenizer,
    "mistral": MistralTokenizer,
    "mistral-hf": MistralTokenizer,
    "qwen": QwenTokenizer,
    "baichuan": BaichuanTokenizer,
    "internlm": InternLMTokenizer
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
        lora_rank: int,
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
