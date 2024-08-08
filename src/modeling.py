from src.models import (
    ParallelModule,
    LoraMistral,
    LoraLlama,
    LoraLlamaVerifier,
    Mistral,
    MistralHf,
    MistralMoeHf,
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
    LoraBaichuan
)
from src.models.modeling_args import (
    LlamaArgs,
    MistralArgs,
    LoraLlamaArgs,
    LoraMistralArgs,
    QwenArgs,
    MistralArgsHf,
    MistralMoeArgsHf,
    LoraQwenArgs,
    BaichuanArgs,
    LoraBaichuanArgs
)
from src.tokenizers import (
    Tokenizer,
    LlamaTokenizer,
    Llama3Tokenizer,
    MistralTokenizer,
    QwenTokenizer,
    BaichuanTokenizer
)


ARGS = {
    "llama": LlamaArgs,
    "lora-llama": LoraLlamaArgs,
    "llama3": LlamaArgs,
    "lora-llama3": LoraLlamaArgs,
    "mistral": MistralArgs,
    "lora-mistral": LoraMistralArgs,
    "mistral-hf": MistralArgsHf,
    "mistral-7b-instruct-v0.2": MistralArgsHf,
    "mixtral-8x7b-instruct-v0.1": MistralMoeArgsHf,
    "qwen": QwenArgs,
    "lora-qwen": LoraQwenArgs,
    "baichuan": BaichuanArgs,
    "lora-baichuan": LoraBaichuanArgs
}


MODELS = {
    "llama": Llama,
    "llama3": Llama3,

    "lora-llama": LoraLlama,
    "lora-llama3": LoraLlama3,

    "mistral": Mistral,
    "lora-mistral": LoraMistral,
    "mistral-hf": MistralHf,
    "mistral-7b-instruct-v0.2": MistralHf,
    "mixtral-8x7b-instruct-v0.1": MistralMoeHf,

    "qwen": Qwen,
    "lora-qwen": LoraQwen,

    "baichuan": Baichuan,
    "lora-baichuan": LoraBaichuan
}

VERIFIERS = {
    "llama": LlamaVerifier,
    "lora-llama": LoraLlamaVerifier,
    "qwen": QwenVerifier,
    "lora-qwen": LoraQwenVerifier,
    "baichuan": BaichuanVerifier,
    "lora-baichuan": LoraBaichuanVerifier
}

TOKENIZERS = {
    "llama": LlamaTokenizer,
    "llama3": Llama3Tokenizer,

    "mistral": MistralTokenizer,
    "mistral-hf": MistralTokenizer,
    "mistral-7b-instruct-v0.2": MistralTokenizer,
    "mixtral-8x7b-instruct-v0.1": MistralTokenizer,

    "qwen": QwenTokenizer,
    "baichuan": BaichuanTokenizer
}


def get_parallel_model(
        model_type: str,
        config_file: str,
        max_seq_len: int,
        tokenizer_file: str,
        lora_rank: int,
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
