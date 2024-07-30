from src.models import (
    ParallelModule,
    LoraLlama30B,
    LoraMistral,
    LoraLlama70B,
    LoraLlama,
    LoraLlamaVerifier,
    Mistral,
    MistralHf,
    MistralMoeHf,
    Llama30B,
    Llama70B,
    Llama,
    LlamaVerifier,
    Qwen,
    QwenVerifier,
    Llama3,
    LoraLlama3, LoraQwen, LoraQwenVerifier, Baichuan, BaichuanVerifier, LoraBaichuanVerifier, LoraBaichuan
)
from src.models.modeling_args import (
    LlamaArgs,
    MistralArgs,
    LoraLlamaArgs,
    LoraMistralArgs,
    QwenArgs,
    MistralArgsHf,
    MistralMoeArgsHf, LoraQwenArgs, BaichuanArgs, LoraBaichuanArgs
)
from src.tokenizers import (
    Tokenizer,
    LlamaTokenizer,
    Llama3Tokenizer,
    Llama3ChatTokenizer,
    MistralTokenizer,
    QwenTokenizer, BaichuanTokenizer
)


ARGS = {
    "llama-1-7b": LlamaArgs,
    "llama-1-13b": LlamaArgs,
    "llama-1-7b-chat": LlamaArgs,
    "llama-1-13b-chat": LlamaArgs,
    "llama-1-30b": LlamaArgs,
    "llama-2-7b": LlamaArgs,
    "llama-2-13b": LlamaArgs,
    "llama-2-70b": LlamaArgs,
    "llama-3-8b": LlamaArgs,
    "llama-3-70b": LlamaArgs,
    "llama-2-7b-chat": LlamaArgs,
    "llama-2-13b-chat": LlamaArgs,
    "llama-2-70b-chat": LlamaArgs,
    "llama-3-8b-instruct": LlamaArgs,
    "llama-3-70b-instruct": LlamaArgs,

    "lora-llama-1-7b": LoraLlamaArgs,
    "lora-llama-1-13b": LoraLlamaArgs,
    "lora-llama-2-7b": LoraLlamaArgs,
    "lora-llama-2-13b": LoraLlamaArgs,
    "lora-llama-1-30b": LoraLlamaArgs,
    "lora-llama-2-70b": LoraLlamaArgs,
    "lora-llama-3-8b": LoraLlamaArgs,
    "lora-llama-3-70b": LoraLlamaArgs,

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
    "llama-1-7b": Llama,
    "llama-1-13b": Llama,
    "llama-1-7b-chat": Llama,
    "llama-1-13b-chat": Llama,
    "llama-1-30b": Llama30B,
    "llama-2-7b": Llama,
    "llama-2-13b": Llama,
    "llama-2-70b": Llama70B,
    "llama-3-8b": Llama3,
    "llama-3-70b": Llama3,
    "llama-2-7b-chat": Llama,
    "llama-2-13b-chat": Llama,
    "llama-2-70b-chat": Llama70B,
    "llama-3-8b-instruct": Llama3,
    "llama-3-70b-instruct": Llama3,
    "lora-llama-1-7b": LoraLlama,
    "lora-llama-1-13b": LoraLlama,
    "lora-llama-2-7b": LoraLlama,
    "lora-llama-2-13b": LoraLlama,
    "lora-llama-1-30b": LoraLlama30B,
    "lora-llama-2-70b": LoraLlama70B,
    "lora-llama-3-8b": LoraLlama3,
    "lora-llama-3-70b": LoraLlama3,

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
    "llama-2-7b": LlamaVerifier,
    "lora-llama-2-7b": LoraLlamaVerifier,
    "qwen": QwenVerifier,
    "lora-qwen": LoraQwenVerifier,
    "baichuan": BaichuanVerifier,
    "lora-baichuan": LoraBaichuanVerifier
}

TOKENIZERS = {
    "llama-1-7b": LlamaTokenizer,
    "llama-1-13b": LlamaTokenizer,
    "llama-1-30b": LlamaTokenizer,
    "llama-2-7b": LlamaTokenizer,
    "llama-2-13b": LlamaTokenizer,
    "llama-2-70b": LlamaTokenizer,
    "llama-3-8b": Llama3Tokenizer,
    "llama-3-70b": Llama3Tokenizer,
    "llama-3-8b-instruct": Llama3ChatTokenizer,
    "llama-3-70b-instruct": Llama3ChatTokenizer,

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
    if lora_rank > 0:
        args = ARGS["lora-" + model_type](
            max_seq_len=max_seq_len,
            dtype=dtype,
            r=lora_rank,
            lora_dtype=lora_dtype,
            use_clamp=use_clamp,
            use_logits_normalize=use_logits_normalize
        ).from_json(config_file)
        model = MODELS["lora-" + model_type](args)
    else:
        args = ARGS[model_type](
            max_seq_len=max_seq_len,
            dtype=dtype,
            use_clamp=use_clamp,
            use_logits_normalize=use_logits_normalize
        ).from_json(config_file)
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
        use_clamp: bool = False
) -> (ParallelModule, Tokenizer):
    if lora_rank > 0:
        args = ARGS["lora-" + model_type](
            max_seq_len=max_seq_len,
            dtype=dtype,
            r=lora_rank,
            lora_dtype=lora_dtype,
            use_clamp=use_clamp
        ).from_json(config_file)
        model = VERIFIERS["lora-" + model_type](args)
    else:
        args = ARGS[model_type](
            max_seq_len=max_seq_len,
            dtype=dtype,
            use_clamp=use_clamp
        ).from_json(config_file)
        model = VERIFIERS[model_type](args)
    tokenizer = TOKENIZERS[model_type](tokenizer_file)
    model.init_weights()
    return model, tokenizer
