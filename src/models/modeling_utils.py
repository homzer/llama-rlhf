from src.models import (
    ParallelModule,
    LoraLlama30B,
    LoraMistral,
    LoraLlama70B,
    LoraLlama,
    Mistral,
    MistralHf,
    MistralMoeHf,
    Llama30B,
    Llama70B,
    Llama,
    Qwen,
    Llama3,
    LoraLlama3
)
from src.models.modeling_args import (
    LlamaArgs,
    MistralArgs,
    LoraLlamaArgs,
    LoraMistralArgs,
    QwenArgsHf,
    MistralArgsHf,
    MistralMoeArgsHf
)
from src.tokenizers import (
    Tokenizer,
    LlamaTokenizer,
    LlamaChatTokenizer,
    Llama3Tokenizer,
    Llama3ChatTokenizer,
    MistralTokenizer,
    MistralChatTokenizer,
    QwenChatTokenizer,
    QwenTokenizer
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

    "mistral-7b": MistralArgs,
    "lora-mistral-7b": LoraMistralArgs,
    "mistral-7b-instruct-v0.2": MistralArgsHf,
    "mixtral-8x7b-instruct-v0.1": MistralMoeArgsHf,

    "qwen-7b": QwenArgsHf,
    "qwen-14b": QwenArgsHf,
    "qwen-72b": QwenArgsHf,
    "qwen-7b-chat": QwenArgsHf,
    "qwen-14b-chat": QwenArgsHf,
    "qwen-72b-chat": QwenArgsHf,
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

    "mistral-7b": Mistral,
    "lora-mistral-7b": LoraMistral,
    "mistral-7b-instruct-v0.2": MistralHf,
    "mixtral-8x7b-instruct-v0.1": MistralMoeHf,

    "qwen-7b": Qwen,
    "qwen-14b": Qwen,
    "qwen-72b": Qwen,
    "qwen-7b-chat": Qwen,
    "qwen-14b-chat": Qwen,
    "qwen-72b-chat": Qwen,
}

TOKENIZERS = {
    "llama-1-7b": LlamaTokenizer,
    "llama-1-13b": LlamaTokenizer,
    "llama-1-7b-chat": LlamaChatTokenizer,
    "llama-1-13b-chat": LlamaChatTokenizer,
    "llama-1-30b": LlamaTokenizer,
    "llama-2-7b": LlamaTokenizer,
    "llama-2-13b": LlamaTokenizer,
    "llama-2-70b": LlamaTokenizer,
    "llama-3-8b": Llama3Tokenizer,
    "llama-3-70b": Llama3Tokenizer,
    "llama-2-7b-chat": LlamaChatTokenizer,
    "llama-2-13b-chat": LlamaChatTokenizer,
    "llama-2-70b-chat": LlamaChatTokenizer,
    "llama-3-8b-instruct": Llama3ChatTokenizer,
    "llama-3-70b-instruct": Llama3ChatTokenizer,
    "lora-llama-1-7b": LlamaTokenizer,
    "lora-llama-1-13b": LlamaTokenizer,
    "lora-llama-2-7b": LlamaTokenizer,
    "lora-llama-2-13b": LlamaTokenizer,
    "lora-llama-1-30b": LlamaTokenizer,
    "lora-llama-2-70b": LlamaTokenizer,
    "lora-llama-3-8b": Llama3Tokenizer,
    "lora-llama-3-70b": Llama3Tokenizer,

    "mistral-7b": MistralTokenizer,
    "lora-mistral-7b": MistralTokenizer,
    "mistral-7b-instruct-v0.2": MistralChatTokenizer,
    "mixtral-8x7b-instruct-v0.1": MistralChatTokenizer,

    "qwen-7b": QwenTokenizer,
    "qwen-14b": QwenTokenizer,
    "qwen-72b": QwenTokenizer,
    "qwen-7b-chat": QwenChatTokenizer,
    "qwen-14b-chat": QwenChatTokenizer,
    "qwen-72b-chat": QwenChatTokenizer,
}


def get_parallel_model(
        model_type: str,
        config_file: str,
        local_rank: int,
        world_size: int,
        max_seq_len: int,
        tokenizer_file: str,
        lora_rank: int,
        dtype: str = 'float16',
        lora_dtype: str = 'float32',
        use_clamp: bool = False
) -> (ParallelModule, Tokenizer):
    if lora_rank > 0:
        model_type = "lora-" + model_type
        args = ARGS[model_type](
            max_seq_len=max_seq_len,
            local_rank=local_rank,
            world_size=world_size,
            dtype=dtype,
            r=lora_rank,
            lora_dtype=lora_dtype,
            use_clamp=use_clamp
        ).from_json(config_file)
    else:
        args = ARGS[model_type](
            max_seq_len=max_seq_len,
            local_rank=local_rank,
            world_size=world_size,
            dtype=dtype,
            use_clamp=use_clamp
        ).from_json(config_file)
    model = MODELS[model_type](args)
    tokenizer = TOKENIZERS[model_type](tokenizer_file)
    model.init_weights()
    return model, tokenizer
