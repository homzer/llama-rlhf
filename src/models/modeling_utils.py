from src.models import (
    ParallelModule,
    LoraLlama30B,
    LoraMistral,
    LoraLlama70B,
    LoraLlama,
    Mistral,
    MistralHf,
    Llama30B,
    Llama70B,
    Llama,
    Qwen
)
from src.models.modeling_args import LlamaArgs, MistralArgs, LoraLlamaArgs, LoraMistralArgs, QwenArgs, MistralArgsHf
from src.tokenizers import Tokenizer, MistralTokenizer, LlamaTokenizer, QwenChatTokenizer, QwenTokenizer
from src.tokenizers.tokenizer_mistral import MistralChatTokenizer


def get_parallel_model(
        model_type: str,
        config_file: str,
        local_rank: int,
        world_size: int,
        max_seq_len: int,
        tokenizer_file: str,
        lora_rank: int
) -> (ParallelModule, Tokenizer):
    if lora_rank > 0:
        if 'mistral' in model_type:
            params = LoraMistralArgs(
                max_seq_len=max_seq_len, local_rank=local_rank, world_size=world_size, r=lora_rank
            ).from_json(config_file)
            model = LoraMistral(params)
            tokenizer = MistralTokenizer(tokenizer_file)
        else:
            params = LoraLlamaArgs(
                max_seq_len=max_seq_len, local_rank=local_rank, world_size=world_size, r=lora_rank
            ).from_json(config_file)
            if '30' in model_type:
                model = LoraLlama30B(params)
            elif '70' in model_type:
                model = LoraLlama70B(params)
            else:
                model = LoraLlama(params)
            tokenizer = LlamaTokenizer(tokenizer_file)
    else:
        if 'mistral' in model_type:
            if 'instruct' in model_type:
                params = MistralArgsHf(
                    max_seq_len=max_seq_len, local_rank=local_rank, world_size=world_size
                ).from_json(config_file)
                model = MistralHf(params)
                tokenizer = MistralChatTokenizer(tokenizer_file)
            else:
                params = MistralArgs(
                    max_seq_len=max_seq_len, local_rank=local_rank, world_size=world_size
                ).from_json(config_file)
                model = Mistral(params)
                tokenizer = MistralTokenizer(tokenizer_file)
        elif 'qwen' in model_type:
            params = QwenArgs(
                max_seq_len=max_seq_len, local_rank=local_rank, world_size=world_size
            ).from_json(config_file)
            model = Qwen(params)
            tokenizer = QwenChatTokenizer(tokenizer_file) if 'chat' in model_type else QwenTokenizer(tokenizer_file)
        else:
            params = LlamaArgs(
                max_seq_len=max_seq_len, local_rank=local_rank, world_size=world_size
            ).from_json(config_file)
            if '30' in model_type:
                model = Llama30B(params)
            elif '70' in model_type:
                model = Llama70B(params)
            else:
                model = Llama(params)
            tokenizer = LlamaTokenizer(tokenizer_file)

    # parameter post initialization
    model.init_weights()
    return model, tokenizer
