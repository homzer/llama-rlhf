from src.tokenizers.tokenizer_llama import LlamaTokenizer


class MistralTokenizer(LlamaTokenizer):
    def __init__(self, model_file: str):
        super().__init__(model_file)
