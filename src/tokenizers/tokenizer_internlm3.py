from src.tokenizers.tokenizer_internlm import InternLMTokenizer


class InternLM3Tokenizer(InternLMTokenizer):
    def __init__(self, model_dir: str):
        super().__init__(model_dir=model_dir)
        self.eos_id = self.model.convert_tokens_to_ids("<|im_end|>")
