from src.models.llama_70b import Llama70B, LoraLlama70B
from src.models.modeling_args import LlamaArgs, LoraLlamaArgs


class Llama3(Llama70B):
    def __init__(self, args: LlamaArgs):
        super().__init__(args)


class LoraLlama3(LoraLlama70B):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
