from src.parallel.model_parallel.layers import VocabParallelEmbedding, ColumnParallelLinear

from src.checkpoint import CheckpointForLlama3
from src.models.llama_hf import LlamaModelHf, LlamaHf
from src.models.modeling_args import LlamaArgsHf


class Llama3ModelHf(LlamaModelHf):
    def __init__(self, args: LlamaArgsHf):
        super().__init__(args)

    def init_weights(self):
        super().init_weights()
        self.embed_tokens = VocabParallelEmbedding(
            self.args.vocab_size, self.args.hidden_size, init_method=lambda x: x
        ).type(self.args.dtype)


class Llama3Hf(LlamaHf):
    def __init__(self, args: LlamaArgsHf):
        super().__init__(args)
        self.checkpoint = CheckpointForLlama3()
        self.model = Llama3ModelHf(args)

    def init_weights(self):
        super().init_weights()
        self.lm_head = ColumnParallelLinear(
            self.args.hidden_size, self.args.vocab_size, bias=False, init_method=lambda x: x
        ).type(self.args.dtype)
