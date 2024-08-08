from fairscale.nn.model_parallel.layers import VocabParallelEmbedding

from src.checkpoint import CheckpointForLlama3
from src.models import Llama, LoraLlama
from src.models.modeling_args import LlamaArgs, LoraLlamaArgs


class Llama3(Llama):
    def __init__(self, args: LlamaArgs):
        super().__init__(args)
        self.checkpoint = CheckpointForLlama3()

    def init_weights(self):
        super().init_weights()
        self.tok_embeddings = VocabParallelEmbedding(
            self.args.vocab_size, self.args.dim, init_method=lambda x: x
        ).type(self.args.dtype)


class LoraLlama3(LoraLlama):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.checkpoint = CheckpointForLlama3()

    def init_weights(self):
        super().init_weights()
        self.tok_embeddings = VocabParallelEmbedding(
            self.args.vocab_size, self.args.dim, init_method=lambda x: x
        ).type(self.args.dtype)

        self._freeze()
