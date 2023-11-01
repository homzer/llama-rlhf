from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding)

from src.modeling.llama_abstract import (
    AbstractAttention,
    AbstractFeedForward,
    AbstractTransformerBlock,
    AbstractLlama
)
from src.modeling.modeling_acts import RMSNorm
from src.modeling.modeling_args import LlamaArgs


class Attention(AbstractAttention):
    def __init__(self, args: LlamaArgs):
        super().__init__(args)

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
            )


class FeedForward(AbstractFeedForward):
    def __init__(self, args: LlamaArgs):
        super().__init__(args)

        self.w1 = ColumnParallelLinear(
            self.dim, self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            self.hidden_dim, self.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            self.dim, self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )


class TransformerBlock(AbstractTransformerBlock):
    def __init__(self, layer_id: int, args: LlamaArgs):
        super().__init__(layer_id, args)
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)


class Llama(AbstractLlama):
    def __init__(self, args: LlamaArgs):
        super().__init__(args)
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))

        self.tok_embeddings = ParallelEmbedding(
            args.vocab_size, args.dim, init_method=lambda x: x
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(
            args.dim, args.vocab_size, bias=False, init_method=lambda x: x
        )
