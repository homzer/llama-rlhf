from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding
)

from src.modeling.modeling_acts import RMSNorm
from src.modeling.llama_abstract_hf import (
    AbstractAttentionHF,
    AbstractFeedForwardHF,
    AbstractTransformerBlockHF,
    AbstractBasicLLaMAHF,
    AbstractLlamaHF,
    LlamaRotaryEmbedding
)
from src.modeling.modeling_args import LlamaArgs


class AttentionHF(AbstractAttentionHF):
    def __init__(self, args: LlamaArgs):
        super().__init__(args)
        self.q_proj = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            )
        self.k_proj = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            )
        self.v_proj = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            )
        self.o_proj = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
            )
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)


class FeedForwardHF(AbstractFeedForwardHF):
    def __init__(self, args: LlamaArgs):
        super().__init__(args)
        self.gate_proj = ColumnParallelLinear(
            self.dim, self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.down_proj = RowParallelLinear(
            self.hidden_dim, self.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        )
        self.up_proj = ColumnParallelLinear(
            self.dim, self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )


class TransformerBlockHF(AbstractTransformerBlockHF):
    def __init__(self, layer_id: int, args: LlamaArgs):
        super().__init__(layer_id, args)
        self.self_attn = AttentionHF(args)
        self.mlp = FeedForwardHF(args)
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)


class BasicLLaMAHF(AbstractBasicLLaMAHF):
    def __init__(self, args: LlamaArgs):
        super().__init__(args)
        self.embed_tokens = ParallelEmbedding(
            args.vocab_size, args.dim, init_method=lambda x: x
        )

        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlockHF(layer_id, args))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)


class LlamaHF(AbstractLlamaHF):
    def __init__(self, args: LlamaArgs):
        super().__init__(args)
        self.model = BasicLLaMAHF(args)
        self.lm_head = ColumnParallelLinear(
            args.dim, args.vocab_size, bias=False, init_method=lambda x: x
        )
