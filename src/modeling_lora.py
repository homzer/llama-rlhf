import torch.nn as nn
import torch.nn.init as init
from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding)

from src.modeling_abstract import (
    RMSNorm,
    AbstractLoraAttention,
    AbstractLoraFeedForward,
    AbstractLoraTransformerBlock,
    AbstractLoraLlama,
    AbstractLoraLlamaVerifier
)
from src.modeling_args import LoraModelArgs


class LoraAttention(AbstractLoraAttention):
    def __init__(self, args: LoraModelArgs):
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

        self.lora_a_wq = nn.Linear(
            args.dim,
            args.r,
            bias=False
        ).float()
        self.lora_b_wq = ColumnParallelLinear(
            args.r,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
            ).float()
        self.lora_a_wk = nn.Linear(
            args.dim,
            args.r,
            bias=False
        ).float()
        self.lora_b_wk = ColumnParallelLinear(
            args.r,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
            ).float()
        self.lora_a_wv = nn.Linear(
            args.dim,
            args.r,
            bias=False
        ).float()
        self.lora_b_wv = ColumnParallelLinear(
            args.r,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
            ).float()
        self.lora_a_wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
            ).float()
        self.lora_b_wo = nn.Linear(
            args.r,
            args.dim,
            bias=False
        ).float()
        init.zeros_(self.lora_b_wo.weight)


class LoraFeedForward(AbstractLoraFeedForward):
    def __init__(self, args: LoraModelArgs):
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

        self.lora_a_w1 = nn.Linear(
            self.dim,
            self.r,
            bias=False
        ).float()
        self.lora_b_w1 = ColumnParallelLinear(
            self.r,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()
        self.lora_a_w2 = RowParallelLinear(
            self.hidden_dim,
            self.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).float()
        self.lora_b_w2 = nn.Linear(
            self.r,
            self.dim,
            bias=False
        ).float()
        init.zeros_(self.lora_b_w2.weight)
        self.lora_a_w3 = nn.Linear(
            self.dim,
            self.r,
            bias=False
        ).float()
        self.lora_b_w3 = ColumnParallelLinear(
            self.r,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()


class LoraTransformerBlock(AbstractLoraTransformerBlock):
    def __init__(self, layer_id: int, args: LoraModelArgs):
        super().__init__(layer_id, args)
        self.attention = LoraAttention(args)
        self.feed_forward = LoraFeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)


class LoraLlama(AbstractLoraLlama):
    def __init__(self, args: LoraModelArgs):
        super().__init__(args)
        for layer_id in range(args.n_layers):
            self.layers.append(LoraTransformerBlock(layer_id, args))

        self.tok_embeddings = ParallelEmbedding(
            args.vocab_size, args.dim, init_method=lambda x: x
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(
            args.dim, args.vocab_size, bias=False, init_method=lambda x: x
        )

        self.lora_a_output = nn.Linear(
            args.dim,
            args.r,
            bias=False
        ).float()
        self.lora_b_output = ColumnParallelLinear(
            args.r,
            args.vocab_size,
            bias=False,
            gather_output=True,
            init_method=init.zeros_
        ).float()

        # Freeze parameters
        self._freeze()


class LoraLlamaVerifier(AbstractLoraLlamaVerifier):
    def __init__(self, args: LoraModelArgs):
        super().__init__(args)
        for layer_id in range(args.n_layers):
            self.layers.append(LoraTransformerBlock(layer_id, args))

        self.tok_embeddings = ParallelEmbedding(
            args.vocab_size, args.dim, init_method=lambda x: x
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.v_head = nn.Linear(args.dim, 1, bias=False).float()

        # Freeze parameters
        self._freeze()
