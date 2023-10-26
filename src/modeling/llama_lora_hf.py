import torch.nn as nn
import torch.nn.init as init
from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding
)

from src.modeling.llama_abstract_hf import (
    AbstractLoraAttentionHF,
    AbstractLoraFeedForwardHF,
    AbstractLoraTransformerBlockHF,
    AbstractLoraLlamaHF,
    AbstractLoraBasicLLaMAHF,
    LlamaRotaryEmbedding
)
from src.modeling.modeling import RMSNorm
from src.modeling.modeling_args import LoraLlamaArgs


class LoraAttentionHF(AbstractLoraAttentionHF):
    def __init__(self, args: LoraLlamaArgs):
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

        self.lora_a_q_proj = nn.Linear(
            args.dim,
            args.r,
            bias=False
        ).float()
        self.lora_b_q_proj = ColumnParallelLinear(
            args.r,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
            ).float()
        self.lora_a_k_proj = nn.Linear(
            args.dim,
            args.r,
            bias=False
        ).float()
        self.lora_b_k_proj = ColumnParallelLinear(
            args.r,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
            ).float()
        self.lora_a_v_proj = nn.Linear(
            args.dim,
            args.r,
            bias=False
        ).float()
        self.lora_b_v_proj = ColumnParallelLinear(
            args.r,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
            ).float()
        self.lora_a_o_proj = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
            ).float()
        self.lora_b_o_proj = nn.Linear(
            args.r,
            args.dim,
            bias=False
        ).float()
        init.zeros_(self.lora_b_wo.weight)

        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)


class LoraFeedForwardHF(AbstractLoraFeedForwardHF):
    def __init__(self, args: LoraLlamaArgs):
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

        self.lora_a_gate_proj = nn.Linear(
            self.dim,
            self.r,
            bias=False
        ).float()
        self.lora_b_gate_proj = ColumnParallelLinear(
            self.r,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()
        self.lora_a_down_proj = RowParallelLinear(
            self.hidden_dim,
            self.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).float()
        self.lora_b_down_proj = nn.Linear(
            self.r,
            self.dim,
            bias=False
        ).float()
        init.zeros_(self.lora_b_w2.weight)
        self.lora_a_up_proj = nn.Linear(
            self.dim,
            self.r,
            bias=False
        ).float()
        self.lora_b_up_proj = ColumnParallelLinear(
            self.r,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()


class LoraTransformerBlockHF(AbstractLoraTransformerBlockHF):
    def __init__(self, layer_id: int, args: LoraLlamaArgs):
        super().__init__(layer_id, args)
        self.self_attn = LoraAttentionHF(args)
        self.mlp = LoraFeedForwardHF(args)
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)


class LoraBasicLLaMA(AbstractLoraBasicLLaMAHF):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        for layer_id in range(args.n_layers):
            self.layers.append(LoraTransformerBlockHF(layer_id, args))

        self.embed_tokens = ParallelEmbedding(
            args.vocab_size, args.dim, init_method=lambda x: x
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)


class LoraLlamaHF(AbstractLoraLlamaHF):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.model = LoraBasicLLaMA(args)
        self.lm_head = ColumnParallelLinear(
            args.dim, args.vocab_size, bias=False, init_method=lambda x: x
        )

        self.lora_a_lm_head = nn.Linear(
            args.dim,
            args.r,
            bias=False
        ).float()
        self.lora_b_lm_head = ColumnParallelLinear(
            args.r,
            args.vocab_size,
            bias=False,
            gather_output=True,
            init_method=init.zeros_
        ).float()

        # Freeze parameters
        self._freeze()
