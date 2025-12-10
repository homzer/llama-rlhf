import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.checkpoint import CheckpointForMinistral3, CheckpointForMistral3
from src.models.modeling import (
    ParallelModelForCausalLM,
    CausalLMOutputs,
    AttentionForCausalLM,
    ParallelVerifier, VerifierOutputs)
from src.models.modeling_acts import Clamp, RMSNorm, LogitsNormalize, RotaryEmbedding
from src.models.modeling_args import Gemma3Args
from src.parallel.initialize import set_model_parallel_barrier
from src.parallel.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding
)
from src.parallel.sequence_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region
)
from src.utils import compute_position_ids, apply_rotary_pos_emb_


class Gemma3Attention(AttentionForCausalLM):
    def __init__(self, args: Gemma3Args):
        super().__init__(args.max_seq_len)
        self.args = args
        self.head_dim = args.text_config_head_dim
        self.num_local_heads = args.text_config_num_attention_heads // args.model_parallel_world_size
        self.num_key_value_heads = args.text_config_num_key_value_heads
        assert args.text_config_num_attention_heads % args.model_parallel_world_size == 0
        assert self.num_key_value_heads % args.model_parallel_world_size == 0
        self.num_local_key_value_heads = self.num_key_value_heads // args.model_parallel_world_size
        self.n_rep = args.text_config_num_attention_heads // args.text_config_num_key_value_heads

        self.rotary_emb = None

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None
        self.q_proj_fn = lambda x: self.q_proj(x)
        self.k_proj_fn = lambda x: self.k_proj(x)
        self.v_proj_fn = lambda x: self.v_proj(x)
        self.o_proj_fn = lambda x: self.o_proj(x)

    def init_weights(self):
        self.q_proj = ColumnParallelLinear(
            self.args.text_config_hidden_size,
            self.args.text_config_num_attention_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            ).type(self.args.dtype)
        self.k_proj = ColumnParallelLinear(
            self.args.text_config_hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.v_proj = ColumnParallelLinear(
            self.args.text_config_hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            ).type(self.args.dtype)
        self.o_proj = RowParallelLinear(
            self.args.text_config_num_attention_heads * self.head_dim,
            self.args.text_config_hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
            ).type(self.args.dtype)

        self.rotary_emb
