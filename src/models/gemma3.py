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
from src.models.modeling_acts import Clamp, RMSNorm, LogitsNormalize
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
