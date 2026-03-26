import torch
import torch.nn as nn
import torch.nn.functional as F

from src.checkpoint import CheckpointForQwen3Moe
from src.models.modeling import AutoModelForCausalLM
from src.models.modeling_args import QwenMoeArgs
from src.models.qwen3 import Qwen3TransformerBlock, Qwen3Head, Qwen3
from src.parallel.model_parallel.mappings import reduce_from_model_parallel_region


class Qwen3Expert(nn.Module):
    def __init__(self, args: QwenMoeArgs):
        super().__init__()
        self.args = args
        self.intermediate_size = args.moe_intermediate_size

        self.gate_proj = None
        self.down_proj = None
        self.up_proj = None
        self.gate_proj_fn = lambda x: self.gate_proj(x)
        self.down_proj_fn = lambda x: self.down_proj(x)
        self.up_proj_fn = lambda x: self.up_proj(x)

    def init_weights(self):
        self.gate_proj = nn.Linear(
            self.args.hidden_size, self.intermediate_size, bias=False,
        ).type(self.args.dtype)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.args.hidden_size, bias=False,
        ).type(self.args.dtype)
        self.up_proj = nn.Linear(
            self.args.hidden_size, self.intermediate_size, bias=False,
        ).type(self.args.dtype)

    def forward(self, x):
        return self.down_proj_fn(F.silu(self.gate_proj_fn(x)) * self.up_proj_fn(x))


class Qwen3MoeFeedForward(nn.Module):
    def __init__(self, args: QwenMoeArgs):
        super().__init__()
        self.args = args
        self.model_parallel_rank = args.model_parallel_rank
        self.model_parallel_world_size = args.model_parallel_world_size
        self.num_experts = args.num_experts
        assert self.num_experts % self.model_parallel_world_size == 0
        self.num_local_experts = self.num_experts // self.model_parallel_world_size
        self.expert_offset = self.model_parallel_rank * self.num_local_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = None
        self.experts = nn.ModuleList([Qwen3Expert(args) for _ in range(self.num_local_experts)])

    def init_weights(self):
        for expert in self.experts:
            expert.init_weights()
        self.gate = nn.Linear(self.args.hidden_size, self.args.num_experts, bias=False).type(self.args.dtype)

    def forward(self, x: torch.Tensor):
        bsz, seq_len, hidden_dim = x.shape
        x = x.view(-1, hidden_dim)
        router_logits = self.gate(x)

        routing_weights = torch.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)  # [b*s, k]
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x)

        final_hidden_states = torch.zeros((bsz * seq_len, hidden_dim), dtype=x.dtype, device=x.device)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)  # [n, k, b*s]
        for expert_idx in range(self.num_local_experts):
            expert = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx + self.expert_offset])
            current_state = x[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x))
        final_hidden_states = reduce_from_model_parallel_region(final_hidden_states)
        final_hidden_states = final_hidden_states.reshape(bsz, seq_len, hidden_dim)
        return final_hidden_states


class Qwen3MoeTransformerBlock(Qwen3TransformerBlock):
    def __init__(self, args: QwenMoeArgs):
        super().__init__(args=args)
        self.mlp = Qwen3MoeFeedForward(args)


class Qwen3MoeHead(Qwen3Head):
    def __init__(self, args: QwenMoeArgs):
        super().__init__(args=args)
        self.layers = torch.nn.ModuleList([Qwen3MoeTransformerBlock(args) for _ in range(args.num_hidden_layers)])


@AutoModelForCausalLM.register("qwen3-moe")
class Qwen3Moe(Qwen3):
    def __init__(self, args: QwenMoeArgs):
        super().__init__(args=args)
        self.model = Qwen3MoeHead(args)
        self.checkpoint = CheckpointForQwen3Moe(num_experts=args.num_experts)
