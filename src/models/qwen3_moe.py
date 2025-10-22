import torch
import torch.nn as nn

from src.models.modeling_args import QwenMoeArgs
from src.models.qwen import QwenFeedForward
from src.models.qwen3 import Qwen3TransformerBlock, Qwen3Head, Qwen3


class Qwen3MoeFeedForward(nn.Module):
    def __init__(self, args: QwenMoeArgs):
        super().__init__()
        self.args = args
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = None
        self.experts = nn.ModuleList(
            [QwenFeedForward(args, intermediate_size=args.moe_intermediate_size) for _ in range(self.num_experts)]
        )

    def init_weights(self):
        for expert in self.experts:
            expert.init_weights()
        self.gate = nn.Linear(self.args.hidden_size, self.args.num_experts, bias=False).type(self.args.dtype)

    def forward(self, x: torch.Tensor):
        bsz, seq_len, hidden_dim = x.shape
        x = x.view(-1, hidden_dim)
        router_logits = self.gate(x)

        routing_weights = torch.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x)

        final_hidden_states = torch.zeros((bsz * seq_len, hidden_dim), dtype=x.dtype, device=x.device)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = x[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x))
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


class Qwen3Moe(Qwen3):
    def __init__(self, args: QwenMoeArgs):
        super().__init__(args=args)
        self.model = Qwen3MoeHead(args)
