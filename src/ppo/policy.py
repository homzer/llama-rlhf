import collections
from typing import List

import torch
import torch.nn as nn

from src.modeling.modeling import ModelForCausalLM, Module, ParallelModule, ParallelModelForCausalLM
from src.ppo.generator import PPOGeneratorForCausalLM

PolicyForwardOutputs = collections.namedtuple(
    "PolicyForwardOutputs", ["obs", "actions", "values", "action_logits", "action_masks"]
)
PolicyEvaluateOutputs = collections.namedtuple(
    "PolicyEvaluateOutputs", ["values", "action_logits"]
)


class AbstractPolicyForCausalLM:
    """ Abstract Actor-Critic Policy """

    def forward(self, *args, **kwargs) -> PolicyForwardOutputs:
        raise NotImplementedError()

    def evaluate_actions(self, *args, **kwargs) -> PolicyEvaluateOutputs:
        raise NotImplementedError()

    def predict_values(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def predict_actions(self, *args, **kwargs):
        raise NotImplementedError()


class ActorCriticPolicyForCausalLM(Module, AbstractPolicyForCausalLM):
    def __init__(
            self,
            model: ModelForCausalLM,
            generator: PPOGeneratorForCausalLM,
            dim: int,
    ):
        super().__init__()
        self.model = model
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.value = nn.Linear(dim, 1).float()
        self.generator = generator

    def forward(self, obs: List[str]) -> PolicyForwardOutputs:
        outputs = self.generator.forward(obs)
        values = self.value.forward(self.ln(outputs.hidden_states)).squeeze(-1)
        # token shift left to get actions
        actions = torch.zeros_like(outputs.tokens)
        actions[:, :-1] = outputs.tokens[:, 1:]

        return PolicyForwardOutputs(
            obs=outputs.tokens,  # [b, s]
            actions=actions,  # [b, s]
            values=values,  # [b, s]
            action_logits=outputs.tokens_logits,
            action_masks=outputs.output_masks
        )

    def predict_values(self, obs) -> torch.Tensor:
        outputs = self.model.forward(obs)
        values = self.value.forward(self.ln(outputs.hidden_states)).squeeze(-1)
        return values

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> PolicyEvaluateOutputs:
        outputs = self.model.forward(obs)
        values = self.value.forward(self.ln(outputs.hidden_states)).squeeze(-1)
        actions_logits = torch.gather(
            outputs.logits,
            dim=-1,
            index=actions.unsqueeze(-1)
        ).squeeze(-1)

        return PolicyEvaluateOutputs(
            values=values,
            action_logits=actions_logits,
        )

    def predict_actions(self, prompts: List[str]) -> List[dict]:
        outputs = self.generator.forward(prompts)
        return outputs.tokens


class ParallelActorCriticPolicyForCausalLM(ParallelModule, AbstractPolicyForCausalLM):
    def __init__(
            self,
            local_rank: int,
            world_size: int,
            model: ParallelModelForCausalLM,
            generator: PPOGeneratorForCausalLM,
            dim: int,
    ):
        super().__init__(local_rank, world_size)
        self.model = model
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.value = nn.Linear(dim, 1).float()
        self.generator = generator

    def forward(self, obs: List[str]) -> PolicyForwardOutputs:
        outputs = self.generator.forward(obs)
        values = self.value.forward(self.ln(outputs.hidden_states)).squeeze(-1)
        # token shift left to get actions
        actions = torch.zeros_like(outputs.tokens)
        actions[:, :-1] = outputs.tokens[:, 1:]

        return PolicyForwardOutputs(
            obs=outputs.tokens,  # [b, s]
            actions=actions,  # [b, s]
            values=values,  # [b, s]
            action_logits=outputs.tokens_logits,
            action_masks=outputs.output_masks
        )

    def predict_values(self, obs) -> torch.Tensor:
        outputs = self.model.forward(obs)
        values = self.value.forward(self.ln(outputs.hidden_states)).squeeze(-1)
        return values

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> PolicyEvaluateOutputs:
        outputs = self.model.forward(obs)
        values = self.value.forward(self.ln(outputs.hidden_states)).squeeze(-1)
        actions_logits = torch.gather(
            outputs.logits,
            dim=-1,
            index=actions.unsqueeze(-1)
        ).squeeze(-1)

        return PolicyEvaluateOutputs(
            values=values,
            action_logits=actions_logits,
        )

    def predict_actions(self, prompts: List[str]) -> List[dict]:
        outputs = self.generator.forward(prompts)
        return outputs.tokens
