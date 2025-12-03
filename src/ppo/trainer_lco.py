import collections

import torch

from src.models.modeling import ParallelModelForCausalLM
from src.trainer import ParallelTrainer
from src.utils import estimate_vocab_adv, create_lco_log_target


class ParallelLCOTrainerForRuleRM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            beta: float = 1.0,
            save_optim: bool = False,
            accumulation_steps: int = 1,
    ):
        super().__init__(policy, optimizer, save_optim=save_optim, accumulation_steps=accumulation_steps)
        self.policy = policy
        self.beta = beta
        self.criterion = torch.nn.KLDivLoss(reduction="none", log_target=True)

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        advantages = rollout_data.advantages.to(self.policy.device())
        advantage_indices = rollout_data.advantage_indices.to(self.policy.device())
        old_logits = rollout_data.logits.to(self.policy.device())

        advantages = advantages.view(-1, advantages.shape[-1])[action_masks.view(-1)]
        advantage_indices = advantage_indices.view(-1, advantage_indices.shape[-1])[action_masks.view(-1)]
        old_logits = old_logits.view(-1, old_logits.shape[-1])[action_masks.view(-1)]
        pos_advantage_masks = advantages > 0
        neg_advantage_masks = ~ pos_advantage_masks

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]

        advantages = estimate_vocab_adv(old_logits, advantages, advantage_indices)

        # compute loss for positive advantage tokens
        pos_log_targets = create_lco_log_target(
            old_logits[pos_advantage_masks], advantages[pos_advantage_masks], beta=self.beta
        )
        loss_pos = self.criterion.forward(
            torch.log_softmax(logits[pos_advantage_masks], dim=-1), target=pos_log_targets.to(logits)
        ).sum(-1).mean().nan_to_num(0.0)

        # compute loss for negative advantage tokens
        neg_log_targets = create_lco_log_target(
            old_logits[neg_advantage_masks], advantages[neg_advantage_masks], beta=self.beta
        )
        loss_neg = self.criterion.forward(
            torch.log_softmax(logits[neg_advantage_masks], dim=-1), target=neg_log_targets.to(logits)
        ).sum(-1).mean().nan_to_num(0.0)

        loss = loss_pos + loss_neg
        self.backward(loss)

        if (self.step + 1) % 100 == 0:
            print(f"Positive Reward Loss: {loss_pos.item()} | Negative Reward Loss: {loss_neg.item()}")

        Outputs = collections.namedtuple('Outputs', ['loss'])
        return Outputs(loss=loss.item())


class ParallelLCOTrainerForDPORM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            beta: float = 1.0,
            save_optim: bool = False,
            accumulation_steps: int = 1,
    ):
        super().__init__(policy, optimizer, save_optim=save_optim, accumulation_steps=accumulation_steps)
        self.policy = policy
        self.beta = beta
        self.criterion = torch.nn.KLDivLoss(reduction="none", log_target=True)

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        advantages = rollout_data.advantages.to(self.policy.device())
        advantage_indices = rollout_data.advantage_indices.to(self.policy.device())
        old_logits = rollout_data.logits.to(self.policy.device())

        advantages = advantages.view(-1, advantages.shape[-1])[action_masks.view(-1)]
        advantage_indices = advantage_indices.view(-1, advantage_indices.shape[-1])[action_masks.view(-1)]
        old_logits = old_logits.view(-1, old_logits.shape[-1])[action_masks.view(-1)]

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]

        advantages = (advantages / torch.std(advantages, dim=-1, keepdim=True)).nan_to_num(0.0)
        advantages_ = torch.full_like(old_logits, fill_value=-100)
        advantages_[torch.arange(advantages_.shape[0])[:, None], advantage_indices] = advantages

        log_targets = create_lco_log_target(old_logits, advantages_, beta=self.beta)
        loss = self.criterion.forward(
            torch.log_softmax(logits, dim=-1), target=log_targets.to(logits)
        ).sum(-1).mean().nan_to_num(0.0)
        self.backward(loss)

        Outputs = collections.namedtuple('Outputs', ['loss'])
        return Outputs(loss=loss.item())
