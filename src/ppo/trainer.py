import collections

import numpy as np
import torch

from src.criterion import MSELoss
from src.ppo.buffer import RolloutBuffer
from src.ppo.policy import ActorCriticPolicyForCausalLM


class PPOTrainerForCausalLM:
    def __init__(
            self,
            policy: ActorCriticPolicyForCausalLM,
            optimizer: torch.optim.Optimizer,
            batch_size: int,
            inner_epochs: int = 4,
    ):
        self.policy = policy
        self.batch_size = batch_size
        self.clip_range = 0.07  # TODO: schedule function
        self.vf_coef = 0.1
        self.lr = 1e-5
        self.step = 0
        self.inner_epochs = inner_epochs
        self.optimizer = optimizer
        self.criterion = MSELoss()

    def train(self, rollout_buffer: RolloutBuffer):
        self.policy.train()
        self.step += 1

        losses = []
        value_losses = []
        policy_losses = []
        for epoch in range(self.inner_epochs):
            for rollout_data in rollout_buffer.get(self.batch_size):
                obs = rollout_data.observations.to(self.policy.device())
                actions = rollout_data.actions.to(self.policy.device())
                action_masks = rollout_data.action_masks.to(self.policy.device())
                advantages = rollout_data.advantages.to(self.policy.device())
                old_action_logits = rollout_data.old_action_logits.to(self.policy.device())
                returns = rollout_data.returns.to(self.policy.device())

                outputs = self.policy.evaluate_actions(obs=obs, actions=actions)

                # Normalize advantage
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages = torch.masked_select(advantages.view(-1), action_masks.view(-1))
                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(outputs.action_logits - old_action_logits)
                ratio = torch.masked_select(ratio.view(-1), action_masks.view(-1))
                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss using the TD(Temporal Difference)(gae_lambda) target
                # Regression training for value function (or critic)
                value_loss = self.criterion.forward(outputs.values, returns, action_masks)

                loss = policy_loss + self.vf_coef * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())

        Outputs = collections.namedtuple('Outputs', ['loss', 'policy_loss', 'value_loss'])
        return Outputs(
            loss=np.mean(losses),
            policy_loss=np.mean(policy_losses),
            value_loss=np.mean(value_losses)
        )

    def load(self, ckpt_dir: str):
        self.policy.load(ckpt_dir)

    def save(self, save_dir: str):
        self.policy.save(save_dir)
