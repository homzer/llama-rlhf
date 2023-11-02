import collections

import torch

from src.criterion import MSELoss
from src.ppo.buffer import RolloutBufferSample
from src.ppo.policy import ActorCriticPolicyForCausalLM


class PPOTrainerForCausalLM:
    def __init__(
            self,
            policy: ActorCriticPolicyForCausalLM,
            optimizer: torch.optim.Optimizer,
    ):
        self.policy = policy
        # TODO: schedule function
        self.clip_range = 0.07
        self.vf_coef = 0.1
        self.step = 0
        self.optimizer = optimizer
        self.criterion = MSELoss()

    def forward(self, rollout_data: RolloutBufferSample):
        self.policy.train()
        self.step += 1

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

        Outputs = collections.namedtuple('Outputs', ['loss', 'policy_loss', 'value_loss'])
        return Outputs(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item()
        )

    def load(self, ckpt_dir: str):
        self.policy.load(ckpt_dir)

    def save(self, save_dir: str):
        self.policy.save(save_dir)
