import collections

import torch

from src.criterion import MSELoss, KLDivLoss
from src.models.modeling import ParallelModelForCausalLM, ParallelVerifier
from src.ppo.buffer import RolloutBufferSample
from src.trainer import ParallelTrainer, Trainer
from src.utils import logits_assignment, powmax


class PPOTrainerForCausalLM(Trainer):
    def __init__(self, policy, optimizer: torch.optim.Optimizer):
        super().__init__(policy, optimizer)
        self.policy = policy
        self.optimizer = optimizer
        # TODO: schedule function
        self.clip_range = 0.07
        self.vf_coef = 0.1
        self.step = 0
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


class ParallelActorTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            actor: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            clip_range: float = 0.2,
            sft_coef: float = 0.0
    ):
        super().__init__(actor, optimizer)
        self.actor = actor
        self.clip_range = clip_range
        self.sft_coef = sft_coef
        self.step = 0
        self.sft_criterion = torch.nn.CrossEntropyLoss()

    def forward(self, rollout_data: RolloutBufferSample):
        self.actor.train()
        self.step += 1

        obs = rollout_data.observations.to(self.actor.device())
        actions = rollout_data.actions.to(self.actor.device())
        action_masks = rollout_data.action_masks.to(self.actor.device())
        advantages = rollout_data.advantages.to(self.actor.device())
        old_action_logprobs = rollout_data.old_action_logprobs.to(self.actor.device())

        outputs = self.actor.forward(obs)
        action_logprobs = torch.gather(
            torch.log_softmax(outputs.logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        # Normalize advantage
        advantages = torch.masked_select(advantages.view(-1), action_masks.view(-1))
        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(action_logprobs - old_action_logprobs)
        ratio = torch.masked_select(ratio.view(-1), action_masks.view(-1))
        # clipped surrogate loss
        actor_loss = advantages * ratio
        if self.clip_range > 0:
            clipped_actor_loss = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            actor_loss = torch.min(actor_loss, clipped_actor_loss)
        actor_loss = - torch.mean(actor_loss)

        # sft loss
        sft_loss = 0.0
        if self.sft_coef != 0.0:
            sft_target = actions.clone().detach().view(-1)
            sft_target[~ action_masks.view(-1)] = -100
            sft_loss = self.sft_coef * self.sft_criterion.forward(
                input=outputs.logits.view(-1, outputs.logits.shape[-1]),
                target=sft_target
            )

        loss = actor_loss + sft_loss

        kl_div = 0.0
        if rollout_data.ref_action_logprobs is not None:
            ref_action_logprobs = rollout_data.ref_action_logprobs.to(self.actor.device())
            # For logging only, compute kl divergence using mse loss
            kl_div = torch.masked_select(
                (0.5 * (action_logprobs.detach() - ref_action_logprobs) ** 2).view(-1), action_masks.view(-1)
            ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        Outputs = collections.namedtuple('Outputs', [
            'loss', "actor_loss", 'advantages', "kl", "sft_loss"])
        return Outputs(
            loss=loss.item(),
            actor_loss=actor_loss.item(),
            advantages=torch.mean(advantages).item(),
            kl=kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div,
            sft_loss=sft_loss.item() if isinstance(sft_loss, torch.Tensor) else sft_loss
        )


class ParallelCriticTrainerForCausalLM(ParallelTrainer):
    def __init__(self, critic: ParallelVerifier, optimizer: torch.optim.Optimizer):
        super().__init__(critic, optimizer)
        self.critic = critic
        self.step = 0
        self.criterion = MSELoss()

    def forward(self, rollout_data: RolloutBufferSample):
        self.critic.train()
        self.step += 1

        obs = rollout_data.observations.to(self.critic.device())
        action_masks = rollout_data.action_masks.to(self.critic.device())
        returns = rollout_data.returns.to(self.critic.device())

        values = self.critic.forward(obs).scores
        loss = self.criterion.forward(values, returns, action_masks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        Outputs = collections.namedtuple('Outputs', ['loss'])
        return Outputs(loss=loss.item())


class ParallelPolicyGradientTrainerForCausalLM(ParallelTrainer):
    def __init__(self, policy: ParallelModelForCausalLM, optimizer: torch.optim.Optimizer):
        super().__init__(policy, optimizer)
        self.policy = policy
        self.clip_range = 0.2
        self.step = 0

    def forward(self, rollout_data: RolloutBufferSample):
        self.policy.train()
        self.step += 1

        obs = rollout_data.observations.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())
        old_action_logprobs = rollout_data.old_action_logprobs.to(self.policy.device())

        outputs = self.policy.forward(obs)

        action_logprobs = torch.gather(
            torch.log_softmax(outputs.logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        ratio = torch.exp(action_logprobs - old_action_logprobs)
        ratio = torch.masked_select(ratio.view(-1), action_masks.view(-1))
        # clipped surrogate loss
        rewards = torch.masked_select(rewards.view(-1), action_masks.view(-1))
        actor_loss_1 = rewards * ratio
        actor_loss_2 = rewards * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        loss = - torch.min(actor_loss_1, actor_loss_2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        Outputs = collections.namedtuple('Outputs', ['loss', 'rewards'])
        return Outputs(loss=loss.item(), rewards=torch.mean(rewards).item())


class ParallelPolicyGradientTrainerWithCrossEntropyForCausalLM(ParallelTrainer):
    def __init__(self, policy: ParallelModelForCausalLM, optimizer: torch.optim.Optimizer):
        super().__init__(policy, optimizer)
        self.policy = policy
        self.step = 0

    def forward(self, rollout_data: RolloutBufferSample):
        self.policy.train()
        self.step += 1

        obs = rollout_data.observations.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())

        outputs = self.policy.forward(obs)

        action_logprobs = torch.gather(
            torch.log_softmax(outputs.logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        action_logprobs = torch.masked_select(action_logprobs.view(-1), action_masks.view(-1))
        rewards = torch.masked_select(rewards.view(-1), action_masks.view(-1))
        # weighted cross-entropy loss
        loss = - torch.mean(rewards * action_logprobs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        Outputs = collections.namedtuple('Outputs', ['loss', 'rewards'])
        return Outputs(loss=loss.item(), rewards=torch.mean(rewards).item())


class ParallelPolicyGradientTrainerWithKLDivForCausalLM(ParallelTrainer):
    def __init__(self, policy: ParallelModelForCausalLM, optimizer: torch.optim.Optimizer):
        super().__init__(policy, optimizer)
        self.policy = policy
        self.delta = 0.6
        self.step = 0
        self.criterion = KLDivLoss(return_scalar=False)

    def modified_kl_loss(self, logits, rewards, actions, action_masks) -> torch.Tensor:
        labels = logits.detach().clone()
        labels = torch.softmax(labels.float(), dim=-1).type_as(labels)
        labels = labels * logits_assignment(  # scaling
            torch.ones_like(labels), actions, (1 + torch.sign(rewards) * self.delta).to(labels)
        )
        labels = powmax(labels, dim=-1)

        loss = torch.masked_select(
            self.criterion.forward(logits, labels, targets_after_softmax=True).view(-1),
            action_masks.view(-1)
        )
        rewards = torch.masked_select(rewards.view(-1), action_masks.view(-1))
        loss = torch.mean(loss * torch.abs(rewards))
        return loss

    def ignore_negative_reward_loss(self, logits, rewards, actions, action_masks) -> torch.Tensor:
        scaling_coef = 3.0
        labels = logits.detach().clone()
        labels = torch.softmax(labels.float(), dim=-1).type_as(labels)
        labels = labels * logits_assignment(  # scaling
            torch.ones_like(labels), actions, scaling_coef
        )
        labels = powmax(labels, dim=-1)
        loss = torch.masked_select(
            self.criterion.forward(logits, labels, targets_after_softmax=True).view(-1),
            action_masks.view(-1)
        )
        threshold = 0.0
        beta = 0.2
        clamp_rewards = torch.masked_select(rewards.view(-1), action_masks.view(-1))
        clamp_rewards = torch.where(clamp_rewards > threshold, clamp_rewards, 0.0)
        clamp_rewards = torch.where(clamp_rewards < 0, (clamp_rewards - threshold) * beta, clamp_rewards)
        loss = torch.mean(torch.masked_select(clamp_rewards * loss, clamp_rewards != 0.0))
        return loss

    def forward(self, rollout_data: RolloutBufferSample):
        self.policy.train()
        self.step += 1

        obs = rollout_data.observations.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())

        outputs = self.policy.forward(obs)

        # loss = self.modified_kl_loss(outputs.logits, rewards, actions, action_masks)
        loss = self.ignore_negative_reward_loss(outputs.logits, rewards, actions, action_masks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        Outputs = collections.namedtuple('Outputs', ['loss', 'rewards'])
        return Outputs(loss=loss.item(), rewards=torch.mean(rewards).item())
