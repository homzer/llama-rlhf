import collections

import torch

from src.criterion import MSELoss
from src.models.modeling import ParallelModelForCausalLM, ParallelVerifier
from src.ppo.buffer import PolicyRolloutBufferSample
from src.trainer import ParallelTrainer, Trainer
from src.utils import log1m_softmax, proxy_neg_distribution


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

    def forward(self, rollout_data: PolicyRolloutBufferSample):
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
    ):
        super().__init__(actor, optimizer)
        self.actor = actor
        self.clip_range = clip_range
        self.step = 0

    def forward(self, rollout_data: PolicyRolloutBufferSample):
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
        loss = advantages * ratio
        if self.clip_range > 0:
            clipped_actor_loss = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            loss = torch.min(loss, clipped_actor_loss)
        loss = - torch.mean(loss)

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

        Outputs = collections.namedtuple('Outputs', ['loss', 'advantages', "kl"])
        return Outputs(
            loss=loss.item(),
            advantages=torch.mean(advantages).item(),
            kl=kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div,
        )


class ParallelActorTrainerWithSFTForCausalLM(ParallelTrainer):
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

    def forward(self, rollout_data: PolicyRolloutBufferSample):
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

    def forward(self, rollout_data: PolicyRolloutBufferSample):
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

    def forward(self, rollout_data: PolicyRolloutBufferSample):
        self.policy.train()
        self.step += 1

        obs = rollout_data.observations.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())

        logits = self.policy.forward(obs).logits
        rewards = rewards.to(logits.dtype)

        action_logprobs = torch.gather(
            torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        logging_log_probs = action_logprobs[0][action_masks[0]]
        logging_rewards = rewards[0][action_masks[0]]

        action_logprobs = torch.masked_select(action_logprobs.view(-1), action_masks.view(-1))
        rewards = torch.masked_select(rewards.view(-1), action_masks.view(-1))
        loss = - torch.mean(rewards * action_logprobs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        Outputs = collections.namedtuple('Outputs', ['loss', 'rewards', 'log_probs', 'token_rewards'])
        return Outputs(loss=loss.item(), rewards=torch.mean(rewards).item(), log_probs=logging_log_probs, token_rewards=logging_rewards)


class ParallelPolicyGradientStableTrainerForCausalLM(ParallelTrainer):
    def __init__(self, policy: ParallelModelForCausalLM, optimizer: torch.optim.Optimizer, neg_coef: float = 0.01):
        super().__init__(policy, optimizer)
        self.policy = policy
        self.neg_coef = neg_coef
        self.step = 0

    def forward(self, rollout_data: PolicyRolloutBufferSample):
        self.policy.train()
        self.step += 1

        obs = rollout_data.observations.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())

        outputs = self.policy.forward(obs)
        rewards = rewards.to(outputs.logits.dtype)

        action_logprobs = torch.gather(
            torch.log_softmax(outputs.logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)
        action_log1m_probs = torch.gather(
            log1m_softmax(outputs.logits), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        action_logprobs = torch.masked_select(action_logprobs.view(-1), action_masks.view(-1))
        action_log1m_probs = torch.masked_select(action_log1m_probs.view(-1), action_masks.view(-1))
        rewards = torch.masked_select(rewards.view(-1), action_masks.view(-1))
        # stabilize for negative rewards probs
        loss = torch.mean(rewards * torch.where(rewards < 0, self.neg_coef * action_log1m_probs, - action_logprobs))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        Outputs = collections.namedtuple('Outputs', ['loss', 'rewards'])
        return Outputs(loss=loss.item(), rewards=torch.mean(rewards).item())


class ParallelPolicyGradientConvexTrainerForCausalLM(ParallelTrainer):
    def __init__(self, policy: ParallelModelForCausalLM, optimizer: torch.optim.Optimizer, delta: float = 0.01):
        super().__init__(policy, optimizer)
        self.policy = policy
        self.step = 0
        self.delta = delta
        self.criterion = torch.nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, rollout_data: PolicyRolloutBufferSample):
        self.policy.train()
        self.step += 1

        obs = rollout_data.observations.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]
        actions = torch.masked_select(actions.view(-1), action_masks.view(-1))
        rewards = torch.masked_select(rewards.to(logits.dtype).view(-1), action_masks.view(-1))
        pos_reward_masks = rewards > 0

        # compute loss for positive reward tokens
        action_logprobs = torch.gather(
            torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)
        loss_pos = - rewards[pos_reward_masks] * action_logprobs[pos_reward_masks]

        # compute loss for negative reward tokens
        log_targets = proxy_neg_distribution(logits[~pos_reward_masks], actions[~pos_reward_masks], self.delta)
        loss_neg = - rewards[~pos_reward_masks] * self.criterion.forward(
            torch.log_softmax(logits[~pos_reward_masks], dim=-1), target=log_targets
        ).sum(-1)

        loss = torch.mean(torch.cat([loss_pos, loss_neg]))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step % 100 == 0:
            loss_pos_item = loss_pos.mean().nan_to_num(0).item()
            loss_neg_item = loss_neg.mean().nan_to_num(0).item()
            print(f"Positive Reward Loss: {loss_pos_item} | Negative Reward Loss: {loss_neg_item}")

        Outputs = collections.namedtuple('Outputs', ['loss', 'rewards'])
        return Outputs(loss=loss.item(), rewards=torch.mean(rewards).item())


class ParallelGRPOTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            model: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            clip_range: float = 0.2,
            kl_coef: float = 0.04
    ):
        super().__init__(model, optimizer)
        self.model = model
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.step = 0

    def forward(self, rollout_data: PolicyRolloutBufferSample):
        self.model.train()
        self.step += 1

        obs = rollout_data.observations.to(self.model.device())
        actions = rollout_data.actions.to(self.model.device())
        action_masks = rollout_data.action_masks.to(self.model.device())
        rewards = rollout_data.rewards.to(self.model.device())
        old_action_logprobs = rollout_data.old_action_logprobs.to(self.model.device())

        outputs = self.model.forward(obs)
        action_logprobs = torch.gather(
            torch.log_softmax(outputs.logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        # Normalize rewards
        rewards = torch.masked_select(rewards.view(-1), action_masks.view(-1))
        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(action_logprobs - old_action_logprobs)
        ratio = torch.masked_select(ratio.view(-1), action_masks.view(-1))
        # clipped surrogate loss
        policy_loss = rewards * ratio
        if self.clip_range > 0:
            clipped_actor_loss = rewards * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = torch.min(policy_loss, clipped_actor_loss)
        policy_loss = - torch.mean(policy_loss)

        kl_loss = 0.0
        if rollout_data.ref_action_logprobs is not None:
            ref_action_logprobs = rollout_data.ref_action_logprobs.to(self.model.device())
            probs_ratios = torch.exp(ref_action_logprobs - action_logprobs)
            kl_loss = self.kl_coef * torch.masked_select(
                (probs_ratios - (ref_action_logprobs - action_logprobs) - 1).view(-1),
                action_masks.view(-1)
            ).mean()

        loss = policy_loss + kl_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        Outputs = collections.namedtuple('Outputs', [
            'loss', "policy_loss", 'rewards', "kl_loss"])
        return Outputs(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            rewards=torch.mean(rewards).item(),
            kl_loss=kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
        )

