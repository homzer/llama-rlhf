import collections

import torch

from src.models.modeling import ParallelModelForCausalLM, ParallelModule
from src.ppo.buffer import PPORolloutBufferSample
from src.trainer import ParallelTrainer
from src.utils import create_target_distribution, estimate_vocab_adv, create_lco_log_target


class ParallelLCOTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            beta: float = 10.0,
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
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        advantages = rollout_data.advantages.to(self.policy.device())
        old_logits = rollout_data.logits.to(self.policy.device())

        actions = actions.view(-1)[action_masks.view(-1)]
        advantages = advantages.view(-1)[action_masks.view(-1)]
        old_logits = old_logits.view(-1, old_logits.shape[-1])[action_masks.view(-1)]
        pos_advantage_masks = advantages > 0
        neg_advantage_masks = ~ pos_advantage_masks

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]

        advantages = estimate_vocab_adv(old_logits, advantages.unsqueeze(-1), actions.unsqueeze(-1))

        # compute loss for positive advantage tokens
        pos_log_targets = create_lco_log_target(
            old_logits[pos_advantage_masks], advantages[pos_advantage_masks], beta=0.1
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


class ParallelLogitsConvexTrainer(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModule,
            optimizer: torch.optim.Optimizer,
            rho_pos: float,
            rho_neg: float,
            min_rho_prob: float = 0.80,
            max_rho_prob: float = 0.99,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(policy, optimizer, save_optim, accumulation_steps=accumulation_steps)
        self.rho_pos = rho_pos
        self.rho_neg = rho_neg
        self.min_rho_prob = min_rho_prob
        self.max_rho_prob = max_rho_prob
        self.criterion = torch.nn.KLDivLoss(reduction="none", log_target=True)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_loss(self, logits, rewards, actions) -> torch.Tensor:
        pos_reward_masks = rewards > 0
        neg_reward_masks = ~ pos_reward_masks

        # compute loss for positive reward tokens
        loss_pos = torch.tensor(0.0).to(logits)
        if torch.sum(pos_reward_masks).item() != 0:
            pos_log_targets = create_target_distribution(
                logits=logits[pos_reward_masks],
                actions=actions[pos_reward_masks],
                rho=self.rho_pos,
                min_rho_prob=self.min_rho_prob,
                max_rho_prob=self.max_rho_prob
            )
            loss_pos = rewards[pos_reward_masks] * self.criterion.forward(
                torch.log_softmax(logits[pos_reward_masks], dim=-1), target=pos_log_targets
            ).sum(-1)
            loss_pos = torch.mean(loss_pos)

        # compute loss for negative reward tokens
        loss_neg = torch.tensor(0.0).to(logits)
        if torch.sum(neg_reward_masks).item() != 0:
            neg_log_targets = create_target_distribution(
                logits=logits[neg_reward_masks],
                actions=actions[neg_reward_masks],
                rho=self.rho_neg,
                min_rho_prob=self.min_rho_prob,
                max_rho_prob=self.max_rho_prob
            )
            loss_neg = - rewards[neg_reward_masks] * self.criterion.forward(
                torch.log_softmax(logits[neg_reward_masks], dim=-1), target=neg_log_targets
            ).sum(-1)
            loss_neg = torch.mean(loss_neg)

        loss = loss_pos + loss_neg

        if (self.step + 1) % 100 == 0:
            print(f"Positive Reward Loss: {loss_pos.item()} | Negative Reward Loss: {loss_neg.item()}")

        return loss


class ParallelPolicyGradientLogitsConvexTrainerForCausalLM(ParallelLogitsConvexTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            rho_pos: float = 1.2,
            rho_neg: float = 0.8,
            min_rho_prob: float = 0.80,
            max_rho_prob: float = 0.99,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(
            policy,
            optimizer,
            rho_pos=rho_pos,
            rho_neg=rho_neg,
            min_rho_prob=min_rho_prob,
            max_rho_prob=max_rho_prob,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps,
        )
        self.policy = policy

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())

        actions = torch.masked_select(actions.view(-1), action_masks.view(-1))
        rewards = torch.masked_select(rewards.view(-1), action_masks.view(-1))

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]
        rewards = rewards.to(logits.dtype)

        loss = self.compute_loss(logits, rewards, actions)
        self.backward(loss)

        Outputs = collections.namedtuple('Outputs', ['loss', 'rewards'])
        return Outputs(loss=loss.item(), rewards=torch.mean(rewards).item())


class ParallelPPOActorLogitsConvexTrainerForCausalLM(ParallelLogitsConvexTrainer):
    def __init__(
            self,
            actor: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            rho_pos: float = 1.2,
            rho_neg: float = 0.8,
            min_rho_prob: float = 0.80,
            max_rho_prob: float = 0.99,
            save_optim: bool = False,
            accumulation_steps: int = 1,
    ):
        super().__init__(
            actor,
            optimizer,
            rho_pos=rho_pos,
            rho_neg=rho_neg,
            min_rho_prob=min_rho_prob,
            max_rho_prob=max_rho_prob,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps,
        )
        self.actor = actor

    def forward(self, rollout_data: PPORolloutBufferSample):
        self.actor.train()

        obs = rollout_data.obs.to(self.actor.device())
        actions = rollout_data.actions.to(self.actor.device())
        action_masks = rollout_data.action_masks.to(self.actor.device())
        advantages = rollout_data.advantages.to(self.actor.device())

        actions = torch.masked_select(actions.view(-1), action_masks.view(-1))
        advantages = torch.masked_select(advantages.view(-1), action_masks.view(-1))

        logits = self.actor.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]
        advantages = advantages.to(logits.dtype)

        loss = self.compute_loss(logits, advantages, actions)
        self.backward(loss)

        Outputs = collections.namedtuple('Outputs', ['loss', 'advantages'])
        return Outputs(loss=loss.item(), advantages=torch.mean(advantages).item())


class ParallelGRPOLogitsConvexTrainerForCausalLM(ParallelLogitsConvexTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            rho_pos: float = 1.2,
            rho_neg: float = 0.8,
            min_rho_prob: float = 0.80,
            max_rho_prob: float = 0.99,
            kl_coef: float = 0.01,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(
            policy,
            optimizer,
            rho_pos=rho_pos,
            rho_neg=rho_neg,
            min_rho_prob=min_rho_prob,
            max_rho_prob=max_rho_prob,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )
        self.policy = policy
        self.kl_coef = kl_coef

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())

        actions = torch.masked_select(actions.view(-1), action_masks.view(-1))
        rewards = torch.masked_select(rewards.view(-1), action_masks.view(-1))

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]
        rewards = rewards.to(logits.dtype)

        policy_loss = self.compute_loss(logits, rewards, actions)

        kl_loss = 0.0
        if hasattr(rollout_data, "ref_action_logprobs"):
            action_logprobs = torch.gather(
                torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
            ).squeeze(-1)
            ref_action_logprobs = rollout_data.ref_action_logprobs.to(self.policy.device())
            ref_action_logprobs = torch.masked_select(ref_action_logprobs.view(-1), action_masks.view(-1))
            probs_ratios = torch.exp(ref_action_logprobs - action_logprobs)
            kl_loss = self.kl_coef * (probs_ratios - (ref_action_logprobs - action_logprobs) - 1).mean()

        loss = policy_loss + kl_loss
        self.backward(loss)

        Outputs = collections.namedtuple('Outputs', [
            'loss', "policy_loss", 'rewards', "kl_loss"])
        return Outputs(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            rewards=torch.mean(rewards).item(),
            kl_loss=kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
        )
