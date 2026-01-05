import collections

import torch

from src.criterion import MSELoss, DPOLoss
from src.models.modeling import ParallelModelForCausalLM, ParallelVerifier
from src.ppo.buffer import PPORolloutBufferSample
from src.trainer import ParallelTrainer, Trainer
from src.utils import masked_mean


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

    def forward(self, rollout_data: PPORolloutBufferSample):
        self.policy.train()
        self.step += 1

        obs = rollout_data.obs.to(self.policy.device())
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


class ParallelPPOTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            clip_range: float = 0.2,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(
            model=policy,
            optimizer=optimizer,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )
        self.policy = policy
        self.clip_range = clip_range

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        advantages = rollout_data.advantages.to(self.policy.device())
        old_action_logprobs = rollout_data.action_logprobs.to(self.policy.device())

        actions = actions.view(-1)[action_masks.view(-1)]
        advantages = advantages.view(-1)[action_masks.view(-1)]
        old_action_logprobs = old_action_logprobs.view(-1)[action_masks.view(-1)]

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]
        action_logprobs = torch.gather(
            torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        ratio = torch.exp(action_logprobs - old_action_logprobs)
        # clipped surrogate loss
        loss = advantages * ratio
        if self.clip_range > 0:
            clipped_loss = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            loss = torch.min(loss, clipped_loss)
        loss = - torch.mean(loss)

        self.backward(loss)

        Outputs = collections.namedtuple('Outputs', ['loss', 'advantages'])
        return Outputs(
            loss=loss.item(),
            advantages=torch.mean(advantages).item(),
        )


class ParallelPPOActorTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            actor: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            clip_range: float = 0.2,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(actor, optimizer, save_optim, accumulation_steps=accumulation_steps)
        self.actor = actor
        self.clip_range = clip_range

    def forward(self, rollout_data: PPORolloutBufferSample):
        self.actor.train()

        obs = rollout_data.obs.to(self.actor.device())
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

        self.backward(loss)

        Outputs = collections.namedtuple('Outputs', ['loss', 'advantages', "kl"])
        return Outputs(
            loss=loss.item(),
            advantages=torch.mean(advantages).item(),
            kl=kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div,
        )


class ParallelPPOCriticTrainerForCausalLM(ParallelTrainer):
    def __init__(self, critic: ParallelVerifier, optimizer: torch.optim.Optimizer, accumulation_steps: int = 1):
        super().__init__(critic, optimizer, accumulation_steps=accumulation_steps)
        self.critic = critic
        self.criterion = MSELoss()

    def forward(self, rollout_data: PPORolloutBufferSample):
        self.critic.train()

        obs = rollout_data.obs.to(self.critic.device())
        action_masks = rollout_data.action_masks.to(self.critic.device())
        returns = rollout_data.returns.to(self.critic.device())

        values = self.critic.forward(obs).scores
        loss = self.criterion.forward(values, returns, action_masks)

        self.backward(loss)

        Outputs = collections.namedtuple('Outputs', ['loss'])
        return Outputs(loss=loss.item())


class ParallelPolicyGradientTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(
            model=policy,
            optimizer=optimizer,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )
        self.policy = policy

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        advantages = rollout_data.advantages.to(self.policy.device())

        actions = actions.view(-1)[action_masks.view(-1)]
        advantages = advantages.view(-1)[action_masks.view(-1)]

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]
        action_logprobs = torch.gather(
            torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        loss = - torch.mean(advantages * action_logprobs)

        self.backward(loss)

        Outputs = collections.namedtuple('Outputs', ['loss', 'advantages'])
        return Outputs(loss=loss.item(), advantages=torch.mean(advantages).item())


class ParallelPolicyGradientGuiderTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(policy, optimizer, save_optim, accumulation_steps=accumulation_steps)
        self.policy = policy

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device()).reshape(-1)
        action_masks = rollout_data.action_masks.to(self.policy.device()).reshape(-1)
        guider_actions = rollout_data.guider_actions.to(self.policy.device()).reshape(-1)
        rewards = rollout_data.rewards.to(self.policy.device()).reshape(-1)

        logits = self.policy.forward(obs).logits
        rewards = rewards.to(logits.dtype)

        logits = torch.reshape(logits, [-1, logits.shape[-1]])[action_masks]
        actions = actions[action_masks]
        guider_actions = guider_actions[action_masks]
        rewards = rewards[action_masks]
        pos_reward_masks = rewards >= 0
        guider_action_masks = (guider_actions != actions) & (rewards < 0)

        if (torch.any(guider_action_masks) or torch.any(pos_reward_masks)).item() is False:  # rare case
            print("Warning: all action masks are False.")
            guider_action_masks[-1] = True

        pos_action_logprobs = torch.gather(
            torch.log_softmax(logits[pos_reward_masks], dim=-1), dim=-1, index=actions[pos_reward_masks].unsqueeze(-1)
        ).squeeze(-1)

        guider_action_logprobs = torch.gather(
            torch.log_softmax(logits[guider_action_masks], dim=-1), dim=-1, index=actions[guider_action_masks].unsqueeze(-1)
        ).squeeze(-1)

        pos_action_loss, guider_action_loss = 0, 0
        if len(pos_action_logprobs) != 0:
            pos_action_loss = - torch.mean(rewards[pos_reward_masks] * pos_action_logprobs)
        if len(guider_action_logprobs) != 0:
            guider_action_loss = torch.mean(rewards[guider_action_masks] * guider_action_logprobs)
        loss = (pos_action_loss + guider_action_loss) * 0.5

        self.backward(loss)

        Outputs = collections.namedtuple('Outputs', ['loss', 'pos_action_loss', 'guider_action_loss'])
        return Outputs(
            loss=loss.item(),
            pos_action_loss=pos_action_loss.item() if isinstance(pos_action_loss, torch.Tensor) else pos_action_loss,
            guider_action_loss=guider_action_loss.item() if isinstance(guider_action_loss, torch.Tensor) else guider_action_loss
        )


class ParallelInvalidActionMaskTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            clip_range: float = 0.2,
            save_optim: bool = False,
            accumulation_steps: int = 1,
    ):
        super().__init__(
            model=policy,
            optimizer=optimizer,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )
        self.policy = policy
        self.clip_range = clip_range

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        old_action_logprobs = rollout_data.action_logprobs.to(self.policy.device())
        valid_action_logits = rollout_data.logits.to(self.policy.device())

        rewards = rewards.view(-1)[action_masks.view(-1)]
        actions = actions.view(-1)[action_masks.view(-1)]
        old_action_logprobs = old_action_logprobs.view(-1)[action_masks.view(-1)]
        valid_action_logits = valid_action_logits.view(-1, valid_action_logits.shape[-1])[action_masks.view(-1)]
        valid_action_masks = torch.softmax(valid_action_logits, dim=-1) >= 0.1
        valid_action_masks[torch.arange(valid_action_masks.shape[0]), actions] = True

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]
        logits = torch.where(valid_action_masks, logits, -100)

        action_logprobs = torch.gather(
            torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(action_logprobs - old_action_logprobs)
        # clipped surrogate loss
        loss = rewards * ratio
        if self.clip_range > 0:
            clipped_actor_loss = rewards * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            loss = torch.min(loss, clipped_actor_loss)
        loss = - torch.mean(loss)

        self.backward(loss)

        Outputs = collections.namedtuple('Outputs', ['loss'])
        return Outputs(loss=loss.item())


class ParallelDPOTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            beta: float = 0.1,
            save_optim: bool = False,
            accumulation_steps: int = 1,
    ):
        super().__init__(
            model=policy,
            optimizer=optimizer,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )
        self.policy = policy
        self.criterion = DPOLoss(beta=beta, reduction="sum")

    def forward(self, rollout_data):
        self.policy.train()

        chosen_obs = rollout_data.chosen_obs.to(self.policy.device())
        rejected_obs = rollout_data.rejected_obs.to(self.policy.device())
        chosen_actions = rollout_data.chosen_actions.to(self.policy.device())
        rejected_actions = rollout_data.rejected_actions.to(self.policy.device())
        chosen_action_masks = rollout_data.chosen_action_masks.to(self.policy.device())
        rejected_action_masks = rollout_data.rejected_action_masks.to(self.policy.device())
        ref_chosen_logprobs = rollout_data.ref_chosen_logprobs.to(self.policy.device())
        ref_rejected_logprobs = rollout_data.ref_rejected_logprobs.to(self.policy.device())

        chosen_logits = self.policy.forward(chosen_obs).logits
        chosen_action_logprobs = torch.gather(
            torch.log_softmax(chosen_logits, dim=-1), dim=-1, index=chosen_actions.unsqueeze(-1)
        ).squeeze(-1)

        rejected_logits = self.policy.forward(rejected_obs).logits
        rejected_action_logprobs = torch.gather(
            torch.log_softmax(rejected_logits, dim=-1), dim=-1, index=rejected_actions.unsqueeze(-1)
        ).squeeze(-1)

        loss = self.criterion.forward(
            chosen_logprobs=chosen_action_logprobs,
            rejected_logprobs=rejected_action_logprobs,
            ref_chosen_logprobs=ref_chosen_logprobs,
            ref_rejected_logprobs=ref_rejected_logprobs,
            chosen_masks=chosen_action_masks,
            rejected_masks=rejected_action_masks
        )
        self.backward(loss)

        Outputs = collections.namedtuple('Outputs', ['loss'])
        return Outputs(loss=loss.item())


class ParallelMiniLLMTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            alpha: float = 0.2,
            clip_range: float = 0.2,
            save_optim: bool = False,
            accumulation_steps: int = 1,
    ):
        super().__init__(
            model=policy,
            optimizer=optimizer,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )
        self.alpha = alpha
        self.clip_range = clip_range
        self.policy = policy
        self.criterion = torch.nn.KLDivLoss(reduction="none", log_target=True)

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        old_action_logprobs = rollout_data.old_action_logprobs.to(self.policy.device())
        teacher_logits = rollout_data.logits.to(self.policy.device())
        teacher_action_logprobs = rollout_data.action_logprobs.to(self.policy.device())

        old_action_logprobs[~action_masks] = 0.0
        teacher_action_logprobs[~action_masks] = 0.0
        mix_action_logprobs = torch.log(
            self.alpha * teacher_action_logprobs.exp() + (1 - self.alpha) * old_action_logprobs.exp()
        )
        normed_rewards = torch.cumsum((teacher_action_logprobs - old_action_logprobs).flip(-1), dim=-1).flip(-1)
        normed_rewards = normed_rewards / torch.cumsum(action_masks.flip(-1), dim=-1).flip(-1)

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]
        teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])[action_masks.view(-1)]
        actions = actions.view(-1)[action_masks.view(-1)]
        old_action_logprobs = old_action_logprobs.view(-1)[action_masks.view(-1)]
        mix_action_logprobs = mix_action_logprobs.view(-1)[action_masks.view(-1)]
        normed_rewards = normed_rewards.view(-1)[action_masks.view(-1)]
        action_logprobs = torch.gather(
            torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        loss_single = (old_action_logprobs - mix_action_logprobs).exp() * self.criterion.forward(
            torch.log_softmax(teacher_logits, dim=-1), target=torch.log_softmax(logits, dim=-1)
        ).sum(-1)

        ratio = torch.exp(action_logprobs - mix_action_logprobs)
        loss_long = normed_rewards * ratio
        loss_long = torch.min(loss_long, normed_rewards * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range))

        loss = (loss_single - loss_long).mean()
        self.backward(loss)
        Outputs = collections.namedtuple('Outputs', ['loss'])
        return Outputs(loss=loss.item())


class ParallelGKDTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            beta: float = 0.9,
            save_optim: bool = False,
            accumulation_steps: int = 1,
    ):
        super().__init__(
            model=policy,
            optimizer=optimizer,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )
        self.beta = beta
        self.policy = policy
        self.criterion = torch.nn.KLDivLoss(reduction="none", log_target=True)

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        teacher_logits = rollout_data.logits.to(self.policy.device())

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]
        teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])[action_masks.view(-1)]
        log_inputs = (self.beta * torch.softmax(logits, dim=-1) + (1 - self.beta) * torch.softmax(logits, dim=-1)).log()
        log_inputs = torch.nan_to_num(log_inputs, neginf=-100)
        loss = self.beta * self.criterion.forward(
            log_inputs, target=torch.log_softmax(teacher_logits.to(logits), dim=-1)
        ).sum(-1)
        loss += (1 - self.beta) * self.criterion.forward(
            log_inputs, target=torch.log_softmax(logits, dim=-1)
        ).sum(-1)
        loss = loss.mean()

        self.backward(loss)
        Outputs = collections.namedtuple('Outputs', ['loss'])
        return Outputs(loss=loss.item())


class ParallelWeightedPolicyGradientTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            pos_weight: float = 0.1,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(policy, optimizer, save_optim, accumulation_steps=accumulation_steps)
        self.policy = policy
        self.pos_weight = pos_weight

    def forward(self, rollout_data: PPORolloutBufferSample):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]
        actions = torch.masked_select(actions.view(-1), action_masks.view(-1))
        rewards = torch.masked_select(rewards.to(logits.dtype).view(-1), action_masks.view(-1))
        pos_reward_masks = rewards > 0

        loss_pos = - self.pos_weight * rewards[pos_reward_masks] * torch.gather(
            torch.log_softmax(logits[pos_reward_masks], dim=-1), dim=-1, index=actions[pos_reward_masks].unsqueeze(-1)
        ).squeeze(-1)

        # compute loss for negative reward tokens
        loss_neg = - rewards[~pos_reward_masks] * torch.gather(
            torch.log_softmax(logits[~pos_reward_masks], dim=-1), dim=-1, index=actions[~pos_reward_masks].unsqueeze(-1)
        ).squeeze(-1)

        loss = torch.mean(torch.cat([loss_pos, loss_neg]))

        self.backward(loss)

        if self.step % 100 == 0:
            loss_pos_item = loss_pos.mean().nan_to_num(0).item()
            loss_neg_item = loss_neg.mean().nan_to_num(0).item()
            print(f"Positive Reward Loss: {loss_pos_item} | Negative Reward Loss: {loss_neg_item}")

        Outputs = collections.namedtuple('Outputs', ['loss', 'rewards'])
        return Outputs(loss=loss.item(), rewards=torch.mean(rewards).item())


class ParallelOREALTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            beta: float = 0.01,
            save_optim: bool = False,
            accumulation_steps: int = 1,
    ):
        super().__init__(policy, optimizer, save_optim, accumulation_steps=accumulation_steps)
        self.policy = policy
        self.beta = beta

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())
        old_action_logprobs = rollout_data.action_logprobs.to(self.policy.device())
        ref_action_logprobs = rollout_data.ref_action_logprobs.to(self.policy.device())

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]
        actions = torch.masked_select(actions.view(-1), action_masks.view(-1))
        action_logprobs = torch.gather(
            torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)
        rewards = torch.masked_select(rewards.to(logits.dtype).view(-1), action_masks.view(-1))
        old_action_logprobs = torch.masked_select(old_action_logprobs.view(-1), action_masks.view(-1))
        ref_action_logprobs = torch.masked_select(ref_action_logprobs.view(-1), action_masks.view(-1))

        pos_reward_masks = rewards > 0
        pos_loss = - action_logprobs[pos_reward_masks]
        neg_loss = action_logprobs[~pos_reward_masks] - old_action_logprobs[~pos_reward_masks]
        loss = torch.mean(torch.cat([pos_loss, neg_loss]))

        # compute kl penalty
        kl_loss = self.beta * torch.abs(action_logprobs - ref_action_logprobs).mean()

        self.backward(loss + kl_loss)

        Outputs = collections.namedtuple('Outputs', ['loss', 'kl_loss'])
        return Outputs(loss=loss.item(), kl_loss=kl_loss.item())


class ParallelNFTTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            epsilon: float = 1.0,
            save_optim: bool = False,
            accumulation_steps: int = 1,
    ):
        super().__init__(policy, optimizer, save_optim, accumulation_steps=accumulation_steps)
        self.policy = policy
        self.epsilon = epsilon

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())
        group_rewards = rollout_data.group_rewards.to(self.policy.device())  # [b, s]
        old_action_logprobs = rollout_data.action_logprobs.to(self.policy.device())

        logits = self.policy.forward(obs).logits
        action_logprobs = torch.gather(
            torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)
        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(action_logprobs - old_action_logprobs)
        ratio = torch.masked_select(ratio.view(-1), action_masks.view(-1))

        rewards = torch.masked_select(rewards.to(logits.dtype).view(-1), action_masks.view(-1))
        group_rewards = torch.masked_select(group_rewards.to(logits.dtype).view(-1), action_masks.view(-1))
        pos_reward_masks = rewards > 0
        pos_ratio = ratio[pos_reward_masks]
        neg_ratio = ratio[~pos_reward_masks]
        neg_group_rewards = group_rewards[~pos_reward_masks]
        neg_ratio = (1 - neg_group_rewards * neg_ratio) / (1 - neg_ratio.float() - 1e-12)
        neg_ratio = (torch.where(neg_ratio >= self.epsilon, neg_ratio, self.epsilon) - neg_ratio).detach() + neg_ratio

        loss = - torch.mean(torch.log(torch.cat([pos_ratio, neg_ratio])))

        self.backward(loss)

        Outputs = collections.namedtuple('Outputs', ['loss', 'rewards'])
        return Outputs(loss=loss.item(), rewards=torch.mean(rewards).item())


class ParallelDAPOTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            clip_range_higher: float = 0.28,
            clip_range_lower: float = 0.20,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(policy, optimizer, save_optim, accumulation_steps=accumulation_steps)
        self.policy = policy
        self.clip_range_higher = clip_range_higher
        self.clip_range_lower = clip_range_lower

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())
        old_action_logprobs = rollout_data.action_logprobs.to(self.policy.device())

        actions = torch.masked_select(actions.view(-1), action_masks.view(-1))
        rewards = torch.masked_select(rewards.view(-1), action_masks.view(-1))
        old_action_logprobs = torch.masked_select(old_action_logprobs.view(-1), action_masks.view(-1))

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]
        rewards = rewards.to(logits.dtype)
        action_logprobs = torch.gather(
            torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        ratio = torch.exp(action_logprobs - old_action_logprobs)
        policy_loss = rewards * ratio
        clipped_actor_loss = rewards * torch.clamp(ratio, 1 - self.clip_range_lower, 1 + self.clip_range_higher)
        policy_loss = torch.min(policy_loss, clipped_actor_loss)
        policy_loss = - torch.sum(policy_loss)  # Token-level policy gradient loss in DAPO

        self.backward(policy_loss)

        Outputs = collections.namedtuple('Outputs', ['loss', 'rewards', "ratio"])
        return Outputs(
            loss=policy_loss.item(),
            rewards=torch.mean(rewards).item(),
            ratio=torch.mean(ratio).detach().cpu().item()
        )


class ParallelGSPOTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            clip_range: float = 0.2,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(policy, optimizer, save_optim, accumulation_steps=accumulation_steps)
        self.policy = policy
        self.clip_range = clip_range

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())
        old_action_logprobs = rollout_data.action_logprobs.to(self.policy.device())

        logits = self.policy.forward(obs).logits
        action_logprobs = torch.gather(
            torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        # sequence-level ratio
        ratio = torch.exp(masked_mean(action_logprobs - old_action_logprobs, mask=action_masks, dim=-1))
        # sequence-level reward
        rewards = masked_mean(rewards, mask=action_masks, dim=-1)

        policy_loss = rewards * ratio
        if self.clip_range > 0:
            clipped_policy_loss = rewards * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = torch.min(policy_loss, clipped_policy_loss)
        policy_loss = - torch.mean(policy_loss)

        self.backward(policy_loss)

        Outputs = collections.namedtuple('Outputs', ['loss', 'rewards', "ratio"])
        return Outputs(
            loss=policy_loss.item(),
            rewards=torch.mean(rewards).item(),
            ratio=torch.mean(ratio).detach().cpu().item()
        )


class ParallelCISPOTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            clip_range_higher: float = 0.28,
            clip_range_lower: float = 1000.0,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(policy, optimizer, save_optim, accumulation_steps=accumulation_steps)
        self.policy = policy
        self.clip_range_higher = clip_range_higher
        self.clip_range_lower = clip_range_lower

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())
        old_action_logprobs = rollout_data.action_logprobs.to(self.policy.device())

        actions = torch.masked_select(actions.view(-1), action_masks.view(-1))
        rewards = torch.masked_select(rewards.view(-1), action_masks.view(-1))
        old_action_logprobs = torch.masked_select(old_action_logprobs.view(-1), action_masks.view(-1))

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]
        rewards = rewards.to(logits.dtype)
        action_logprobs = torch.gather(
            torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        ratio = torch.exp(action_logprobs - old_action_logprobs).detach()
        ratio = torch.clamp(ratio, 1 - self.clip_range_lower, 1 + self.clip_range_higher)
        policy_loss = ratio * rewards * action_logprobs
        policy_loss = - torch.sum(policy_loss)  # Token-level policy gradient loss in DAPO

        self.backward(policy_loss)

        Outputs = collections.namedtuple('Outputs', ['loss', 'rewards', "ratio"])
        return Outputs(
            loss=policy_loss.item(),
            rewards=torch.mean(rewards).item(),
            ratio=torch.mean(ratio).detach().cpu().item()
        )


class ParallelGRPOTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            clip_range: float = 0.2,
            kl_coef: float = 0.04,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(policy, optimizer, save_optim, accumulation_steps=accumulation_steps)
        self.policy = policy
        self.clip_range = clip_range
        self.kl_coef = kl_coef

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        advantages = rollout_data.advantages.to(self.policy.device())
        old_action_logprobs = rollout_data.action_logprobs.to(self.policy.device())

        outputs = self.policy.forward(obs)
        action_logprobs = torch.gather(
            torch.log_softmax(outputs.logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        advantages = torch.masked_select(advantages.view(-1), action_masks.view(-1))
        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(action_logprobs - old_action_logprobs)
        ratio = torch.masked_select(ratio.view(-1), action_masks.view(-1))
        # clipped surrogate loss
        policy_loss = advantages * ratio
        if self.clip_range > 0:
            clipped_actor_loss = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = torch.min(policy_loss, clipped_actor_loss)
        policy_loss = - torch.mean(policy_loss)

        kl_loss = 0.0
        if hasattr(rollout_data, "ref_action_logprobs"):
            ref_action_logprobs = rollout_data.ref_action_logprobs.to(self.policy.device())
            probs_ratios = torch.exp(ref_action_logprobs - action_logprobs)
            kl_loss = self.kl_coef * torch.masked_select(
                (probs_ratios - (ref_action_logprobs - action_logprobs) - 1).view(-1),
                action_masks.view(-1)
            ).mean()

        loss = policy_loss + kl_loss

        self.backward(loss)

        Outputs = collections.namedtuple('Outputs', [
            'loss', "policy_loss", 'advantages', "kl_loss", "ratio"])
        return Outputs(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            advantages=torch.mean(advantages).item(),
            kl_loss=kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            ratio=torch.mean(ratio).detach().cpu().item()
        )
