"""
Gradient Norm Visualize
"""
import collections
import os

import fire
import numpy as np
import torch

from policy_train_ppo_with_rule_rm import collect_actor_buffer_with_label, collect_rule_based_verifier_buffer
from src.dataset import JsonDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.models.modeling import ParallelModelForCausalLM
from src.parallel.data_parallel.utils import gather_tensor_from_data_parallel_region
from src.parallel.initialize import setup_model_parallel
from src.parallel.optimizer import ParallelOptimizer
from src.ppo.buffer import PPORolloutBuffer, CriticRolloutBuffer, RolloutBuffer, PPORolloutBufferSample
from src.trainer import ParallelTrainer
from src.utils import json_load, print_current_func_args, proxy_neg_distribution, json_dump, create_target_distribution, \
    create_target_distribution

Outputs = collections.namedtuple('Outputs', [
    'gradient', 'action_probs', 'entropy', 'pos_gradient', 'neg_gradient'])


class ParallelGRPOTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            model: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            clip_range: float = 0.2,
            kl_coef: float = 0.04,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(model, optimizer, save_optim, accumulation_steps=accumulation_steps)
        self.model = model
        self.clip_range = clip_range
        self.kl_coef = kl_coef

    def forward(self, rollout_data: PPORolloutBufferSample):
        self.model.train()

        obs = rollout_data.obs.to(self.model.device())
        actions = rollout_data.actions.to(self.model.device())
        action_masks = rollout_data.action_masks.to(self.model.device())
        rewards = rollout_data.rewards.to(self.model.device())
        old_action_logprobs = rollout_data.old_action_logprobs.to(self.model.device())

        logits = self.model.forward(obs).logits
        logits = torch.reshape(logits, shape=[-1, logits.shape[-1]])[action_masks.view(-1)]
        actions = torch.masked_select(actions.view(-1), action_masks.view(-1))
        old_action_logprobs = torch.masked_select(old_action_logprobs.view(-1), action_masks.view(-1))
        rewards = torch.masked_select(rewards.to(logits.dtype).view(-1), action_masks.view(-1))

        action_logprobs = torch.gather(
            torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(action_logprobs - old_action_logprobs)
        # clipped surrogate loss
        policy_loss = rewards * ratio
        if self.clip_range > 0:
            clipped_actor_loss = rewards * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = torch.min(policy_loss, clipped_actor_loss)

        pos_reward_masks = rewards > 0
        loss_pos = - policy_loss[pos_reward_masks]
        loss_neg = - policy_loss[~pos_reward_masks]
        pos_grad = compute_average_grad_norm_with_loss(policy=self.model, optimizer=self.optimizer, loss=loss_pos)
        neg_grad = compute_average_grad_norm_with_loss(policy=self.model, optimizer=self.optimizer, loss=loss_neg)

        policy_loss = - torch.mean(policy_loss)

        kl_loss = 0.0
        if rollout_data.ref_action_logprobs is not None:
            ref_action_logprobs = rollout_data.ref_action_logprobs.to(self.model.device())
            ref_action_logprobs = torch.masked_select(ref_action_logprobs.view(-1), action_masks.view(-1))
            probs_ratios = torch.exp(ref_action_logprobs - action_logprobs)
            kl_loss = self.kl_coef * (probs_ratios - (ref_action_logprobs - action_logprobs) - 1).mean()

        loss = policy_loss + kl_loss

        self.step += 1
        loss = loss / self.accumulation_steps
        loss.backward()
        gradient = compute_average_grad_norm(self.model)
        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        with torch.no_grad():
            entropy = - torch.sum(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1), dim=-1)
            entropy = entropy.mean().item()
            action_probs = gather_tensor_from_data_parallel_region(action_logprobs.exp().mean()).mean().item()

        return Outputs(gradient=gradient, action_probs=action_probs, entropy=entropy, pos_gradient=pos_grad, neg_gradient=neg_grad)


class ParallelPolicyGradientTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(policy, optimizer, save_optim, accumulation_steps=accumulation_steps)
        self.policy = policy
        self.clip_range = 0.2

    def forward(self, rollout_data: PPORolloutBufferSample):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())

        logits = self.policy.forward(obs).logits
        logits = torch.reshape(logits, shape=[-1, logits.shape[-1]])[action_masks.view(-1)]
        actions = torch.masked_select(actions.view(-1), action_masks.view(-1))
        rewards = torch.masked_select(rewards.to(logits.dtype).view(-1), action_masks.view(-1))

        action_logprobs = torch.gather(
            torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        pos_reward_masks = rewards > 0
        loss_pos = - rewards[pos_reward_masks] * action_logprobs[pos_reward_masks]
        loss_neg = - rewards[~pos_reward_masks] * action_logprobs[~pos_reward_masks]

        pos_grad = compute_average_grad_norm_with_loss(policy=self.policy, optimizer=self.optimizer, loss=loss_pos)
        neg_grad = compute_average_grad_norm_with_loss(policy=self.policy, optimizer=self.optimizer, loss=loss_neg)

        loss = - torch.mean(rewards * action_logprobs)

        self.step += 1
        loss = loss / self.accumulation_steps
        loss.backward()
        gradient = compute_average_grad_norm(self.policy)
        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        with torch.no_grad():
            entropy = - torch.sum(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1), dim=-1)
            entropy = entropy.mean().item()

        action_probs = compute_average_action_probs(logits, actions)

        return Outputs(gradient=gradient, action_probs=action_probs, entropy=entropy, pos_gradient=pos_grad, neg_gradient=neg_grad)


class ParallelPolicyGradientConvexTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            delta: float = 0.01,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(policy, optimizer, save_optim, accumulation_steps=accumulation_steps)
        self.policy = policy
        self.delta = delta
        self.criterion = torch.nn.KLDivLoss(reduction='none', log_target=True)

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

        loss_pos = - rewards[pos_reward_masks] * torch.gather(
            torch.log_softmax(logits[pos_reward_masks], dim=-1), dim=-1, index=actions[pos_reward_masks].unsqueeze(-1)
        ).squeeze(-1)

        # compute loss for negative reward tokens
        log_targets = proxy_neg_distribution(logits[~pos_reward_masks], actions[~pos_reward_masks], self.delta)
        loss_neg = - rewards[~pos_reward_masks] * self.criterion.forward(
            torch.log_softmax(logits[~pos_reward_masks], dim=-1), target=log_targets
        ).sum(-1)

        pos_grad = compute_average_grad_norm_with_loss(policy=self.policy, optimizer=self.optimizer, loss=loss_pos)
        neg_grad = compute_average_grad_norm_with_loss(policy=self.policy, optimizer=self.optimizer, loss=loss_neg)

        loss = torch.mean(torch.cat([loss_pos, loss_neg]))

        self.step += 1
        loss = loss / self.accumulation_steps
        loss.backward()
        gradient = compute_average_grad_norm(self.policy)
        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.step % 100 == 0:
            loss_pos_item = loss_pos.mean().nan_to_num(0).item()
            loss_neg_item = loss_neg.mean().nan_to_num(0).item()
            print(f"Positive Reward Loss: {loss_pos_item} | Negative Reward Loss: {loss_neg_item}")

        with (torch.no_grad()):
            entropy = - torch.sum(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1), dim=-1)
            entropy = entropy.mean().item()

        action_probs = compute_average_action_probs(logits, actions)

        return Outputs(gradient=gradient, action_probs=action_probs, entropy=entropy, pos_gradient=pos_grad, neg_gradient=neg_grad)


class ParallelPolicyGradientConvexBoundedTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            rho_pos: float = 1.8,
            rho_neg: float = 0.8,
            save_optim: bool = False,
            accumulation_steps: int = 1,
            skip_log_grad: bool = False
    ):
        super().__init__(policy, optimizer, save_optim, accumulation_steps=accumulation_steps)
        self.policy = policy
        self.rho_pos = rho_pos
        self.rho_neg = rho_neg
        self.skip_log_grad = skip_log_grad
        self.criterion = torch.nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, rollout_data: PPORolloutBufferSample):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        old_action_logprobs = rollout_data.old_action_logprobs.to(self.policy.device())
        rewards = rollout_data.rewards.to(self.policy.device())

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]
        actions = torch.masked_select(actions.view(-1), action_masks.view(-1))
        old_action_logprobs = torch.masked_select(old_action_logprobs.view(-1), action_masks.view(-1))
        rewards = torch.masked_select(rewards.to(logits.dtype).view(-1), action_masks.view(-1))
        pos_reward_masks = rewards > 0

        pos_log_targets = create_target_distribution(
            logits=logits[pos_reward_masks],
            actions=actions[pos_reward_masks],
            old_action_logprobs=old_action_logprobs[pos_reward_masks],
            rho=self.rho_pos
        )
        loss_pos = rewards[pos_reward_masks] * self.criterion.forward(
            torch.log_softmax(logits[pos_reward_masks], dim=-1), target=pos_log_targets
        ).sum(-1)

        # compute loss for negative reward tokens
        neg_log_targets = create_target_distribution(
            logits=logits[~pos_reward_masks],
            actions=actions[~pos_reward_masks],
            old_action_logprobs=old_action_logprobs[~pos_reward_masks],
            rho=self.rho_neg
        )
        loss_neg = - rewards[~pos_reward_masks] * self.criterion.forward(
            torch.log_softmax(logits[~pos_reward_masks], dim=-1), target=neg_log_targets
        ).sum(-1)

        if self.skip_log_grad:
            pos_grad = 0.0
            neg_grad = 0.0
        else:
            pos_grad = compute_average_grad_norm_with_loss(policy=self.policy, optimizer=self.optimizer, loss=loss_pos)
            neg_grad = compute_average_grad_norm_with_loss(policy=self.policy, optimizer=self.optimizer, loss=loss_neg)

        loss = torch.mean(torch.cat([loss_pos, loss_neg]))

        self.step += 1
        loss = loss / self.accumulation_steps
        loss.backward()
        gradient = compute_average_grad_norm(self.policy)
        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.step % 100 == 0:
            loss_pos_item = loss_pos.mean().nan_to_num(0).item()
            loss_neg_item = loss_neg.mean().nan_to_num(0).item()
            print(f"Positive Reward Loss: {loss_pos_item} | Negative Reward Loss: {loss_neg_item}")

        with (torch.no_grad()):
            entropy = - torch.sum(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1), dim=-1)
            entropy = entropy.mean().item()

        action_probs = compute_average_action_probs(logits, actions)

        return Outputs(gradient=gradient, action_probs=action_probs, entropy=entropy, pos_gradient=pos_grad, neg_gradient=neg_grad)


def filter_rollout_buffer(
        policy_rollout_buffer: RolloutBuffer,
        verifier_rollout_buffer: CriticRolloutBuffer,
        mode: int
):
    scores = []
    for i in range(verifier_rollout_buffer.size()):
        nonzero_indices = np.nonzero(verifier_rollout_buffer["action_masks"][i])[0]
        if len(nonzero_indices) > 0:
            scores.append(verifier_rollout_buffer["scores"][i][nonzero_indices[-1]])
        else:
            scores.append(0.0)
    scores = np.stack(scores, axis=0)
    indices = np.arange(policy_rollout_buffer.size())
    pos_indices = indices[scores > 0]
    neg_indices = indices[scores < 0]
    print("Number of positive samples:", len(pos_indices))
    print("Number of negative samples:", len(neg_indices))
    mix_indices = []
    for pos_idx, neg_idx in zip(pos_indices, neg_indices):
        mix_indices.append(pos_idx)
        mix_indices.append(neg_idx)

    if mode == 0:
        policy_rollout_buffer.rearrange(neg_indices)
        verifier_rollout_buffer.rearrange(neg_indices)
    elif mode == 1:
        policy_rollout_buffer.rearrange(pos_indices)
        verifier_rollout_buffer.rearrange(pos_indices)
    else:
        policy_rollout_buffer.rearrange(mix_indices)
        verifier_rollout_buffer.rearrange(mix_indices)
    return policy_rollout_buffer, verifier_rollout_buffer


@torch.no_grad()
def compute_average_grad_norm(
        policy: ParallelModelForCausalLM,
        return_tensor: bool = False,
        gather_from_data_parallel_region: bool = True
) -> float | torch.Tensor:
    grad_norm = []
    for name, param in policy.named_parameters():
        if "q_proj.weight" in name or "k_proj.weight" in name or "v_proj.weight" in name:
            grad_norm.append(torch.linalg.norm(param.grad))
    grad_norm = torch.stack(grad_norm)
    if gather_from_data_parallel_region:
        grad_norm = gather_tensor_from_data_parallel_region(grad_norm).mean()
    else:
        grad_norm = grad_norm.mean()
    return grad_norm if return_tensor else grad_norm.item()


@torch.no_grad()
def compute_average_action_probs(logits: torch.Tensor, actions: torch.Tensor) -> float:
    action_probs = torch.gather(
        torch.softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
    ).squeeze(-1).mean()
    return gather_tensor_from_data_parallel_region(action_probs).mean().item()


def compute_average_grad_norm_with_loss(
        policy: ParallelModelForCausalLM, optimizer: torch.optim.Optimizer, loss: torch.Tensor
) -> float:
    if len(loss) == 0:
        grad_norm = torch.tensor(0.).to(loss)
    else:
        loss.mean().backward(retain_graph=True)
        grad_norm = compute_average_grad_norm(policy, return_tensor=True, gather_from_data_parallel_region=False)
    optimizer.zero_grad()
    return gather_tensor_from_data_parallel_region(grad_norm).mean().item()


def compute_result_dict(policy: ParallelModelForCausalLM, action_logprobs: torch.Tensor, entropy: torch.Tensor) -> dict:
    grad_norms = []
    for name, param in policy.named_parameters():
        if "q_proj.weight" in name or "k_proj.weight" in name or "v_proj.weight" in name:
            grad_norms.append(torch.linalg.norm(param.grad).item())
    action_probs = action_logprobs.exp()
    return {"gradient": np.mean(grad_norms).item(), "action_probs": action_probs.mean().item(),
            "entropy": entropy.mean().item()}


def run(
        train_file: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        save_buffer_name: str = "grad.jsonl",
        max_batch_size: int = 6,
        max_generate_batch_size: int = 48,
        max_seq_len: int = 1024,
        temperature: float = 0.6,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
        epochs: int = 1,
        chunk_size: int = 1024,
        inner_epochs: int = 100,
        lr: float = 1e-6,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        train_strategy: str = "vanilla",
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
        clip_range: float = 0.0,
        delta: float = 0.3,
        rho_neg: float = 0.8,
        rho_pos: float = 1.8,
        mode: int = 0  # 0 for negative, 1 for positive, 2 for both
):
    parallel_infos = setup_model_parallel(
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )

    print_current_func_args()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir

    datalist = json_load(train_file)
    chunk_size = chunk_size or len(datalist)
    local_epochs = len(datalist) // chunk_size
    begin_global_epoch = begin_epoch // local_epochs
    begin_local_epoch = begin_epoch % local_epochs
    for global_epoch in range(begin_global_epoch, epochs):
        for local_epoch in range(begin_local_epoch, local_epochs):
            epoch = local_epoch + global_epoch * local_epochs
            print(f"Epoch - {epoch} of {local_epochs * epochs}")
            dataset = JsonDataset(f=datalist[local_epoch * chunk_size: (local_epoch + 1) * chunk_size])
            if len(dataset) == 0:
                continue

            # Collecting policy buffer
            policy_rollout_buffer = collect_actor_buffer_with_label(
                actor_model_type=policy_model_type,
                actor_config_file=policy_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=policy_tokenizer_file,
                dtype=dtype,
                actor_ckpt_dir=policy_ckpt_dir,
                epoch=epoch,
                actor_save_dir=save_dir,
                use_chat_template=use_chat_template,
                dataset=dataset,
                max_generate_batch_size=max_generate_batch_size,
                temperature=temperature,
                top_p=top_p,
                num_samples_per_prompt=num_samples_per_prompt
            )

            verifier_rollout_buffer = collect_rule_based_verifier_buffer(
                actor_rollout_buffer=policy_rollout_buffer, task="math"
            )

            # Filter for correct answer buffers
            policy_rollout_buffer, verifier_rollout_buffer = filter_rollout_buffer(
                policy_rollout_buffer=policy_rollout_buffer,
                verifier_rollout_buffer=verifier_rollout_buffer,
                mode=mode
            )

            rollout_buffer = PPORolloutBuffer(
                obs=policy_rollout_buffer["obs"],
                actions=policy_rollout_buffer["actions"],
                rewards=verifier_rollout_buffer["scores"],
                values=verifier_rollout_buffer["scores"],
                action_logits=policy_rollout_buffer["action_logits"],
                action_masks=policy_rollout_buffer["action_masks"],
                action_logprobs=policy_rollout_buffer["action_logprobs"],
                use_last_token_reward=True,
                reward_normalize=False
            )

            # Training
            policy, policy_tokenizer = get_parallel_model(
                model_type=policy_model_type,
                config_file=policy_config_file,
                max_seq_len=max_seq_len,
                tokenizer_file=policy_tokenizer_file,
                dtype=dtype,
            )
            policy.load(policy_ckpt_dir)
            optimizer = ParallelOptimizer(torch.optim.Adam(policy.parameters(), lr=lr))
            if train_strategy == "vanilla":
                print("Using ParallelPolicyGradientTrainerForCausalLM")
                trainer = ParallelPolicyGradientTrainerForCausalLM(policy, optimizer)
            elif train_strategy == "ratio":
                print("Using ParallelGRPOTrainerForCausalLM")
                trainer = ParallelGRPOTrainerForCausalLM(policy, optimizer, clip_range=clip_range)
            elif train_strategy == "convex":
                print("Using ParallelPolicyGradientConvexTrainerForCausalLM")
                trainer = ParallelPolicyGradientConvexTrainerForCausalLM(policy, optimizer, delta=delta)
            elif train_strategy == "convex-bounded":
                print("Using ParallelPolicyGradientConvexBoundedTrainerForCausalLM")
                trainer = ParallelPolicyGradientConvexBoundedTrainerForCausalLM(policy, optimizer, rho_pos=rho_pos, rho_neg=rho_neg, skip_log_grad=True)
            else:
                raise ValueError(train_strategy)

            print('Policy training ...')
            results = []
            timer = Timer(total=(len(rollout_buffer) // max_batch_size) * inner_epochs, episode=100)
            for data in rollout_buffer.get(max_batch_size, shuffle=False):
                for inner_epoch in range(inner_epochs):
                    timer.step()
                    outputs = trainer.forward(data)
                    results.append(dict(
                        gradient=outputs.gradient,
                        action_probs=outputs.action_probs,
                        entropy=outputs.entropy
                    ))
                break
            if parallel_infos.global_rank == 0:
                os.makedirs(save_dir, exist_ok=True)
                json_dump(results, os.path.join(save_dir, save_buffer_name))
            exit(0)


if __name__ == '__main__':
    fire.Fire(run)
