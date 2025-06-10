"""
Gradient Norm Visualize
"""
import os

import fire
import numpy as np
import torch

from policy_train_policy_gradient_with_rule_rm import collect_actor_buffer_with_label, \
    collect_rule_based_verifier_buffer
from src.dataset import JsonDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.models.modeling import ParallelModelForCausalLM
from src.parallel.initialize import setup_model_parallel
from src.parallel.optimizer import ParallelOptimizer
from src.ppo.buffer import PolicyRolloutBuffer, CriticRolloutBuffer, RolloutBuffer, PolicyRolloutBufferSample
from src.utils import json_load, print_current_func_args, proxy_neg_distribution, json_dump


def reinforce_forward(
        policy: ParallelModelForCausalLM,
        rollout_data: PolicyRolloutBufferSample
) -> (torch.Tensor, torch.Tensor):
    policy.train()
    obs = rollout_data.observations.to(policy.device())
    actions = rollout_data.actions.to(policy.device())
    action_masks = rollout_data.action_masks.to(policy.device())
    rewards = rollout_data.rewards.to(policy.device())
    logits = policy.forward(obs).logits
    rewards = rewards.to(logits.dtype)
    action_logprobs = torch.gather(
        torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
    ).squeeze(-1)
    action_logprobs = torch.masked_select(action_logprobs.view(-1), action_masks.view(-1))
    rewards = torch.masked_select(rewards.view(-1), action_masks.view(-1))
    loss = - torch.mean(rewards * action_logprobs)
    return loss, action_logprobs


def ppo_forward(
        policy: ParallelModelForCausalLM,
        rollout_data: PolicyRolloutBufferSample,
        clip_range: float = 0
) -> (torch.Tensor, torch.Tensor):
    policy.train()
    obs = rollout_data.observations.to(policy.device())
    actions = rollout_data.actions.to(policy.device())
    action_masks = rollout_data.action_masks.to(policy.device())
    rewards = rollout_data.rewards.to(policy.device())
    old_action_logprobs = rollout_data.old_action_logprobs.to(policy.device())
    logits = policy.forward(obs).logits
    action_logprobs = torch.gather(
        torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
    ).squeeze(-1)
    rewards = torch.masked_select(rewards.view(-1), action_masks.view(-1))
    # ratio between old and new policy, should be one at the first iteration
    action_logprobs = torch.masked_select(action_logprobs.view(-1), action_masks.view(-1))
    old_action_logprobs = torch.masked_select(old_action_logprobs.view(-1), action_masks.view(-1))
    ratio = torch.exp(action_logprobs - old_action_logprobs)
    # ratio = torch.masked_select(ratio.view(-1), action_masks.view(-1))
    # clipped surrogate loss
    policy_loss = rewards * ratio
    if clip_range > 0:
        clipped_actor_loss = rewards * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = torch.min(policy_loss, clipped_actor_loss)
    loss = - torch.mean(policy_loss)
    return loss, action_logprobs


def convex_forward(
        policy: ParallelModelForCausalLM,
        rollout_data: PolicyRolloutBufferSample,
        delta: float = 0.01
) -> (torch.Tensor, torch.Tensor):
    criterion = torch.nn.KLDivLoss(reduction='none', log_target=True)
    policy.train()
    obs = rollout_data.observations.to(policy.device())
    actions = rollout_data.actions.to(policy.device())
    action_masks = rollout_data.action_masks.to(policy.device())
    rewards = rollout_data.rewards.to(policy.device())
    logits = policy.forward(obs).logits
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
    log_targets = proxy_neg_distribution(logits[~pos_reward_masks], actions[~pos_reward_masks], delta)
    loss_neg = - rewards[~pos_reward_masks] * criterion.forward(
        torch.log_softmax(logits[~pos_reward_masks], dim=-1), target=log_targets
    ).sum(-1)
    loss = torch.mean(torch.cat([loss_pos, loss_neg]))
    return loss, action_logprobs


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


def compute_result_dict(policy: ParallelModelForCausalLM, action_logprobs: torch.Tensor) -> dict:
    grad_norms = []
    for name, param in policy.named_parameters():
        if "q_proj.weight" in name or "k_proj.weight" in name or "v_proj.weight" in name:
            grad_norms.append(torch.linalg.norm(param.grad).item())
    action_probs = action_logprobs.exp()
    return {"gradient": np.mean(grad_norms).item(), "action_probs": action_probs.mean().item()}


def run(
        train_file: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
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
                policy_rollout_buffer=policy_rollout_buffer, task="math"
            )

            # Filter for correct answer buffers
            policy_rollout_buffer, verifier_rollout_buffer = filter_rollout_buffer(
                policy_rollout_buffer=policy_rollout_buffer,
                verifier_rollout_buffer=verifier_rollout_buffer,
                mode=mode
            )

            rollout_buffer = PolicyRolloutBuffer(
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
                trainer = reinforce_forward
            elif train_strategy == "ratio":
                print("Using ParallelGRPOTrainerForCausalLM")
                trainer = ppo_forward
            elif train_strategy == "convex":
                print("Using ParallelPolicyGradientConvexTrainerForCausalLM")
                trainer = convex_forward
            else:
                raise ValueError(train_strategy)

            print('Policy training ...')
            results = []
            timer = Timer(total=(len(rollout_buffer) // max_batch_size) * inner_epochs, episode=100)
            for data in rollout_buffer.get(max_batch_size, shuffle=False):
                for inner_epoch in range(inner_epochs):
                    timer.step()
                    loss, action_logprobs = trainer(policy, data)
                    optimizer.zero_grad()
                    loss.backward()
                    results.append(compute_result_dict(policy, action_logprobs))
                    optimizer.step()
                    print(f'Loss: {loss}')
                break
            if parallel_infos.global_rank == 0:
                os.makedirs(save_dir, exist_ok=True)
                json_dump(results, os.path.join(save_dir, "grad.jsonl"))
            exit(0)


if __name__ == '__main__':
    fire.Fire(run)
