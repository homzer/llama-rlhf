from typing import List, Union

import numpy as np
import torch

from src.ppo.buffer import RolloutBuffer, PolicyRolloutBuffer, PolicyRolloutBufferSample, EnvRolloutBuffer
from src.ppo.env import LlamaRewardEnv
from src.ppo.policy import AbstractPolicyForCausalLM, AbstractParallelPolicyForCausalLM


class BufferCollector:
    def __init__(
            self,
            env: LlamaRewardEnv,
            policy: Union[AbstractPolicyForCausalLM, AbstractParallelPolicyForCausalLM],
            buffer_size: int,
            max_seq_len: int,
    ):
        self.env = env
        self.policy = policy
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size

    def forward(self, instructions: List[str]) -> RolloutBuffer:
        bzs = len(instructions)
        with torch.no_grad():
            policy_outputs = self.policy.forward(instructions)
        obs = policy_outputs.obs.cpu().numpy()
        actions = policy_outputs.actions.cpu().numpy()
        values = policy_outputs.values.cpu().numpy()
        action_logits = policy_outputs.action_logits.cpu().numpy()
        action_masks = policy_outputs.action_masks.cpu().numpy()

        rewards = np.zeros((bzs, self.max_seq_len), dtype=np.float32)
        action_rewards = self.env.step(instructions, actions, action_masks)
        for i in range(bzs):
            rewards[i, :][action_masks[i]] = action_rewards[i]

        rollout_buffer = RolloutBuffer(
            obs=obs,
            actions=actions,
            rewards=rewards,
            values=values,
            action_logits=action_logits,
            action_masks=action_masks
        )

        rollout_buffer.compute_returns_and_advantage()

        return rollout_buffer


class PolicyBufferCollector:
    def __init__(self, policy: Union[AbstractPolicyForCausalLM, AbstractParallelPolicyForCausalLM]):
        self.policy = policy

    def forward(self, instructions: List[str]) -> PolicyRolloutBuffer:
        with torch.no_grad():
            policy_outputs = self.policy.forward(instructions)
        obs = policy_outputs.obs.cpu().numpy()
        actions = policy_outputs.actions.cpu().numpy()
        values = policy_outputs.values.cpu().numpy()
        action_logits = policy_outputs.action_logits.cpu().numpy()
        action_masks = policy_outputs.action_masks.cpu().numpy()

        return PolicyRolloutBuffer(obs, actions, values, action_logits, action_masks)


class EnvBufferCollector:
    def __init__(self, env: LlamaRewardEnv):
        self.env = env

    def forward(self, rollout_data: PolicyRolloutBufferSample) -> EnvRolloutBuffer:
        action_rewards = self.env.step(
            obs=rollout_data.instructions,
            actions=rollout_data.actions,
            action_masks=rollout_data.action_masks
        )
        return EnvRolloutBuffer(action_rewards, rollout_data.action_masks)

