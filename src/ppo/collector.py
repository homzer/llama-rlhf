from typing import List

import numpy as np
import torch

from src.ppo.buffer import RolloutBuffer
from src.ppo.env import CodeEnv
from src.ppo.policy import ActorCriticPolicyForCausalLM


class BufferCollector:
    def __init__(
            self,
            env: CodeEnv,
            policy: ActorCriticPolicyForCausalLM,
            buffer_size: int,
            max_seq_len: int,
    ):
        self.env = env
        self.policy = policy
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size

    def collect(self, instructions: List[str], labels: List[str]) -> RolloutBuffer:
        bzs = len(instructions)
        self.policy.eval()
        rollout_buffer = RolloutBuffer(
            buffer_size=self.buffer_size,
            max_seq_len=self.max_seq_len
        ).reset()
        # instr -> LLM -> output
        with torch.no_grad():
            policy_outputs = self.policy.forward(instructions)
        obs = policy_outputs.obs.cpu().numpy()
        actions = policy_outputs.actions.cpu().numpy()
        values = policy_outputs.values.cpu().numpy()
        action_logits = policy_outputs.action_logits.cpu().numpy()
        action_masks = policy_outputs.action_masks.cpu().numpy()

        rewards = np.zeros((bzs, self.max_seq_len), dtype=np.float32)
        for i, action in enumerate(actions):
            env_outputs = self.env.step(
                actions=action,
                label=labels[i],
                action_masks=action_masks[i],
            )
            # shift left
            rewards[i, :][action_masks[i]] = env_outputs.reward

        rollout_buffer.add(
            obs=obs,
            actions=actions,
            rewards=rewards,
            values=values,
            action_logits=action_logits,
            action_masks=action_masks
        )

        rollout_buffer.compute_returns_and_advantage()

        return rollout_buffer
