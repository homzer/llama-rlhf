from typing import List

import numpy as np
import torch

from src.ppo.buffer import RolloutBuffer
from src.ppo.env import LlamaRewardEnv
from src.ppo.policy import ActorCriticPolicyForCausalLM


class BufferCollector:
    def __init__(
            self,
            env: LlamaRewardEnv,
            policy: ActorCriticPolicyForCausalLM,
            buffer_size: int,
            max_seq_len: int,
    ):
        self.env = env
        self.policy = policy
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size

    def collect(self, instructions: List[str]) -> RolloutBuffer:
        bzs = len(instructions)
        rollout_buffer = RolloutBuffer(
            buffer_size=self.buffer_size,
            max_seq_len=self.max_seq_len
        ).reset()
        with torch.no_grad():
            policy_outputs = self.policy.forward(instructions)
        obs = policy_outputs.obs.cpu().numpy()
        actions = policy_outputs.actions.cpu().numpy()
        values = policy_outputs.values.cpu().numpy()
        action_logits = policy_outputs.action_logits.cpu().numpy()
        action_masks = policy_outputs.action_masks.cpu().numpy()

        # rewards = np.zeros((bzs, self.max_seq_len), dtype=np.float32)
        # action_rewards = self.env.step(instructions, actions, action_masks)
        # for i in range(bzs):
        #     rewards[i, :][action_masks[i]] = action_rewards[i]
        rewards = torch.randn((bzs, self.max_seq_len), dtype=torch.float32).numpy()

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
