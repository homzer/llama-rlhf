import collections
from typing import Generator

import numpy as np
import torch

RolloutBufferSample = collections.namedtuple(
    "RolloutBufferSample", [
        "observations",
        "actions",
        "old_values",
        "old_action_logits",
        "advantages",
        "returns",
        "action_masks"
    ]
)


class RolloutBuffer:
    def __init__(self, buffer_size: int, max_seq_len: int, output_tensor=True):
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self.output_tensor = output_tensor

        self.observations = None
        self.actions = None
        self.rewards = None
        self.values = None
        self.action_logits = None
        self.action_masks = None

        self.advantages = None
        self.returns = None

        self.gamma = 0.99
        self.gae_lambda = 0.8
        self.size = 0

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.int64)
        self.actions = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)
        self.action_logits = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)
        self.action_masks = np.zeros((self.buffer_size, self.max_seq_len), dtype=bool)
        self.advantages = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)

        return self

    def add(
            self,
            obs: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            values: np.ndarray,
            action_logits: np.ndarray,
            action_masks: np.ndarray,
    ):
        bzs = obs.shape[0]
        if bzs + self.size > self.buffer_size:
            raise RuntimeError("Buffer is full!")
        self.observations[self.size: self.size + bzs, :] = obs.copy()
        self.actions[self.size: self.size + bzs, :] = actions.copy()
        self.rewards[self.size: self.size + bzs, :] = rewards.copy()
        self.values[self.size: self.size + bzs, :] = values.copy()
        self.action_logits[self.size: self.size + bzs, :] = action_logits.copy()
        self.action_masks[self.size: self.size + bzs, :] = action_masks.copy()
        self.size += bzs

    def compute_returns_and_advantage(self):
        last_gae_lam = 0
        for step in reversed(range(self.max_seq_len - 1)):
            next_values = self.values[:, step + 1]
            delta = self.rewards[:, step] + self.gamma * next_values - self.values[:, step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam * self.action_masks[:, step + 1]
            self.advantages[:, step] = last_gae_lam
        self.returns = self.advantages + self.values

    def get(self, batch_size: int) -> Generator[RolloutBufferSample, None, None]:
        indices = np.random.permutation(self.size)
        start_idx = 0
        while start_idx < self.size:
            batch_indices = indices[start_idx: start_idx + batch_size]
            yield RolloutBufferSample(
                observations=self.to_tensor(self.observations[batch_indices]),
                actions=self.to_tensor(self.actions[batch_indices]),
                old_values=self.to_tensor(self.values[batch_indices]),
                old_action_logits=self.to_tensor(self.action_logits[batch_indices]),
                advantages=self.to_tensor(self.advantages[batch_indices]),
                returns=self.to_tensor(self.returns[batch_indices]),
                action_masks=self.to_tensor(self.action_masks[batch_indices])
            )
            start_idx += batch_size

    def to_tensor(self, x):
        return torch.tensor(x) if self.output_tensor else x
