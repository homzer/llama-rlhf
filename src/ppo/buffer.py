import collections
from typing import Generator, List, Union

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

PolicyRolloutBufferSample = collections.namedtuple(
    "PolicyRolloutBufferSample", [
        "instructions",
        "obs",
        "actions",
        "values",
        "action_logits",
        "action_masks"
    ]
)

EnvRolloutBufferSample = collections.namedtuple(
    "EnvRolloutBufferSample", [
        "rewards",
    ]
)


# class RolloutBuffer:
#     def __init__(self, buffer_size: int, max_seq_len: int, output_tensor=True):
#         self.max_seq_len = max_seq_len
#         self.buffer_size = buffer_size
#         self.output_tensor = output_tensor
#
#         self.obs = None
#         self.actions = None
#         self.rewards = None
#         self.values = None
#         self.action_logits = None
#         self.action_masks = None
#
#         self.advantages = None
#         self.returns = None
#
#         self.gamma = 0.99
#         self.gae_lambda = 0.8
#         self.size = 0
#
#     def reset(self):
#         self.obs = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.int64)
#         self.actions = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.int64)
#         self.rewards = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)
#         self.values = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)
#         self.action_logits = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)
#         self.action_masks = np.zeros((self.buffer_size, self.max_seq_len), dtype=bool)
#         self.advantages = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)
#
#         return self
#
#     def add(
#             self,
#             obs: np.ndarray,
#             actions: np.ndarray,
#             rewards: np.ndarray,
#             values: np.ndarray,
#             action_logits: np.ndarray,
#             action_masks: np.ndarray,
#     ):
#         bzs = obs.shape[0]
#         if bzs + self.size > self.buffer_size:
#             raise RuntimeError("Buffer is full!")
#         self.obs[self.size: self.size + bzs, :] = obs.copy()
#         self.actions[self.size: self.size + bzs, :] = actions.copy()
#         self.rewards[self.size: self.size + bzs, :] = rewards.copy()
#         self.values[self.size: self.size + bzs, :] = values.copy()
#         self.action_logits[self.size: self.size + bzs, :] = action_logits.copy()
#         self.action_masks[self.size: self.size + bzs, :] = action_masks.copy()
#         self.size += bzs
#
#     def compute_returns_and_advantage(self):
#         last_gae_lam = 0
#         for step in reversed(range(self.max_seq_len - 1)):
#             next_values = self.values[:, step + 1]
#             delta = self.rewards[:, step] + self.gamma * next_values - self.values[:, step]
#             last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam * self.action_masks[:, step + 1]
#             self.advantages[:, step] = last_gae_lam
#         self.returns = self.advantages + self.values
#
#     def get(self, batch_size: int) -> Generator[RolloutBufferSample, None, None]:
#         indices = np.random.permutation(self.size)
#         start_idx = 0
#         while start_idx < self.size:
#             batch_indices = indices[start_idx: start_idx + batch_size]
#             yield RolloutBufferSample(
#                 observations=torch.tensor(self.obs[batch_indices]),
#                 actions=torch.tensor(self.actions[batch_indices]),
#                 old_values=torch.tensor(self.values[batch_indices]),
#                 old_action_logits=torch.tensor(self.action_logits[batch_indices]),
#                 advantages=torch.tensor(self.advantages[batch_indices]),
#                 returns=torch.tensor(self.returns[batch_indices]),
#                 action_masks=torch.tensor(self.action_masks[batch_indices])
#             )
#             start_idx += batch_size
#
#     def to_tensor(self, x):
#         return torch.tensor(x) if self.output_tensor else x


class RolloutBuffer:
    def __init__(
            self,
            obs: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            values: np.ndarray,
            action_logits: np.ndarray,
            action_masks: np.ndarray,
    ):
        self.obs = None
        self.actions = None
        self.rewards = None
        self.values = None
        self.action_logits = None
        self.action_masks = None
        self.advantages = None
        self.returns = None

        self.gamma = 0.99
        self.gae_lambda = 0.8
        self.buffer_size = obs.shape[0]
        self.max_seq_len = obs.shape[1]

        self._set(obs, actions, rewards, values, action_logits, action_masks)

    def _set(self, obs, actions, rewards, values, action_logits, action_masks):
        self.obs = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.int64)
        self.actions = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.int64)
        self.values = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)
        self.action_logits = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)
        self.action_masks = np.zeros((self.buffer_size, self.max_seq_len), dtype=bool)
        self.advantages = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)

        self.obs[:, :] = obs.copy()
        self.actions[:, :] = actions.copy()
        self.rewards[:, :] = rewards.copy()
        self.values[:, :] = values.copy()
        self.action_logits[:, :] = action_logits.copy()
        self.action_masks[:, :] = action_masks.copy()

        assert np.sum(self.rewards[~ self.action_masks]) == 0  # Check rewards correctness

    def compute_returns_and_advantage(self):
        last_gae_lam = 0
        for step in reversed(range(self.max_seq_len - 1)):
            next_values = self.values[:, step + 1]
            delta = self.rewards[:, step] + self.gamma * next_values - self.values[:, step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam * self.action_masks[:, step + 1]
            self.advantages[:, step] = last_gae_lam
        self.returns = self.advantages + self.values

    def get(self, batch_size: int) -> Generator[RolloutBufferSample, None, None]:
        size = self.obs.shape[0]
        indices = np.random.permutation(size)
        start_idx = 0
        while start_idx < size:
            batch_indices = indices[start_idx: start_idx + batch_size]
            yield RolloutBufferSample(
                observations=torch.tensor(self.obs[batch_indices]),
                actions=torch.tensor(self.actions[batch_indices]),
                old_values=torch.tensor(self.values[batch_indices]),
                old_action_logits=torch.tensor(self.action_logits[batch_indices]),
                advantages=torch.tensor(self.advantages[batch_indices]),
                returns=torch.tensor(self.returns[batch_indices]),
                action_masks=torch.tensor(self.action_masks[batch_indices])
            )
            start_idx += batch_size


class PolicyRolloutBuffer:
    def __init__(
            self,
            instructions: List[str] = None,
            obs: np.ndarray = None,
            actions: np.ndarray = None,
            values: np.ndarray = None,
            action_logits: np.ndarray = None,
            action_masks: np.ndarray = None
    ):
        self.instructions = None
        self.obs = None
        self.actions = None
        self.values = None
        self.action_logits = None
        self.action_masks = None

        if obs is not None:
            self._set(instructions, obs, actions, values, action_logits, action_masks)

    def __len__(self):
        return 0 if self.instructions is None else len(self.instructions)

    def _set(self, instructions, obs, actions, values, action_logits, action_masks):
        buffer_size = obs.shape[0]
        max_seq_len = obs.shape[1]

        self.obs = np.zeros((buffer_size, max_seq_len), dtype=np.int64)
        self.actions = np.zeros((buffer_size, max_seq_len), dtype=np.int64)
        self.values = np.zeros((buffer_size, max_seq_len), dtype=np.float32)
        self.action_logits = np.zeros((buffer_size, max_seq_len), dtype=np.float32)
        self.action_masks = np.zeros((buffer_size, max_seq_len), dtype=bool)

        self.instructions = np.array(instructions)
        self.obs[:, :] = obs.copy()
        self.actions[:, :] = actions.copy()
        self.values[:, :] = values.copy()
        self.action_logits[:, :] = action_logits.copy()
        self.action_masks[:, :] = action_masks.copy()

    def extend(self, rollout_buffer):
        if self.obs is None:
            self._set(
                rollout_buffer.instructions,
                rollout_buffer.obs,
                rollout_buffer.actions,
                rollout_buffer.values,
                rollout_buffer.action_logits,
                rollout_buffer.action_masks
            )
        else:
            self.instructions = np.concatenate([self.instructions, rollout_buffer.instructions], axis=0)
            self.obs = np.concatenate([self.obs, rollout_buffer.obs], axis=0)
            self.actions = np.concatenate([self.actions, rollout_buffer.actions], axis=0)
            self.values = np.concatenate([self.values, rollout_buffer.values], axis=0)
            self.action_logits = np.concatenate([self.action_logits, rollout_buffer.action_logits], axis=0)
            self.action_masks = np.concatenate([self.action_masks, rollout_buffer.action_masks], axis=0)

        return self

    def get(self, batch_size: int) -> Generator[PolicyRolloutBufferSample, None, None]:
        size = self.obs.shape[0]
        indices = np.arange(size)
        start_idx = 0
        while start_idx < size:
            batch_indices = indices[start_idx: start_idx + batch_size]
            yield PolicyRolloutBufferSample(
                instructions=self.instructions[batch_indices],
                obs=self.obs[batch_indices],
                actions=self.actions[batch_indices],
                values=self.values[batch_indices],
                action_logits=self.action_logits[batch_indices],
                action_masks=self.action_masks[batch_indices]
            )
            start_idx += batch_size


class EnvRolloutBuffer:
    def __init__(self, rewards: Union[np.ndarray, List] = None, action_masks: np.ndarray = None):
        self.rewards = None
        if rewards is not None:
            self._set(rewards, action_masks)

    def _set(self, rewards, action_masks):
        buffer_size = action_masks.shape[0]
        max_seq_len = action_masks.shape[1]
        self.rewards = np.zeros((buffer_size, max_seq_len), dtype=np.float32)
        if type(rewards) is list:
            for i in range(buffer_size):
                self.rewards[i, :][action_masks[i]] = rewards[i]
        else:
            self.rewards[:, :] = rewards.copy()

    def extend(self, rollout_buffer):
        if self.rewards is None:
            self._set(rollout_buffer.rewards, rollout_buffer.action_masks)
        else:
            self.rewards = np.concatenate([self.rewards, rollout_buffer.rewards], axis=0)
        return self

    def get(self, batch_size: int) -> Generator[EnvRolloutBufferSample, None, None]:
        size = self.rewards.shape[0]
        indices = np.arange(size)
        start_idx = 0
        while start_idx < size:
            batch_indices = indices[start_idx: start_idx + batch_size]
            yield EnvRolloutBufferSample(rewards=self.rewards[batch_indices])
            start_idx += batch_size
