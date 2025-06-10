import collections
import json
import os.path
from typing import Generator, List, Union

import numpy as np
import torch

from src.entities import SlimLogits
from src.utils import masked_mean

PolicyRolloutBufferSample = collections.namedtuple(
    "RolloutBufferSample", [
        "observations",
        "actions",
        "old_values",
        "old_action_logits",
        "old_action_logprobs",
        "advantages",
        "returns",
        "action_masks",
        "rewards",
        "ref_action_logprobs"
    ]
)

ActorRolloutBufferSample = collections.namedtuple(
    "ActorRolloutBufferSample", [
        "instructions",
        "obs",
        "actions",
        "action_logits",
        "action_masks",
        "action_logprobs",
        "responses"
    ]
)

ActorRolloutBufferWithLabelSample = collections.namedtuple(
    "ActorRolloutBufferWithLabelSample", [
        "instructions",
        "obs",
        "actions",
        "action_logits",
        "action_masks",
        "action_logprobs",
        "responses",
        "labels"
    ]
)

SolverRolloutBufferSample = collections.namedtuple(
    "SolverRolloutBufferSample", [
        "instructions",
        "actions",
        "action_masks"
    ]
)

OutputRolloutBufferSample = collections.namedtuple(
    "OutputRolloutBufferSample", [
        "instructions",
        "outputs"
    ]
)

CriticRolloutBufferSample = collections.namedtuple(
    "RewardRolloutBufferSample", [
        "scores",
        "action_masks"
    ]
)

LogitsRolloutBufferSample = collections.namedtuple(
    "LogitsRolloutBufferSample", [
        "instructions",
        "outputs",
        "logits",
        "output_tokens_logps"
    ]
)

LogitsRolloutBufferSampleV0 = collections.namedtuple(
    "LogitsRolloutBufferSample", [
        "instructions",
        "logits"
    ]
)


class RolloutBuffer(dict):
    def __init__(self, **kwargs):
        super().__init__()
        self.set(**kwargs)

    def set(self, **kwargs):
        """initialize buffer."""
        for key, value in kwargs.items():
            self[key] = value.copy() if isinstance(value, np.ndarray) else np.array(value)

    def size(self) -> int:
        """Return the size of the rollout buffer"""
        if len(self) == 0:
            raise RuntimeError("Rollout buffer is not initialized.")
        for key in self.keys():
            assert len(self[key]) == len(self[next(iter(self))])
        return len(self[next(iter(self))])

    def rearrange(self, indices: List[int] | np.ndarray):
        for key in self.keys():
            self[key] = self[key][indices]

    def shuffle(self):
        self.rearrange(np.random.permutation(self.size()))

    def save(self, save_dir: str, overwrite: bool = True):
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, "buffer.jsonl")
        print(f"Saving buffer to {save_file} ......")
        with open(save_file, 'w' if overwrite else 'a', encoding='utf-8') as writer:
            for i in range(self.size()):
                data = dict()
                for key in self.keys():
                    data[key] = self[key][i].tolist()
                writer.write(json.dumps(data, ensure_ascii=False) + '\n')
        print("Saving done!")

    @classmethod
    def load(cls, buffer_dir: str, start: int = 0, stop: int = None) -> "RolloutBuffer":
        buffer_file = os.path.join(buffer_dir, "buffer.jsonl")
        print(f"Loading buffer from {buffer_file} ......")
        kwargs = dict()
        with open(buffer_file, 'r', encoding="utf-8") as reader:
            for i, line in enumerate(reader):
                if stop is not None and stop <= i:
                    break
                if start <= i:
                    for key, value in json.loads(line).items():
                        if key not in kwargs:
                            kwargs[key] = []
                        kwargs[key].append(value)

        buffer = RolloutBuffer(**kwargs)
        print("Loading done!")
        return buffer

    def extend(self, rollout_buffer: "RolloutBuffer") -> "RolloutBuffer":
        if len(rollout_buffer) == 0:
            return self
        if len(self) == 0 or self.size() == 0:
            self.set(**rollout_buffer)
        else:
            for key in self.keys():
                self[key] = np.concatenate([self[key], rollout_buffer[key]], axis=0)
        return self

    def get(self, batch_size: int) -> Generator:
        indices = np.arange(self.size())
        start_idx = 0
        RolloutBufferSample = collections.namedtuple(
            "RolloutBufferSample", field_names=[key for key in self.keys()]
        )
        while start_idx < self.size():
            batch_indices = indices[start_idx: start_idx + batch_size]
            data = dict()
            for key in self.keys():
                data[key] = self[key][batch_indices]
            yield RolloutBufferSample(**data)
            start_idx += batch_size


class PolicyRolloutBuffer:
    def __init__(
            self,
            obs: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            values: np.ndarray,
            action_logits: np.ndarray,
            action_masks: np.ndarray,
            action_logprobs: np.ndarray,
            ref_action_logprobs: np.ndarray = None,
            gamma: float = 0.9,
            gae_lambda: float = 0.8,
            kl_coef: float = 0.1,
            mu: float = 0.0,
            reward_normalize: bool = True,
            reward_sub_mean: bool = False,
            use_last_token_reward: bool = False,
            last_token_reward_only: bool = False,
            reward_is_q: bool = False
    ):
        self.obs = None
        self.actions = None
        self.rewards = None
        self.values = None
        self.action_logits = None
        self.action_masks = None
        self.action_logprobs = None
        self.ref_action_logprobs = None
        self.advantages = None
        self.returns = None

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.kl_coef = kl_coef
        self.mu = mu
        self.reward_normalize = reward_normalize
        self.reward_sub_mean = reward_sub_mean
        self.use_last_token_reward = use_last_token_reward
        self.last_token_reward_only = last_token_reward_only
        self.reward_is_q = reward_is_q

        self.buffer_size = obs.shape[0]
        self.max_seq_len = obs.shape[1]

        self._set(obs, actions, rewards, values, action_logits, action_masks, action_logprobs, ref_action_logprobs)

        self.compute_returns_and_advantage()

    def __len__(self):
        return 0 if self.obs is None else self.obs.shape[0]

    def _set(self, obs, actions, rewards, values, action_logits, action_masks, action_logprobs, ref_action_logprobs=None):
        self.advantages = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)

        self.obs = obs.copy()
        self.actions = actions.copy()
        self.rewards = rewards.copy()
        self.values = values.copy()
        self.action_logits = action_logits.copy()
        self.action_masks = action_masks.copy()
        self.action_logprobs = action_logprobs.copy()
        self.origin_rewards = rewards.copy()  # for logging only
        self.origin_values = values.copy()  # for logging only
        if ref_action_logprobs is not None:
            self.ref_action_logprobs = ref_action_logprobs.copy()

        assert np.sum(self.rewards[~ self.action_masks]) == 0  # Check rewards correctness

        if self.use_last_token_reward:
            for i in range(self.buffer_size):
                nonzero_indices = np.nonzero(self.action_masks[i])[0]
                if len(nonzero_indices) > 0:
                    self.rewards[i][self.action_masks[i]] = self.rewards[i][nonzero_indices[-1]]

        if self.reward_normalize:
            if self.reward_sub_mean:
                self.rewards = (self.rewards - np.mean(
                    self.rewards[self.action_masks])) / np.std(self.rewards[self.action_masks])
            else:
                self.rewards = self.rewards / np.std(self.rewards[self.action_masks])

        if self.last_token_reward_only:
            for i in range(self.buffer_size):
                nonzero_indices = np.nonzero(self.action_masks[i])[0]
                if len(nonzero_indices) > 0:
                    score = self.rewards[i][nonzero_indices[-1]]
                    self.rewards[i] = 0.0
                    self.rewards[i][nonzero_indices[-1]] = score

        # Adding KL penalty
        self.rewards += - self.kl_coef * self.compute_kl_penalty()

        self.rewards[~ self.action_masks] = 0.0

    def compute_kl_penalty(self):
        if self.ref_action_logprobs is None:
            return np.zeros_like(self.action_logprobs)
        return np.abs(self.action_logprobs - self.ref_action_logprobs)  # using abs kl loss

    def compute_returns_and_advantage(self):
        if self.reward_is_q:
            if self.reward_normalize:
                if self.reward_sub_mean:
                    self.values = (self.values - np.mean(
                        self.values[self.action_masks])) / np.std(self.values[self.action_masks])
                else:
                    self.values = self.values / np.std(self.values[self.action_masks])
            self.advantages = self.rewards - (1 - self.gae_lambda) * self.values
            self.returns = self.rewards
        else:
            last_gae_lam = 0
            for step in reversed(range(self.max_seq_len - 1)):
                next_values = self.values[:, step + 1] * np.where(self.action_masks[:, step + 1], 1, 0)
                delta = self.rewards[:, step] + self.gamma * next_values - self.values[:, step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam * self.action_masks[:, step + 1]
                self.advantages[:, step] = last_gae_lam
            self.returns = self.advantages + self.values

    def get(self, batch_size: int, shuffle: bool = True) -> Generator[PolicyRolloutBufferSample, None, None]:
        size = self.obs.shape[0]
        indices = np.random.permutation(size) if shuffle else np.arange(size)
        start_idx = 0
        while start_idx < size:
            batch_indices = indices[start_idx: start_idx + batch_size]
            yield PolicyRolloutBufferSample(
                observations=torch.tensor(self.obs[batch_indices]),
                actions=torch.tensor(self.actions[batch_indices]),
                old_values=torch.tensor(self.values[batch_indices]),
                old_action_logits=torch.tensor(self.action_logits[batch_indices]),
                old_action_logprobs=torch.tensor(self.action_logprobs[batch_indices]),
                advantages=torch.tensor(self.advantages[batch_indices]),
                returns=torch.tensor(self.returns[batch_indices]),
                action_masks=torch.tensor(self.action_masks[batch_indices]),
                rewards=torch.tensor(self.rewards[batch_indices]),
                ref_action_logprobs=torch.tensor(self.ref_action_logprobs[batch_indices]) if (
                        self.ref_action_logprobs is not None
                ) else None
            )
            start_idx += batch_size


class LogitsRolloutBuffer:
    def __init__(
            self,
            instructions: Union[List[str], np.ndarray] = None,
            outputs: Union[List[str], np.ndarray] = None,
            logits: torch.Tensor = None,
            output_tokens_logps: Union[np.ndarray, torch.Tensor] = None,
            logits_topk: int = None
    ):
        self.ignore_logits = logits_topk is None or logits_topk <= 0
        self.logits = None
        self.instructions = None
        self.outputs = None
        self.output_tokens_logps = None

        self.__cache_logits = None

        if instructions is not None:
            assert logits is not None
            assert outputs is not None
            assert output_tokens_logps is not None
            logits = None if self.ignore_logits else SlimLogits(logits=logits, n=logits_topk)
            self._set(instructions, outputs, logits, output_tokens_logps)

    def save(self, save_dir: str, overwrite: bool = True, self_clean: bool = False):
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, "buffer.jsonl")
        print(f"Saving buffer to {save_file} ......")
        with open(save_file, 'w' if overwrite else 'a', encoding='utf-8') as writer:
            for i in range(len(self)):
                writer.write(json.dumps(dict(
                    instruction=self.instructions[i],
                    output=self.outputs[i],
                    logits={} if self.ignore_logits else self.logits[i].to_dict(),
                    output_tokens_logps=self.output_tokens_logps[i].tolist(),
                ), ensure_ascii=False) + '\n')
        if self_clean:
            self.__reset()
        print("Saving done!")

    def load(self, buffer_file: str, start: int = 0, stop: int = None) -> "LogitsRolloutBuffer":
        print(f"Loading buffer from {buffer_file} ......")
        self.instructions = []
        self.outputs = []
        self.logits = []
        self.output_tokens_logps = []
        with open(buffer_file, 'r', encoding="utf-8") as reader:
            for i, line in enumerate(reader):
                if stop is not None and stop <= i:
                    break
                if start <= i:
                    data = json.loads(line)
                    self.instructions.append(data['instruction'])
                    self.outputs.append(data['output'])
                    self.logits.append(data['logits'])
                    self.output_tokens_logps.append(data['output_tokens_logps'])
        self.instructions = np.array(self.instructions)
        self.outputs = np.array(self.outputs)
        self.logits = None if self.ignore_logits else SlimLogits().from_dict(self.logits)
        self.output_tokens_logps = np.array(self.output_tokens_logps)
        print("Loading done!")
        return self

    def __flush(self):
        if self.__cache_logits is not None:
            self.logits.extend(self.__cache_logits)
            self.__cache_logits = None

    def __reset(self):
        self.logits = None
        self.instructions = None
        self.outputs = None
        self.output_tokens_logps = None
        self.__cache_logits = None

    def __len__(self):
        self.__flush()
        if self.instructions is not None:
            assert len(self.instructions) == len(self.outputs)
            assert len(self.instructions) == len(self.output_tokens_logps)
            return len(self.instructions)
        return 0

    def _set(self, instructions, outputs, logits: SlimLogits, output_tokens_logps):
        if not isinstance(instructions, np.ndarray):
            instructions = np.array(instructions)
        if not isinstance(outputs, np.ndarray):
            outputs = np.array(outputs)
        if not isinstance(output_tokens_logps, np.ndarray):
            output_tokens_logps = output_tokens_logps.float().cpu().numpy()
        self.instructions = instructions
        self.outputs = outputs
        if not self.ignore_logits:
            self.logits = logits
        self.output_tokens_logps = output_tokens_logps

    def extend(self, rollout_buffer: "LogitsRolloutBuffer"):
        if self.instructions is None:
            self.ignore_logits = rollout_buffer.ignore_logits
            self._set(
                rollout_buffer.instructions,
                rollout_buffer.outputs,
                rollout_buffer.logits,
                rollout_buffer.output_tokens_logps
            )
        else:
            if len(rollout_buffer) > 0:
                self.instructions = np.concatenate([self.instructions, rollout_buffer.instructions], axis=0)
                self.outputs = np.concatenate([self.outputs, rollout_buffer.outputs], axis=0)
                self.output_tokens_logps = np.concatenate(
                    [self.output_tokens_logps, rollout_buffer.output_tokens_logps], axis=0)
                if not self.ignore_logits:  # Extremely slow, using cache
                    if self.__cache_logits is None:
                        self.__cache_logits = rollout_buffer.logits
                    else:
                        self.__cache_logits.extend(rollout_buffer.logits)
                if self.__cache_logits is not None and len(self.__cache_logits) > 1000:
                    self.__flush()

        return self

    def get_logps(self, batch_size: int) -> Generator[torch.Tensor, None, None]:
        """ Only fetching output_tokens_logps. """
        self.__flush()
        size = len(self)
        indices = np.arange(size)
        start_idx = 0
        while start_idx < size:
            batch_indices = indices[start_idx: start_idx + batch_size]
            yield torch.tensor(self.output_tokens_logps[batch_indices])
            start_idx += batch_size

    def get(self, batch_size: int) -> Generator[LogitsRolloutBufferSample, None, None]:
        self.__flush()
        size = len(self)
        indices = np.arange(size)
        start_idx = 0
        while start_idx < size:
            batch_indices = indices[start_idx: start_idx + batch_size]
            logits = None
            if not self.ignore_logits:
                logits = torch.zeros(
                    (len(batch_indices), self.logits.max_seq_len, self.logits.vocab_size), dtype=torch.float32
                )
                for i, bi in enumerate(batch_indices):
                    logits[i, :, :] = self.logits.fetch(bi)

            yield LogitsRolloutBufferSample(
                instructions=self.instructions[batch_indices].tolist(),
                outputs=self.outputs[batch_indices].tolist(),
                logits=logits,
                output_tokens_logps=torch.tensor(self.output_tokens_logps[batch_indices])
            )
            start_idx += batch_size


class CriticRolloutBuffer(RolloutBuffer):
    def __init__(self, scores: np.ndarray | List[float] = None, action_masks: np.ndarray = None):
        super().__init__()
        self.set(scores=scores, action_masks=action_masks)

    def set(self, scores=None, action_masks=None):
        if scores is not None and action_masks is not None:
            if isinstance(scores, np.ndarray) and scores.shape == action_masks.shape:
                self["scores"] = scores.copy()
            else:
                self["scores"] = np.zeros_like(action_masks, dtype=np.float32)
                for i in range(len(action_masks)):
                    self["scores"][i, :][action_masks[i]] = scores[i]
            self["action_masks"] = action_masks.copy()

    def mean(self, use_last_token_reward: bool) -> float:
        if use_last_token_reward:
            rewards = []
            for i in range(self.size()):
                nonzero_indices = np.nonzero(self["action_masks"][i])[0]
                if len(nonzero_indices) > 0:
                    rewards.append(self["scores"][i][nonzero_indices][-1].item())
            return np.mean(rewards)
        else:
            return masked_mean(self["scores"], self["action_masks"])

    @classmethod
    def load(cls, buffer_dir: str, start: int = 0, stop: int = None) -> "CriticRolloutBuffer":
        return CriticRolloutBuffer(**super().load(buffer_dir=buffer_dir, start=start, stop=stop))
