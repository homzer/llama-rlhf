import collections
import copy
import json
import os.path
from typing import Generator, List, Union

import numpy as np
import torch

from src.entities import SlimLogits

RolloutBufferSample = collections.namedtuple(
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
        "scores"
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


class RolloutBuffer:
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
            reward_normalize: bool = True
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
        self.reward_normalize = reward_normalize

        self.buffer_size = obs.shape[0]
        self.max_seq_len = obs.shape[1]

        self._set(obs, actions, rewards, values, action_logits, action_masks, action_logprobs, ref_action_logprobs)

        self.compute_returns_and_advantage()

    def __len__(self):
        return 0 if self.obs is None else self.obs.shape[0]

    def _set(self, obs, actions, rewards, values, action_logits, action_masks, action_logprobs, action_ref_logprobs=None):
        self.obs = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.int64)
        self.actions = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)
        self.action_logits = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)
        self.action_logprobs = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)
        self.action_masks = np.zeros((self.buffer_size, self.max_seq_len), dtype=bool)
        self.advantages = np.zeros((self.buffer_size, self.max_seq_len), dtype=np.float32)

        self.obs[:, :] = obs.copy()
        self.actions[:, :] = actions.copy()
        self.rewards[:, :] = rewards.copy()
        self.values[:, :] = values.copy()
        self.action_logits[:, :] = action_logits.copy()
        self.action_masks[:, :] = action_masks.copy()
        self.action_logprobs[:, :] = action_logprobs.copy()
        if action_ref_logprobs is not None:
            self.ref_action_logprobs[:, :] = action_ref_logprobs.copy()

        assert np.sum(self.rewards[~ self.action_masks]) == 0  # Check rewards correctness

        if self.reward_normalize:
            # Normalize rewards
            self.rewards = (self.rewards - np.mean(
                self.rewards[self.action_masks])) / (np.std(self.rewards[self.action_masks]) + 1e-12)
        # Adding KL penalty
        self.rewards += - self.kl_coef * self.compute_kl_penalty()

        self.rewards[~ self.action_masks] = 0.0

    def compute_kl_penalty(self):
        if self.ref_action_logprobs is None:
            return np.zeros_like(self.action_logprobs)
        return 0.5 * (self.action_logprobs - self.ref_action_logprobs) ** 2  # using mse loss

    def compute_returns_and_advantage(self):
        last_gae_lam = 0
        for step in reversed(range(self.max_seq_len - 1)):
            next_values = self.values[:, step + 1] * np.where(self.action_masks[:, step + 1], 1, 0)
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
                old_action_logprobs=torch.tensor(self.action_logprobs[batch_indices]),
                advantages=torch.tensor(self.advantages[batch_indices]),
                returns=torch.tensor(self.returns[batch_indices]),
                action_masks=torch.tensor(self.action_masks[batch_indices]),
                rewards=torch.tensor(self.rewards[batch_indices]),
                ref_action_logprobs=torch.tensor(self.ref_action_logprobs[batch_indices])
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
        self.__cache_instructions = None
        self.__cache_outputs = None
        self.__cache_output_tokens_logps = None

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
        if self.__cache_instructions is not None:
            assert self.__cache_outputs is not None
            if not self.ignore_logits:
                assert self.__cache_logits is not None
            assert self.__cache_output_tokens_logps is not None
            self.instructions = np.stack([*self.instructions, *self.__cache_instructions], axis=0)
            self.outputs = np.stack([*self.outputs, *self.__cache_outputs], axis=0)
            self.output_tokens_logps = np.stack(
                [*self.output_tokens_logps, *self.__cache_output_tokens_logps], axis=0
            )
            if not self.ignore_logits:
                self.logits.extend(self.__cache_logits)
            self.__cache_logits = None
            self.__cache_instructions = None
            self.__cache_outputs = None
            self.__cache_output_tokens_logps = None
        else:
            assert self.__cache_outputs is None
            if not self.ignore_logits:
                assert self.__cache_logits is None
            assert self.__cache_output_tokens_logps is None

    def __reset(self):
        self.logits = None
        self.instructions = None
        self.outputs = None
        self.output_tokens_logps = None
        self.__cache_logits = None
        self.__cache_instructions = None
        self.__cache_outputs = None
        self.__cache_output_tokens_logps = None

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
        if len(self) == 0:
            self.ignore_logits = rollout_buffer.ignore_logits
            self._set(
                rollout_buffer.instructions,
                rollout_buffer.outputs,
                rollout_buffer.logits,
                rollout_buffer.output_tokens_logps
            )
        else:
            if len(rollout_buffer) > 0:  # TODO Extremely slow, using cache
                if self.__cache_instructions is None:
                    self.__cache_instructions = rollout_buffer.instructions
                    self.__cache_outputs = rollout_buffer.outputs
                    if not self.ignore_logits:
                        self.__cache_logits = rollout_buffer.logits
                    self.__cache_output_tokens_logps = rollout_buffer.output_tokens_logps
                else:
                    self.__cache_instructions = np.stack(
                        [*self.__cache_instructions, *rollout_buffer.instructions], axis=0
                    )
                    self.__cache_outputs = np.stack(
                        [*self.__cache_instructions, *rollout_buffer.outputs], axis=0
                    )
                    self.__cache_output_tokens_logps = np.stack(
                        [*self.__cache_output_tokens_logps, *rollout_buffer.output_tokens_logps], axis=0
                    )
                    if not self.ignore_logits:
                        self.__cache_logits.extend(rollout_buffer.logits)
                if len(self.__cache_instructions) > 1000:
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
                instructions=self.instructions[batch_indices],
                outputs=self.outputs[batch_indices],
                logits=logits,
                output_tokens_logps=torch.tensor(self.output_tokens_logps[batch_indices])
            )
            start_idx += batch_size


class SolverRolloutBuffer:
    def __init__(
            self,
            instructions: Union[List[str], np.ndarray] = None,
            actions: np.ndarray = None,
            action_masks: np.ndarray = None
    ):
        self.instructions = None
        self.actions = None
        self.action_masks = None

        if instructions is not None:
            assert actions is not None
            assert action_masks is not None
            self._set(instructions, actions, action_masks)

    def __len__(self):
        return 0 if self.instructions is None else len(self.instructions)

    def copy(self):
        return copy.deepcopy(self)

    def pop(self, index: int) -> SolverRolloutBufferSample:
        if index >= len(self):
            raise IndexError(f"pop index {index} out of range {len(self)}")
        buffer_sample = SolverRolloutBufferSample(
            instructions=self.instructions[index: index + 1],
            actions=self.actions[index: index + 1],
            action_masks=self.action_masks[index: index + 1]
        )
        self.instructions = np.concatenate([self.instructions[: index], self.instructions[index + 1:]], axis=0)
        self.actions = np.concatenate([self.actions[: index], self.actions[index + 1:]], axis=0)
        self.action_masks = np.concatenate([self.action_masks[: index], self.action_masks[index + 1:]], axis=0)
        return buffer_sample

    def _set(self, instructions, actions, action_masks):
        assert len(instructions) == len(actions)
        assert len(instructions) == len(action_masks)
        buffer_size = actions.shape[0]
        max_seq_len = actions.shape[1]

        self.actions = np.zeros((buffer_size, max_seq_len), dtype=np.int64)
        self.action_masks = np.zeros((buffer_size, max_seq_len), dtype=bool)

        if not isinstance(instructions, np.ndarray):
            instructions = np.array(instructions)
        self.instructions = instructions
        self.actions[:, :] = actions.copy()
        self.action_masks[:, :] = action_masks.copy()

    def extend(self, rollout_buffer: "SolverRolloutBuffer"):
        if len(self) == 0:
            self._set(
                rollout_buffer.instructions,
                rollout_buffer.actions,
                rollout_buffer.action_masks
            )
        else:
            self.instructions = np.concatenate([self.instructions, rollout_buffer.instructions], axis=0)
            self.actions = np.concatenate([self.actions, rollout_buffer.actions], axis=0)
            self.action_masks = np.concatenate([self.action_masks, rollout_buffer.action_masks], axis=0)

        return self

    def get(self, batch_size: int) -> Generator[SolverRolloutBufferSample, None, None]:
        size = self.actions.shape[0]
        indices = np.arange(size)
        start_idx = 0
        while start_idx < size:
            batch_indices = indices[start_idx: start_idx + batch_size]
            yield SolverRolloutBufferSample(
                instructions=self.instructions[batch_indices],
                actions=self.actions[batch_indices],
                action_masks=self.action_masks[batch_indices]
            )
            start_idx += batch_size


class OutputRolloutBuffer:
    def __init__(
            self,
            instructions: Union[List[str], np.ndarray] = None,
            outputs: Union[List[str], np.ndarray] = None,
    ):
        self.instructions = None
        self.outputs = None

        if instructions is not None:
            assert outputs is not None
            self._set(instructions, outputs)

    def __len__(self):
        return 0 if self.instructions is None else len(self.instructions)

    def _set(self, instructions, outputs):
        assert len(instructions) == len(outputs)
        if not isinstance(instructions, np.ndarray):
            instructions = np.array(instructions)
        if not isinstance(outputs, np.ndarray):
            outputs = np.array(outputs)
        self.instructions = instructions
        self.outputs = outputs

    def extend(self, rollout_buffer: "OutputRolloutBuffer"):
        if len(self) == 0:
            self._set(
                rollout_buffer.instructions,
                rollout_buffer.outputs,
            )
        else:
            self.instructions = np.concatenate([self.instructions, rollout_buffer.instructions], axis=0)
            self.outputs = np.concatenate([self.outputs, rollout_buffer.outputs], axis=0)

        return self

    def get(self, batch_size: int) -> Generator[OutputRolloutBufferSample, None, None]:
        size = self.outputs.shape[0]
        indices = np.arange(size)
        start_idx = 0
        while start_idx < size:
            batch_indices = indices[start_idx: start_idx + batch_size]
            yield OutputRolloutBufferSample(
                instructions=self.instructions[batch_indices],
                outputs=self.outputs[batch_indices],
            )
            start_idx += batch_size


class ActorRolloutBuffer:
    def __init__(
            self,
            instructions: List[str] = None,
            obs: np.ndarray = None,
            actions: np.ndarray = None,
            action_logits: np.ndarray = None,
            action_masks: np.ndarray = None,
            action_logprobs: np.ndarray = None,
            responses: List[str] = None
    ):
        self.instructions = None
        self.obs = None
        self.actions = None
        self.action_logits = None
        self.action_masks = None
        self.action_logprobs = None
        self.responses = None

        if obs is not None:
            self._set(instructions, obs, actions, action_logits, action_masks, action_logprobs, responses)

    def __len__(self):
        return 0 if self.instructions is None else len(self.instructions)

    def _set(self, instructions, obs, actions, action_logits, action_masks, action_logprobs, responses):
        buffer_size = obs.shape[0]
        max_seq_len = obs.shape[1]

        self.obs = np.zeros((buffer_size, max_seq_len), dtype=np.int64)
        self.actions = np.zeros((buffer_size, max_seq_len), dtype=np.int64)
        self.action_logits = np.zeros((buffer_size, max_seq_len), dtype=np.float32)
        self.action_masks = np.zeros((buffer_size, max_seq_len), dtype=bool)
        self.action_logprobs = np.zeros((buffer_size, max_seq_len), dtype=np.float32)

        self.instructions = np.array(instructions)
        self.obs[:, :] = obs.copy()
        self.actions[:, :] = actions.copy()
        self.action_logits[:, :] = action_logits.copy()
        self.action_masks[:, :] = action_masks.copy()
        self.action_logprobs[:, :] = action_logprobs.copy()
        self.responses = np.array(responses)

    def extend(self, rollout_buffer: "ActorRolloutBuffer"):
        if self.obs is None:
            self._set(
                rollout_buffer.instructions,
                rollout_buffer.obs,
                rollout_buffer.actions,
                rollout_buffer.action_logits,
                rollout_buffer.action_masks,
                rollout_buffer.action_logprobs,
                rollout_buffer.responses
            )
        else:
            self.instructions = np.concatenate([self.instructions, rollout_buffer.instructions], axis=0)
            self.obs = np.concatenate([self.obs, rollout_buffer.obs], axis=0)
            self.actions = np.concatenate([self.actions, rollout_buffer.actions], axis=0)
            self.action_logits = np.concatenate([self.action_logits, rollout_buffer.action_logits], axis=0)
            self.action_masks = np.concatenate([self.action_masks, rollout_buffer.action_masks], axis=0)
            self.action_logprobs = np.concatenate([self.action_logprobs, rollout_buffer.action_logprobs], axis=0)
            self.responses = np.concatenate([self.responses, rollout_buffer.responses], axis=0)

        return self

    def get(self, batch_size: int) -> Generator[ActorRolloutBufferSample, None, None]:
        size = self.obs.shape[0]
        indices = np.arange(size)
        start_idx = 0
        while start_idx < size:
            batch_indices = indices[start_idx: start_idx + batch_size]
            yield ActorRolloutBufferSample(
                instructions=self.instructions[batch_indices],
                obs=self.obs[batch_indices],
                actions=self.actions[batch_indices],
                action_logits=self.action_logits[batch_indices],
                action_masks=self.action_masks[batch_indices],
                action_logprobs=self.action_logprobs[batch_indices],
                responses=self.responses[batch_indices]
            )
            start_idx += batch_size


class CriticRolloutBuffer:
    def __init__(self, scores: Union[np.ndarray, List] = None, action_masks: np.ndarray = None):
        self.scores = None
        if scores is not None:
            self._set(scores, action_masks)

    def __len__(self):
        return len(self.scores) if self.scores is not None else 0

    def _set(self, scores, action_masks=None):
        if action_masks is None:
            if isinstance(scores, list):
                scores = np.array(scores)
            assert isinstance(scores, np.ndarray)
            self.scores = scores.copy()
        else:
            buffer_size = action_masks.shape[0]
            max_seq_len = action_masks.shape[1]
            self.scores = np.zeros((buffer_size, max_seq_len), dtype=np.float32)
            for i in range(buffer_size):
                self.scores[i, :][action_masks[i]] = scores[i]

    def extend(self, rollout_buffer: "CriticRolloutBuffer"):
        if self.scores is None:
            self._set(rollout_buffer.scores)
        else:
            self.scores = np.concatenate([self.scores, rollout_buffer.scores], axis=0)
        return self

    def get(self, batch_size: int) -> Generator[CriticRolloutBufferSample, None, None]:
        size = len(self)
        indices = np.arange(size)
        start_idx = 0
        while start_idx < size:
            batch_indices = indices[start_idx: start_idx + batch_size]
            yield CriticRolloutBufferSample(scores=self.scores[batch_indices])
            start_idx += batch_size
