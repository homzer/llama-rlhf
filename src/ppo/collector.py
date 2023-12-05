from typing import List, Union

import numpy as np
import torch

from src.dataset import JsonDataset
from src.evaluator import GSM8KEvaluator
from src.modeling.llama_lora import LoraLlamaVerifier
from src.modeling.modeling import ModelForCausalLM, ParallelModelForCausalLM
from src.ppo.buffer import (
    RolloutBuffer,
    PolicyRolloutBuffer,
    CriticRolloutBuffer,
    ActorRolloutBuffer
)
from src.ppo.generator import CriticismGeneratorForCausalLM, ActionGeneratorForCausalLM
from src.ppo.policy import AbstractPolicyForCausalLM, AbstractParallelPolicyForCausalLM
from src.tokenizer import Tokenizer, LlamaTokenizer


class BufferCollector:
    def __init__(
            self,
            generator: CriticismGeneratorForCausalLM,
            policy: Union[AbstractPolicyForCausalLM, AbstractParallelPolicyForCausalLM],
            buffer_size: int,
            max_seq_len: int,
    ):
        self.generator = generator
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
        action_rewards = self.generator.forward(instructions, actions, action_masks)
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

        return PolicyRolloutBuffer(instructions, obs, actions, values, action_logits, action_masks)


class ActorBufferCollector:
    def __init__(
            self,
            actor: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int
    ):
        self.generator = ActionGeneratorForCausalLM(
            model=actor, tokenizer=tokenizer, max_seq_len=max_seq_len
        )

    def forward(self, instructions: List[str]) -> ActorRolloutBuffer:
        outputs = self.generator.forward(instructions)
        obs = outputs.obs.cpu().numpy()
        actions = outputs.actions.cpu().numpy()
        action_logits = outputs.action_logits.cpu().numpy()
        action_masks = outputs.action_masks.cpu().numpy()
        return ActorRolloutBuffer(instructions, obs, actions, action_logits, action_masks)


class CriticBufferCollector:
    def __init__(self, critic: LoraLlamaVerifier, tokenizer: LlamaTokenizer, max_seq_len: int):
        self.generator = CriticismGeneratorForCausalLM(
            verifier=critic, tokenizer=tokenizer, max_seq_len=max_seq_len
        )

    def forward(self, instructions: np.ndarray, actions: np.ndarray, action_masks: np.ndarray) -> CriticRolloutBuffer:
        action_scores = self.generator.forward(
            obs=instructions.tolist(),
            actions=actions,
            action_masks=action_masks
        )
        return CriticRolloutBuffer(action_scores, action_masks)


class LabelBufferCollector:
    def __init__(self, task: str, dataset: JsonDataset, tokenizer: LlamaTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.evaluator = {
            "GSM8K": GSM8KEvaluator,
        }[task]()
        self.map = {}
        assert "instruction" in dataset[0].keys()
        assert "label" in dataset[0].keys()
        for data in dataset:
            self.map[data['instruction']] = data['label']

    def forward(self, instructions: np.ndarray, actions: np.ndarray, action_masks: np.ndarray) -> CriticRolloutBuffer:
        instructions = instructions.tolist()
        outputs = []
        for action, action_mask in zip(actions, action_masks):
            outputs.append(self.tokenizer.decode(action[action_mask].tolist()))

        scores = np.ones_like(action_masks)
        for i, (instruction, output) in enumerate(zip(instructions, outputs)):
            answers = self.evaluator.forward(output)
            if self.evaluator.format_label(self.map[instruction]) not in answers[-1:]:
                scores[i] *= 0

        return CriticRolloutBuffer(scores, action_masks)
