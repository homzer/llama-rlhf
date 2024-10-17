from typing import List, Union

import numpy as np

from src.dataset import JsonDataset
from src.evaluator import GSM8KEvaluator
from src.models.modeling import ModelForCausalLM, ParallelModelForCausalLM, Verifier, ParallelVerifier
from src.ppo.buffer import (
    CriticRolloutBuffer,
    ActorRolloutBuffer,
    LogitsRolloutBuffer,
    OutputRolloutBuffer)
from src.ppo.generator import (
    CriticGeneratorForCausalLM,
    ActorGeneratorForCausalLM,
    LogitsGeneratorForCausalLM)
from src.tokenizers.tokenizer import Tokenizer


class OutputBufferCollector:
    def __init__(
            self,
            solver: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int,
            temperature: float = 0.0,
            top_p: float = 0.95
    ):
        self.generator = ActorGeneratorForCausalLM(
            model=solver, tokenizer=tokenizer, max_seq_len=max_seq_len, temperature=temperature, top_p=top_p
        )

    def forward(self, instructions: List[str]) -> OutputRolloutBuffer:
        outputs = self.generator.forward(instructions)
        return OutputRolloutBuffer(instructions, outputs.responses)


class ActorBufferCollector:
    def __init__(
            self,
            actor: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int,
            temperature: float = 0.0,
            top_p: float = 0.95

    ):
        self.generator = ActorGeneratorForCausalLM(
            model=actor, tokenizer=tokenizer, max_seq_len=max_seq_len, temperature=temperature, top_p=top_p
        )

    def forward(self, instructions: List[str]) -> ActorRolloutBuffer:
        generator_outputs = self.generator.forward(instructions)
        obs = generator_outputs.obs.float().cpu().numpy()
        actions = generator_outputs.actions.float().cpu().numpy()
        action_logits = generator_outputs.action_logits.float().cpu().numpy()
        action_masks = generator_outputs.action_masks.float().cpu().numpy()
        action_logprobs = generator_outputs.action_logprobs.float().cpu().numpy()
        responses = generator_outputs.responses
        return ActorRolloutBuffer(instructions, obs, actions, action_logits, action_masks, action_logprobs, responses)


class CriticBufferCollector:
    def __init__(self, critic: Union[Verifier, ParallelVerifier], tokenizer: Tokenizer, max_seq_len: int):
        self.generator = CriticGeneratorForCausalLM(
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
    def __init__(self, task: str, dataset: JsonDataset, tokenizer: Tokenizer, max_seq_len: int):
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

        scores = []
        for i, (instruction, output) in enumerate(zip(instructions, outputs)):
            answers = self.evaluator.forward(output)
            if self.evaluator.format_label(self.map[instruction]) not in answers[-1:]:
                scores.append(0)
            else:
                scores.append(1)

        return CriticRolloutBuffer(scores, action_masks)


class LogitsBufferCollector:
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int,
            logits_topk: int = None
    ):
        self.logits_topk = logits_topk
        self.generator = LogitsGeneratorForCausalLM(
            model=model, tokenizer=tokenizer, max_seq_len=max_seq_len
        )

    def forward(self, instructions: List[str], responses: List[str]) -> LogitsRolloutBuffer:
        assert len(instructions) == len(responses)
        generator_outputs = self.generator.forward(instructions, responses)
        return LogitsRolloutBuffer(
            instructions=instructions,
            outputs=responses,
            logits=generator_outputs.logits,
            output_tokens_logps=generator_outputs.tokens_logps,
            logits_topk=self.logits_topk
        )
