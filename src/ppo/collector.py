from typing import List, Union

import numpy as np

from src.models.modeling import ModelForCausalLM, ParallelModelForCausalLM, Verifier, ParallelVerifier
from src.ppo.buffer import (
    CriticRolloutBuffer,
    LogitsRolloutBuffer,
    RolloutBuffer
)
from src.ppo.generator import (
    CriticGeneratorForCausalLM,
    ActorGeneratorForCausalLM,
    LogitsGeneratorForCausalLM,
    ActorForwardGeneratorForCausalLM
)
from src.tokenizers.tokenizer import Tokenizer


class ActorBufferCollector:
    def __init__(
            self,
            actor: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int,
            temperature: float = 1.0,
            top_p: float = 1.0

    ):
        self.generator = ActorGeneratorForCausalLM(
            model=actor, tokenizer=tokenizer, max_seq_len=max_seq_len, temperature=temperature, top_p=top_p
        )

    def forward(self, instructions: List[str]) -> RolloutBuffer:
        generator_outputs = self.generator.forward(instructions)
        return RolloutBuffer(
            instructions=instructions,
            obs=generator_outputs.obs.cpu().numpy(),
            actions=generator_outputs.actions.cpu().numpy(),
            action_logits=generator_outputs.action_logits.float().cpu().numpy(),
            action_masks=generator_outputs.action_masks.cpu().numpy(),
            action_logprobs=generator_outputs.action_logprobs.float().cpu().numpy(),
            responses=generator_outputs.responses
        )


class ActorForwardBufferCollector:
    def __init__(
            self,
            actor: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int
    ):
        self.generator = ActorForwardGeneratorForCausalLM(
            model=actor, tokenizer=tokenizer, max_seq_len=max_seq_len
        )

    def forward(self, instructions: List[str], responses: List[str]) -> RolloutBuffer:
        generator_outputs = self.generator.forward(instructions, responses)
        return RolloutBuffer(
            instructions=instructions,
            obs=generator_outputs.obs.cpu().numpy(),
            actions=generator_outputs.actions.cpu().numpy(),
            action_logits=generator_outputs.action_logits.float().cpu().numpy(),
            action_masks=generator_outputs.action_masks.cpu().numpy(),
            action_logprobs=generator_outputs.action_logprobs.float().cpu().numpy(),
            responses=responses
        )


# class ActorGroupBufferCollector:
#     def __init__(
#             self,
#             actor: Union[ModelForCausalLM, ParallelModelForCausalLM],
#             tokenizer: Tokenizer,
#             max_seq_len: int,
#             temperature: float = 1.0,
#             top_p: float = 1.0,
#             num_samples_per_prompt: int = 1,
#             diverse_prob: float = None
#     ):
#         self.generator = ActorGroupGeneratorForCausalLM(
#             model=actor,
#             tokenizer=tokenizer,
#             max_seq_len=max_seq_len,
#             temperature=temperature,
#             top_p=top_p,
#             num_samples_per_prompt=num_samples_per_prompt,
#             diverse_prob=diverse_prob
#         )
#         self.num_samples_per_prompt = num_samples_per_prompt
#
#     def forward(self, instructions: List[str]):
#         generator_outputs = self.generator.forward(instructions)
#         obs = generator_outputs.obs.float().cpu().numpy()
#         actions = generator_outputs.actions.float().cpu().numpy()
#         action_logits = generator_outputs.action_logits.float().cpu().numpy()
#         action_masks = generator_outputs.action_masks.float().cpu().numpy()
#         action_logprobs = generator_outputs.action_logprobs.float().cpu().numpy()
#         responses = generator_outputs.responses
#         instructions = [instruction for instruction in instructions for _ in range(self.num_samples_per_prompt)]
#         return ActorRolloutBuffer(
#             instructions=instructions,
#             obs=obs,
#             actions=actions,
#             action_logits=action_logits,
#             action_masks=action_masks,
#             action_logprobs=action_logprobs,
#             responses=responses
#         )


class CriticBufferCollector:
    def __init__(self, critic: Union[Verifier, ParallelVerifier], tokenizer: Tokenizer, max_seq_len: int):
        self.generator = CriticGeneratorForCausalLM(
            verifier=critic, tokenizer=tokenizer, max_seq_len=max_seq_len
        )

    def forward(self, instructions: List[str], actions: np.ndarray, action_masks: np.ndarray) -> CriticRolloutBuffer:
        generator_outputs = self.generator.forward(
            obs=instructions,
            actions=actions,
            action_masks=action_masks
        )
        return CriticRolloutBuffer(scores=generator_outputs.token_scores, action_masks=action_masks)


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
