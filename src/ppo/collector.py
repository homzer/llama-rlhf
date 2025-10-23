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
    ActorForwardGeneratorForCausalLM,
    ActorPrefixGeneratorForCausalLM
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
            responses=responses,
            output_actions=generator_outputs.output_actions.cpu().numpy()
        )


class ActorPrefixBufferCollector:
    def __init__(
            self,
            actor: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int,
            temperature: float = 1.0,
            top_p: float = 1.0
    ):
        self.generator = ActorPrefixGeneratorForCausalLM(
            model=actor, tokenizer=tokenizer, max_seq_len=max_seq_len, temperature=temperature, top_p=top_p
        )

    def forward(self, instructions: List[str], prefixes: List[str]) -> RolloutBuffer:
        generator_outputs = self.generator.forward(instructions, prefixes)
        return RolloutBuffer(
            instructions=instructions,
            obs=generator_outputs.obs.cpu().numpy(),
            actions=generator_outputs.actions.cpu().numpy(),
            action_logits=generator_outputs.action_logits.float().cpu().numpy(),
            action_masks=generator_outputs.action_masks.cpu().numpy(),
            action_logprobs=generator_outputs.action_logprobs.float().cpu().numpy(),
            prefix_masks=generator_outputs.prefix_masks.cpu().numpy(),
            responses=generator_outputs.responses
        )


class CriticBufferCollector:
    def __init__(
            self,
            critic: Union[Verifier, ParallelVerifier],
            tokenizer: Tokenizer,
            max_seq_len: int,
            use_last_token_reward: bool = False,
            last_token_reward_only: bool = False
    ):
        self.use_last_token_reward = use_last_token_reward
        self.last_token_reward_only = last_token_reward_only
        self.generator = CriticGeneratorForCausalLM(
            verifier=critic, tokenizer=tokenizer, max_seq_len=max_seq_len
        )

    def forward(self, instructions: List[str], actions: np.ndarray, action_masks: np.ndarray) -> CriticRolloutBuffer:
        generator_outputs = self.generator.forward(
            obs=instructions,
            actions=actions,
            action_masks=action_masks
        )
        return CriticRolloutBuffer(
            scores=generator_outputs.token_scores,
            action_masks=action_masks,
            use_last_token_reward=self.use_last_token_reward,
            last_token_reward_only=self.last_token_reward_only
        )


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
