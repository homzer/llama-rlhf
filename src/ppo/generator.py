import collections
from typing import List, Union

import numpy as np
import torch

from src.generator import GeneratorForVerifier, GeneratorForCausalLM, PrefixGeneratorForCausalLM, \
    ForwardGeneratorForCausalLM
from src.models.modeling import ModelForCausalLM, ParallelModelForCausalLM, ParallelVerifier, Verifier
from src.tokenizers import Tokenizer
from src.trainer import prepare_for_forward

ActorGeneratorOutputs = collections.namedtuple("ActorGeneratorOutputs", [
    'responses', 'obs', 'actions', 'action_logits', 'action_masks', 'action_logprobs'
])

ActorForwardGeneratorOutputs = collections.namedtuple("ActorGeneratorOutputs", [
    'responses', 'obs', 'actions', 'action_logits', 'action_masks', 'action_logprobs', 'output_actions'
])

ActorPrefixGeneratorOutputs = collections.namedtuple("ActorPrefixGeneratorOutputs", [
    'responses', 'obs', 'actions', 'action_logits', 'action_masks', 'action_logprobs', 'prefix_masks'
])

ActorLogitsGeneratorOutputs = collections.namedtuple("ActorLogitsGeneratorOutputs", [
    'logits', 'obs', 'actions', 'action_masks', 'action_logprobs'
])

CriticGeneratorOutputs = collections.namedtuple("CriticGeneratorOutputs", [
    "token_scores"
])

LogitsGeneratorOutputs = collections.namedtuple("LogitsGeneratorOutputs", [
    'logits',
    'tokens_logps'
])


class ActorGeneratorForCausalLM(GeneratorForCausalLM):
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int,
            temperature: float = 0.0,
            top_p: float = 0.95
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            temperature=temperature,
            top_p=top_p
        )

    def forward(self, instructions: List[str] | List[List[int]]) -> ActorGeneratorOutputs:
        self.model.eval()
        prep_outputs = self.prepare_for_generation(instructions)
        forward_outputs = self.model_forward(
            prep_outputs.tokens, prep_outputs.input_masks, prep_outputs.start_pos
        )
        output_masks = self.get_output_masks(forward_outputs.tokens, prep_outputs.input_masks)
        responses = self.decode_response(forward_outputs.tokens, output_masks)
        # input tokens shift left to get output tokens
        output_tokens = torch.zeros_like(forward_outputs.tokens)
        output_tokens[:, :-1] = forward_outputs.tokens[:, 1:]
        return ActorGeneratorOutputs(
            responses=responses,
            obs=forward_outputs.tokens,
            actions=output_tokens,
            action_logits=forward_outputs.tokens_logits,
            action_masks=output_masks,
            action_logprobs=forward_outputs.tokens_logprobs
        )


class ActorForwardGeneratorForCausalLM(ForwardGeneratorForCausalLM):
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int,
    ):
        super().__init__(model, tokenizer=tokenizer, max_seq_len=max_seq_len)

    def forward(
            self, instructions: List[str] | List[List[int]], responses: List[str] | List[List[int]]
    ) -> ActorForwardGeneratorOutputs:
        self.model.eval()

        prep_outputs = self.prepare_for_forward(instructions, responses)
        forward_outputs = self.model_forward(prep_outputs.tokens, prep_outputs.labels)

        return ActorForwardGeneratorOutputs(
            responses=responses,
            obs=prep_outputs.tokens,
            actions=prep_outputs.labels,
            action_logits=forward_outputs.output_token_logits,
            action_masks=prep_outputs.masks,
            action_logprobs=forward_outputs.output_token_logprobs,
            output_actions=forward_outputs.output_tokens
        )


class ActorLogitsGeneratorForCausalLM(ForwardGeneratorForCausalLM):
    def __init__(
            self,
            model: ModelForCausalLM | ParallelModelForCausalLM,
            tokenizer: Tokenizer,
            max_seq_len: int
    ):
        super().__init__(model, tokenizer=tokenizer, max_seq_len=max_seq_len)

    def model_forward(self, tokens, labels):
        tokens = tokens.to(self.model.device())
        labels = labels.to(self.model.device())
        with torch.no_grad():
            outputs = self.model.forward(tokens)
        output_token_logprobs = torch.gather(
            torch.log_softmax(outputs.logits, dim=-1), dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        Outputs = collections.namedtuple("Outputs", ['logits', 'tokens', 'output_token_logprobs'])
        return Outputs(logits=outputs.logits, tokens=tokens, output_token_logprobs=output_token_logprobs)

    def forward(
            self, instructions: List[str] | List[List[int]], responses: List[str] | List[List[int]]
    ) -> ActorLogitsGeneratorOutputs:
        self.model.eval()

        prep_outputs = self.prepare_for_forward(instructions, responses)
        forward_outputs = self.model_forward(prep_outputs.tokens, prep_outputs.labels)

        return ActorLogitsGeneratorOutputs(
            logits=forward_outputs.logits,
            obs=forward_outputs.tokens,
            actions=prep_outputs.labels,
            action_masks=prep_outputs.masks,
            action_logprobs=forward_outputs.output_token_logprobs
        )


class ActorPrefixGeneratorForCausalLM(PrefixGeneratorForCausalLM):
    def __init__(
            self,
            model: ModelForCausalLM | ParallelModelForCausalLM,
            tokenizer: Tokenizer,
            max_seq_len: int,
            temperature: float = 0.0,
            top_p: float = 1.0
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            temperature=temperature,
            top_p=top_p
        )

    def forward(
            self, instructions: List[str] | List[List[int]], prefixes: List[str] | List[List[int]]
    ) -> ActorPrefixGeneratorOutputs:
        self.model.eval()
        prep_outputs = self.prepare_for_generation(instructions, prefixes)
        forward_outputs = self.model_forward(
            prep_outputs.tokens, prep_outputs.input_masks, prep_outputs.start_pos
        )
        output_masks = self.get_output_masks(forward_outputs.tokens, prep_outputs.input_masks)
        responses = self.decode_response(forward_outputs.tokens, output_masks)
        # input tokens shift left to get output tokens
        output_tokens = torch.zeros_like(forward_outputs.tokens)
        output_tokens[:, :-1] = forward_outputs.tokens[:, 1:]
        # shift left to get prefix masks
        prefix_masks = torch.full_like(prep_outputs.prefix_masks, fill_value=False)
        prefix_masks[:, :-1] = prep_outputs.prefix_masks[:, 1:]
        return ActorPrefixGeneratorOutputs(
            responses=responses,
            obs=forward_outputs.tokens,
            actions=output_tokens,
            action_masks=output_masks,
            action_logits=forward_outputs.tokens_logits,
            action_logprobs=forward_outputs.tokens_logprobs,
            prefix_masks=prefix_masks
        )


class CriticGeneratorForCausalLM:
    def __init__(self, verifier: Union[Verifier, ParallelVerifier], tokenizer: Tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.generator = GeneratorForVerifier(verifier, tokenizer, max_seq_len)

    def forward(
            self,
            obs: List[str],
            actions: Union[np.ndarray, torch.Tensor],
            action_masks: Union[np.ndarray, torch.Tensor]
    ) -> CriticGeneratorOutputs:
        outputs = []
        for action, action_mask in zip(actions, action_masks):
            outputs.append(action[action_mask].tolist())
        token_scores = self.generator.forward(obs, outputs).tokens_scores
        return CriticGeneratorOutputs(token_scores=token_scores)


class LogitsGeneratorForCausalLMV0:
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int,
    ):
        self.model = model
        self.vocab_size = tokenizer.vocab_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def prepare_for_forward(
            self,
            instructions: Union[List[str], List[List[int]]],
            responses: Union[List[str], List[List[int]]]
    ):
        return prepare_for_forward(
            instructions=instructions,
            responses=responses,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len
        )

    def model_forward(self, tokens):
        with torch.no_grad():
            outputs = self.model.forward(tokens)

        Outputs = collections.namedtuple('Outputs', ['logits'])
        return Outputs(logits=outputs.logits)

    def forward(
            self,
            instructions: Union[List[str], List[List[int]]],
            responses: Union[List[str], List[List[int]]]
    ) -> LogitsGeneratorOutputs:
        self.model.eval()
        prep_outputs = self.prepare_for_forward(instructions, responses)
        logits = self.model_forward(prep_outputs.tokens).logits

        # retrieve token probs
        tokens_logps = torch.log_softmax(
            logits.float() if logits.dtype == torch.float16 else logits, dim=-1
        ).type_as(logits)
        labels = prep_outputs.labels.to(logits.device).long()
        labels[labels == -100] = 0
        tokens_logps = torch.gather(tokens_logps, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        tokens_logps = tokens_logps * prep_outputs.masks.to(logits.device)  # [b, s]

        return LogitsGeneratorOutputs(logits, tokens_logps)
