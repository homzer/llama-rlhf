import collections
import random
from typing import List, Union

import numpy as np
import torch

from src.generator import GeneratorForVerifier, GeneratorForCausalLM, sampling_strategy
from src.models.modeling import ModelForCausalLM, ParallelModelForCausalLM, ParallelVerifier, Verifier
from src.tokenizers import Tokenizer
from src.utils import truncate

ActionGeneratorOutputs = collections.namedtuple("ActionGeneratorOutputs", [
    'responses', 'obs', 'actions', 'action_logits', 'action_masks', 'action_logprobs'
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

    def model_forward(self, tokens, input_masks=None, start_pos=None):
        bsz = tokens.shape[0]
        prev_pos = 0
        tokens = tokens.clone()
        unfinished_sequences = torch.ones(size=[bsz], dtype=torch.long, device=self.model.device())
        tokens_logits = torch.zeros(tokens.shape)
        tokens_logprobs = torch.zeros(tokens.shape)
        for cur_pos in range(start_pos, self.max_seq_len):
            with torch.no_grad():
                outputs = self.model.forward(
                    tokens[:, prev_pos: cur_pos], prev_pos, use_cache=True
                )
            # next_tokens = sampling_strategy(outputs.logits, self.temperature, self.top_p)
            next_tokens = self.sampling(outputs.logits, tokens=tokens, cur_pos=cur_pos)
            tokens_logits = tokens_logits.to(outputs.logits)
            tokens_logprobs = tokens_logprobs.to(outputs.logits)
            tokens_logits[:, prev_pos: cur_pos] = torch.gather(
                outputs.logits, dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)
            tokens_logprobs[:, prev_pos: cur_pos] = torch.gather(
                torch.log_softmax(outputs.logits, dim=-1), dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)
            next_token = next_tokens[:, -1].reshape(-1)
            next_token = torch.where(
                input_masks[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            unfinished_sequences = unfinished_sequences * (
                    next_token != self.tokenizer.eos_id
            ).long()
            if unfinished_sequences.max() == 0:
                break

        self.model.flush()
        Outputs = collections.namedtuple("Outputs", ['tokens', 'tokens_logits', 'tokens_logprobs'])
        return Outputs(tokens=tokens, tokens_logits=tokens_logits, tokens_logprobs=tokens_logprobs)

    def forward(self, instructions: Union[List[str], List[List[int]]]) -> ActionGeneratorOutputs:
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
        return ActionGeneratorOutputs(
            responses=responses,
            obs=forward_outputs.tokens,
            actions=output_tokens,
            action_logits=forward_outputs.tokens_logits,
            action_masks=output_masks,
            action_logprobs=forward_outputs.tokens_logprobs
        )


class DiversityActorGeneratorForCausalLM(ActorGeneratorForCausalLM):
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int,
            temperature: float = 1.0,
            top_p: float = 1.0,
            num_samples_per_prompt: int = 1,
            diverse_prob: float = 0.1
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            temperature=temperature,
            top_p=top_p
        )
        self.num_samples_per_prompt = num_samples_per_prompt
        self.diverse_prob = diverse_prob
        self.recorded_tokens = None

    # TODO: Copied from scr.generator.DiversityGeneratorForCausalLM().sampling()
    def sampling(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        # check for recorded_tokens
        tokens, cur_pos = kwargs.get("tokens"), kwargs.get("cur_pos")
        logits = logits[:, -1, :]  # [b, v]
        logits_masks = torch.full_like(logits, fill_value=False, dtype=torch.bool)  # [b, v]
        for i in range(tokens.shape[0]):
            for recorded_token in self.recorded_tokens[i]:
                if random.random() < self.diverse_prob and (recorded_token[: cur_pos] == tokens[i][: cur_pos]).all():
                    logits_masks[i][recorded_token[cur_pos]] = True
        logits = logits - logits_masks * 10000.
        next_tokens = sampling_strategy(logits, self.temperature, self.top_p)
        return next_tokens

    @staticmethod
    def stack_and_flatten(x: Union[List[List[torch.Tensor]], List[List[str]]]):
        if isinstance(x[0][0], torch.Tensor):
            return torch.stack([torch.stack(a, dim=0) for a in x], dim=0)
        elif isinstance(x[0][0], str):
            return [item for sublist in x for item in sublist]
        else:
            raise TypeError(type(x[0][0]))

    def forward(self, instructions: Union[List[str], List[List[int]]]) -> ActionGeneratorOutputs:
        self.model.eval()
        prep_outputs = self.prepare_for_generation(instructions)
        responses = [[] for _ in range(len(instructions))]
        obs = [[] for _ in range(len(instructions))]
        actions = [[] for _ in range(len(instructions))]
        action_logits = [[] for _ in range(len(instructions))]
        action_masks = [[] for _ in range(len(instructions))]
        action_logprobs = [[] for _ in range(len(instructions))]

        self.recorded_tokens = [[] for _ in range(len(instructions))]
        for _ in range(self.num_samples_per_prompt):
            forward_outputs = self.model_forward(
                prep_outputs.tokens, prep_outputs.input_masks, prep_outputs.start_pos
            )
            for i, recorded_token in enumerate(self.recorded_tokens):
                recorded_token.append(forward_outputs.tokens[i])
            output_masks = self.get_output_masks(forward_outputs.tokens, prep_outputs.input_masks)
            output_tokens = torch.zeros_like(forward_outputs.tokens)
            output_tokens[:, :-1] = forward_outputs.tokens[:, 1:]
            for i in range(len(instructions)):
                obs[i].append(forward_outputs.tokens[i])
                actions[i].append(output_tokens[i])
                action_logits[i].append(forward_outputs.tokens_logits[i])
                action_masks[i].append(output_masks[i])
                action_logprobs[i].append(forward_outputs.tokens_logprobs[i])
            for i, response in enumerate(self.decode_response(forward_outputs.tokens, output_masks)):
                responses[i].append(response)
        self.recorded_tokens = None

        return ActionGeneratorOutputs(
            responses=self.stack_and_flatten(responses),
            obs=self.stack_and_flatten(obs),
            actions=self.stack_and_flatten(actions),
            action_logits=self.stack_and_flatten(action_logits),
            action_masks=self.stack_and_flatten(action_masks),
            action_logprobs=self.stack_and_flatten(action_logprobs)
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


class LogitsGeneratorForCausalLM:
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

    def prepare_for_generation(
            self,
            instructions: Union[List[str], List[List[int]]],
            responses: Union[List[str], List[List[int]]]
    ):
        """ TODO: duplicated code with `ParallelSolverTrainer().prepare_for_generation()` """
        bsz = len(instructions)
        tokens = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).long()
        labels = torch.full((bsz, self.max_seq_len), -100).long()
        for i, (instruction, response) in enumerate(zip(instructions, responses)):
            if isinstance(instruction, str):
                instruction_ids = self.tokenizer.encode(instruction, bos=True, eos=False)
            elif isinstance(instruction, list) and isinstance(instruction[0], int):
                instruction_ids = instruction
            else:
                raise TypeError(type(instruction))

            if isinstance(response, str):
                output_ids = self.tokenizer.encode(response, bos=False, eos=True)
            elif isinstance(response, list) and isinstance(response[0], int):
                output_ids = response
            else:
                raise TypeError(type(response))

            instruction_ids, output_ids = truncate(instruction_ids, output_ids, self.max_seq_len)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            tokens[i, :instr_len + output_len] = torch.tensor(instruction_ids + output_ids).long()
            labels[i, instr_len - 1: instr_len - 1 + output_len] = torch.tensor(output_ids).long()
        masks = (labels != -100)
        Output = collections.namedtuple('Outputs', ['tokens', 'labels', 'masks'])
        return Output(tokens=tokens, labels=labels, masks=masks)

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
        prep_outputs = self.prepare_for_generation(instructions, responses)
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
