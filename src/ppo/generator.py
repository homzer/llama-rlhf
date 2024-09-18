import collections
from typing import List, Union

import numpy as np
import torch

from src.generator import GeneratorForVerifier, GeneratorForCausalLM, sampling_strategy
from src.models.modeling import ModelForCausalLM, ParallelModelForCausalLM, ParallelVerifier, Verifier
from src.tokenizers import Tokenizer
from src.utils import truncate

ActionGeneratorOutputs = collections.namedtuple("ActionGeneratorOutputs", [
    'outputs', 'obs', 'actions', 'action_logits', 'action_masks', 'action_logprobs'
])

SolverGeneratorOutputs = collections.namedtuple("SolverGeneratorOutputs", [
    'outputs', 'actions', 'action_masks'
])

LogitsGeneratorOutputs = collections.namedtuple("LogitsGeneratorOutputs", [
    'logits',
    'tokens_logps'
])


class SolverGeneratorForCausalLM(GeneratorForCausalLM):
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

    def forward(self, instructions: Union[List[str], List[List[int]]]) -> SolverGeneratorOutputs:
        self.model.eval()
        prep_outputs = self.prepare_for_generation(instructions)
        forward_outputs = self.model_forward(
            prep_outputs.tokens, prep_outputs.input_masks, prep_outputs.start_pos
        )
        prompt_lengths = torch.sum(prep_outputs.input_masks, dim=-1)
        output_masks = self.get_output_masks(forward_outputs.tokens, prompt_lengths)
        outputs = self.decode_response(forward_outputs.tokens, output_masks)
        # input tokens shift left to get output tokens
        output_tokens = torch.zeros_like(forward_outputs.tokens)
        output_tokens[:, :-1] = forward_outputs.tokens[:, 1:]
        return SolverGeneratorOutputs(
            outputs=outputs,
            actions=output_tokens,
            action_masks=output_masks,
        )


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


class ActorGeneratorForCausalLM(SolverGeneratorForCausalLM):
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
            next_tokens = sampling_strategy(outputs.logits, self.temperature, self.top_p)
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

        prompt_lengths = torch.sum(prep_outputs.input_masks, dim=-1)
        output_masks = self.get_output_masks(forward_outputs.tokens, prompt_lengths)
        outputs = self.decode_response(forward_outputs.tokens, output_masks)
        # input tokens shift left to get output tokens
        output_tokens = torch.zeros_like(forward_outputs.tokens)
        output_tokens[:, :-1] = forward_outputs.tokens[:, 1:]
        return ActionGeneratorOutputs(
            outputs=outputs,
            obs=forward_outputs.tokens,
            actions=output_tokens,
            action_logits=forward_outputs.tokens_logits,
            action_masks=output_masks,
            action_logprobs=forward_outputs.tokens_logprobs
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
    ) -> List[List[float]]:
        outputs = []
        for action, action_mask in zip(actions, action_masks):
            outputs.append(action[action_mask].tolist())
        return self.generator.forward(obs, outputs).tokens_scores
