import collections
from typing import List

import numpy as np
import torch

from src.models.modeling import ModelForCausalLM, ParallelModelForCausalLM, ParallelVerifier, Verifier
from src.tokenizers.tokenizer import Tokenizer
from src.trainer import prepare_for_forward
from src.utils import sample_top_p, masked_mean, truncate


def sampling_strategy(logits: torch.Tensor, t: float, p: float):
    if len(logits.shape) == 2:
        logits = logits[:, None, :]  # [b, 1, v]
    assert len(logits.shape) == 3
    seq_length = logits.shape[1]
    # only perform sampling on the last token
    last_logits = logits[:, -1, :]  # [b, v]
    if t > 0:  # Top-p Sampling
        next_tokens = sample_top_p(torch.softmax(last_logits.float() / t, dim=-1).to(last_logits.dtype), p)
    else:  # Greedy Sampling
        next_tokens = torch.argmax(last_logits, dim=-1, keepdim=True)
    if seq_length > 1:
        next_tokens = torch.cat([torch.argmax(logits[:, :-1, :], dim=-1), next_tokens], dim=-1)
    return next_tokens  # [b, s]


def prepare_for_generation(
        instructions: List[str] | List[List[int]] | np.ndarray,
        tokenizer: Tokenizer,
        max_seq_len: int,
):
    bsz = len(instructions)
    if isinstance(instructions, np.ndarray):
        instructions = instructions.tolist()
    if isinstance(instructions[0], str):
        prompt_tokens = []
        for x in instructions:
            x = tokenizer.encode(x, bos=True, eos=False)
            prompt_tokens.append(x[: max_seq_len])
    elif isinstance(instructions[0], list) and isinstance(instructions[0][0], int):
        prompt_tokens = instructions
    else:
        raise TypeError(type(instructions))
    min_prompt_size = min([len(t) for t in prompt_tokens])
    tokens = torch.full((bsz, max_seq_len), tokenizer.pad_id).long()
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t).long()
    input_masks = tokens != tokenizer.pad_id
    Outputs = collections.namedtuple("Outputs", [
        'tokens', 'input_masks', 'start_pos'
    ])
    return Outputs(tokens=tokens, input_masks=input_masks, start_pos=min_prompt_size)


def prepare_for_generation_with_prefix(
        instructions: List[str] | List[List[int]] | np.ndarray,
        prefixes: List[str] | List[List[int]] | np.ndarray,
        tokenizer: Tokenizer,
        max_seq_len: int,
):
    assert len(instructions) == len(prefixes)
    bsz = len(instructions)
    if isinstance(instructions, np.ndarray):
        instructions = instructions.tolist()
    if isinstance(instructions[0], str):
        prompt_tokens = []
        for x in instructions:
            x = tokenizer.encode(x, bos=True, eos=False)
            prompt_tokens.append(x[: max_seq_len])
    elif isinstance(instructions[0], list) and isinstance(instructions[0][0], int):
        prompt_tokens = instructions
    else:
        raise TypeError(type(instructions))

    prefix_masks = torch.full((bsz, max_seq_len), False)
    if isinstance(prefixes, np.ndarray):
        prefixes = prefixes.tolist()
    if isinstance(prefixes[0], str):
        for i, x in enumerate(prefixes):
            x = tokenizer.encode(x, bos=False, eos=False)
            prefix_masks[i][len(prompt_tokens[i]): len(prompt_tokens[i]) + len(x)] = True
            prompt_tokens[i].extend(x)
            prompt_tokens[i] = prompt_tokens[i][:max_seq_len]
    elif isinstance(prefixes[0], list) and isinstance(prefixes[0][0], int):
        for i, x in enumerate(prefixes):
            prefix_masks[i][len(prompt_tokens[i]): len(prompt_tokens[i]) + len(x)] = True
            prompt_tokens[i].extend(x)
            prompt_tokens[i] = prompt_tokens[i][:max_seq_len]
    else:
        raise TypeError(type(prefixes))

    min_prompt_size = min([len(t) for t in prompt_tokens])
    tokens = torch.full((bsz, max_seq_len), tokenizer.pad_id).long()
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t).long()
    input_masks = tokens != tokenizer.pad_id
    Outputs = collections.namedtuple("Outputs", [
        'tokens', 'input_masks', 'prefix_masks', 'start_pos'
    ])
    return Outputs(tokens=tokens, input_masks=input_masks, prefix_masks=prefix_masks, start_pos=min_prompt_size)


def get_output_masks(tokens: torch.Tensor, input_masks: torch.Tensor, tokenizer: Tokenizer) -> torch.Tensor:
    prompt_lengths = torch.sum(input_masks, dim=-1)
    output_masks = torch.full_like(tokens, fill_value=True)
    for i, t in enumerate(tokens.tolist()):
        output_masks[i][: prompt_lengths[i] - 1] = False
        if tokenizer.eos_id in t[prompt_lengths[i]:]:
            # find index of eos
            end = t.index(tokenizer.eos_id, prompt_lengths[i])
            output_masks[i][end:] = False
        else:
            output_masks[i][-1:] = False
    return output_masks.to(torch.bool)


class GeneratorForCausalLM:
    def __init__(
            self,
            model: ModelForCausalLM | ParallelModelForCausalLM,
            tokenizer: Tokenizer,
            max_seq_len: int,
            temperature: float = 0.0,
            top_p: float = 1.0
    ):
        self.model = model
        self.vocab_size = tokenizer.vocab_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p

    def prepare_for_generation(self, instructions: List[str] | List[List[int]]):
        return prepare_for_generation(
            instructions=instructions,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len
        )

    def sampling(self, logits: torch.Tensor) -> torch.Tensor:
        return sampling_strategy(logits, self.temperature, self.top_p)

    def model_forward(self, tokens, input_masks, start_pos):
        bsz = tokens.shape[0]
        prev_pos = 0
        tokens = tokens.to(self.model.device()).clone()
        input_masks = input_masks.to(self.model.device())
        unfinished_sequences = torch.ones(size=[bsz], dtype=torch.long, device=self.model.device())
        tokens_logits = torch.zeros(tokens.shape)
        tokens_logprobs = torch.zeros(tokens.shape)
        for cur_pos in range(start_pos, self.max_seq_len):
            with torch.no_grad():
                outputs = self.model.forward(
                    tokens[:, prev_pos: cur_pos], prev_pos, use_cache=True
                )
                next_tokens = self.sampling(outputs.logits)
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
                        torch.any(torch.stack([next_token != self.tokenizer.eos_id, input_masks[:, cur_pos]]), dim=0)
                ).long()
                if unfinished_sequences.max() == 0:
                    break

        self.model.flush()
        Outputs = collections.namedtuple("Outputs", ['tokens', 'tokens_logits', 'tokens_logprobs'])
        return Outputs(tokens=tokens, tokens_logits=tokens_logits, tokens_logprobs=tokens_logprobs)

    def get_output_masks(self, tokens, input_masks):
        return get_output_masks(
            tokens=tokens,
            input_masks=input_masks,
            tokenizer=self.tokenizer
        )

    def decode_response(self, tokens, output_masks):
        responses = []
        # shift right
        shifted_output_masks = torch.full_like(output_masks, fill_value=False)
        shifted_output_masks[:, 1:] = output_masks[:, :-1]
        for t, m in zip(tokens, shifted_output_masks):
            responses.append(self.tokenizer.decode(t[m].tolist()))
        return responses

    def forward(self, instructions: List[str] | List[List[int]]) -> List[str]:
        self.model.eval()
        prep_outputs = self.prepare_for_generation(instructions)
        forward_outputs = self.model_forward(
            prep_outputs.tokens, prep_outputs.input_masks, prep_outputs.start_pos
        )
        output_masks = self.get_output_masks(forward_outputs.tokens, prep_outputs.input_masks)
        responses = self.decode_response(forward_outputs.tokens, output_masks)
        return responses


class ForwardGeneratorForCausalLM:
    def __init__(
            self,
            model: ModelForCausalLM | ParallelModelForCausalLM,
            tokenizer: Tokenizer,
            max_seq_len: int,
    ):
        self.model = model
        self.vocab_size = tokenizer.vocab_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def prepare_for_forward(
            self,
            instructions: List[str] | List[List[int]],
            responses: List[str] | List[List[int]],
    ):
        outputs = prepare_for_forward(
            instructions=instructions,
            responses=responses,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len
        )
        outputs.labels[outputs.labels == -100] = 0
        Output = collections.namedtuple('Outputs', ['tokens', 'labels', 'masks'])
        return Output(tokens=outputs.tokens, labels=outputs.labels, masks=outputs.masks)

    def model_forward(self, tokens, labels):
        tokens = tokens.to(self.model.device())
        labels = labels.to(self.model.device())

        with torch.no_grad():
            outputs = self.model.forward(tokens)
        output_token_logits = torch.gather(
            outputs.logits, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        output_token_logprobs = torch.gather(
            torch.log_softmax(outputs.logits, dim=-1), dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        output_tokens = torch.argmax(outputs.logits, dim=-1)
        Outputs = collections.namedtuple("Outputs", [
            'tokens', 'output_tokens', 'output_token_logits', 'output_token_logprobs'])
        return Outputs(
            tokens=tokens,
            output_tokens=output_tokens,
            output_token_logits=output_token_logits,
            output_token_logprobs=output_token_logprobs
        )

    def forward(self, instructions: List[str] | List[List[int]], responses: List[str] | List[List[int]]):
        self.model.eval()

        prep_outputs = self.prepare_for_forward(instructions, responses)
        forward_outputs = self.model_forward(prep_outputs.tokens, prep_outputs.labels)

        Output = collections.namedtuple('Output', [
            'output_token_logprobs', 'output_masks'])
        return Output(
            output_token_logprobs=forward_outputs.output_token_logprobs,
            output_masks=prep_outputs.masks
        )


class PrefixGeneratorForCausalLM(GeneratorForCausalLM):
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

    def prepare_for_generation(self, instructions: List[str] | List[List[int]], prefixes: List[str] | List[List[int]]):
        return prepare_for_generation_with_prefix(
            instructions=instructions,
            prefixes=prefixes,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len
        )

    def forward(self, instructions: List[str] | List[List[int]], prefixes: List[str] | List[List[int]]) -> List[str]:
        self.model.eval()
        prep_outputs = self.prepare_for_generation(instructions, prefixes)
        forward_outputs = self.model_forward(
            prep_outputs.tokens, prep_outputs.input_masks, prep_outputs.start_pos
        )
        output_masks = self.get_output_masks(forward_outputs.tokens, prep_outputs.input_masks)
        responses = self.decode_response(forward_outputs.tokens, output_masks)
        return responses


class GroupGeneratorForCausalLM(GeneratorForCausalLM):
    def __init__(
            self,
            model: ModelForCausalLM | ParallelModelForCausalLM,
            tokenizer: Tokenizer,
            max_seq_len: int,
            num_samples_per_prompt: int = 1,
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
        self.num_samples_per_prompt = num_samples_per_prompt
        self.recorded_tokens = None

    def forward(self, instructions: List[str] | List[List[int]]) -> List[List[str]]:
        self.model.eval()
        prep_outputs = self.prepare_for_generation(instructions)
        responses = [[] for _ in range(len(instructions))]
        for _ in range(self.num_samples_per_prompt):
            forward_outputs = self.model_forward(
                prep_outputs.tokens, prep_outputs.input_masks, prep_outputs.start_pos
            )
            output_masks = self.get_output_masks(forward_outputs.tokens, prep_outputs.input_masks)
            for i, response in enumerate(self.decode_response(forward_outputs.tokens, output_masks)):
                responses[i].append(response)
            print(instructions[-1] + "\n" + responses[-1][-1])
        return responses


class GeneratorForVerifier:
    def __init__(
            self,
            model: Verifier | ParallelVerifier,
            tokenizer: Tokenizer,
            max_seq_len: int,
            reduce: str = "mean"
    ):
        self.model = model
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.reduce = reduce
        assert self.reduce in ["mean", "last"]

    def prepare_for_generation(
            self,
            instructions: List[str] | List[List[int]] | np.ndarray,
            outputs: List[str] | List[List[int]] | np.ndarray
    ):
        if isinstance(instructions, np.ndarray):
            instructions = instructions.tolist()
        if isinstance(outputs, np.ndarray):
            outputs = outputs.tolist()
        bsz = len(instructions)
        tokens = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).long()
        masks = torch.full((bsz, self.max_seq_len), False)
        for i, (instruction, output) in enumerate(zip(instructions, outputs)):
            if type(instruction) is str:
                instruction_ids = self.tokenizer.encode(instruction, bos=True, eos=False)
            elif type(instruction) is list:
                assert type(instruction[0]) is int, type(instruction[0])
                instruction_ids = instruction
            else:
                raise TypeError(type(instruction))
            if type(output) is str:
                output_ids = self.tokenizer.encode(output, bos=False, eos=True)
            elif type(output) is list:
                assert len(output) == 0 or type(output[0]) is int, type(output[0])
                output_ids = output
            else:
                raise TypeError(type(output))
            instruction_ids, output_ids = truncate(instruction_ids, output_ids, self.max_seq_len)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            tokens[i, :instr_len + output_len] = torch.tensor(instruction_ids + output_ids).long()
            masks[i, instr_len: instr_len + output_len] = True
        Output = collections.namedtuple('Outputs', ['tokens', 'masks'])
        return Output(tokens=tokens, masks=masks)

    def forward(self, instructions: List[str] | List[List[int]], outputs: List[str] | List[List[int]]):
        self.model.eval()
        examples = self.prepare_for_generation(instructions, outputs)
        with torch.no_grad():
            tokens_scores = self.model.forward(examples.tokens).scores
        tokens_scores = tokens_scores.detach().cpu()
        result_tokens_scores = []
        for i, score in enumerate(tokens_scores):
            result_tokens_scores.append(torch.masked_select(score, examples.masks[i]).tolist())
        if self.reduce == "mean":
            scores = masked_mean(tokens_scores, examples.masks, dim=-1).tolist()
        else:  # "last"
            scores = []
            for i, score in enumerate(tokens_scores):
                ids = examples.masks[i].nonzero()
                scores.append(score[ids[-1].item() if len(ids) > 0 else -1].item())
        Output = collections.namedtuple('Output', ['scores', 'tokens_scores'])
        return Output(scores=scores, tokens_scores=result_tokens_scores)
