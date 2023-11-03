import collections
from typing import List, Union

import torch

from src.modeling.llama_abstract import AbstractLoraLlamaVerifier
from src.modeling.modeling import ModelForCausalLM
from src.tokenizer import LlamaTokenizer, Tokenizer
from src.utils import sample_top_p, masked_mean


class GeneratorForCausalLM:
    def __init__(self,  model: ModelForCausalLM,  tokenizer: Tokenizer, max_seq_len: int):
        self.model = model
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def forward(self, instructions: List[str], t: float = 0.0, p: float = 1.0) -> List[dict]:
        bsz = len(instructions)
        prompt_tokens = []
        for x in instructions:
            x = self.tokenizer.encode(x, bos=True, eos=False)
            prompt_tokens.append(x[: self.max_seq_len])
        min_prompt_size = min([len(t) for t in prompt_tokens])
        tokens = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).cuda().long()
        for k, tks in enumerate(prompt_tokens):
            tokens[k, : len(tks)] = torch.tensor(tks).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        unfinished_sequences = torch.ones(size=[bsz], dtype=torch.long).cuda()
        for cur_pos in range(start_pos, self.max_seq_len):
            with torch.no_grad():
                outputs = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos, use_cache=True)
                logits = outputs.logits[:, -1, :]
            if t > 0:
                probs = torch.softmax(logits / t, dim=-1)
                next_token = sample_top_p(probs, p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            unfinished_sequences = unfinished_sequences * (
                    next_token != self.tokenizer.eos_id).cuda().long()
            if unfinished_sequences.max() == 0:
                break
        decoded = []
        for i, tks in enumerate(tokens.tolist()):
            prompt_length = len(prompt_tokens[i])
            # cut to max gen len
            tks = tks[: self.max_seq_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in tks[1:]:
                tks = tks[: tks.index(self.tokenizer.eos_id, 1)]
            decoded.append(dict(
                instruction=self.tokenizer.decode(tks[:prompt_length]),
                output=self.tokenizer.decode(tks[prompt_length:])
            ))
        self.model.flush()
        return decoded


class GeneratorForVerifier:
    def __init__(self, model: AbstractLoraLlamaVerifier, tokenizer: LlamaTokenizer, max_seq_len: int):
        self.model = model
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def _truncating_strategy(self, instruction_ids, output_ids):
        instruction_length = len(instruction_ids)
        output_length = len(output_ids)
        if instruction_length >= self.max_seq_len:
            print(f'WARNING: Length of instruction {instruction_length} '
                  f'exceeds the max input length {self.max_seq_len}')
            instruction_ids = instruction_ids[:self.max_seq_len]
            instruction_length = len(instruction_ids)
        sequence_length = instruction_length + output_length
        if sequence_length > self.max_seq_len:
            exceed_length = sequence_length - self.max_seq_len
            output_ids = output_ids[:-exceed_length]
        return instruction_ids, output_ids

    def _prepare_for_generation(
            self,
            instructions: Union[List[str], List[List[int]]],
            outputs: Union[List[str], List[List[int]]]
    ):
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
                assert type(output[0]) is int, type(output[0])
                output_ids = output
            else:
                raise TypeError(type(output))
            instruction_ids, output_ids = self._truncating_strategy(instruction_ids, output_ids)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            tokens[i, :instr_len + output_len] = torch.tensor(instruction_ids + output_ids).long()
            masks[i, instr_len: instr_len + output_len] = True
        Output = collections.namedtuple('Outputs', ['tokens', 'masks'])
        return Output(tokens=tokens, masks=masks)

    def forward(self, instructions: Union[List[str], List[List[int]]], outputs: Union[List[str], List[List[int]]]):
        examples = self._prepare_for_generation(instructions, outputs)
        with torch.no_grad():
            tokens_rewards = self.model.forward(examples.tokens).cpu()
        result_tokens_rewards = []
        for i, tr in enumerate(tokens_rewards):
            result_tokens_rewards.append(torch.masked_select(tr, examples.masks[i]).tolist())
        rewards = masked_mean(tokens_rewards, examples.masks).tolist()
        Output = collections.namedtuple('Output', ['rewards', 'tokens_rewards'])
        return Output(rewards=rewards, tokens_rewards=result_tokens_rewards)
