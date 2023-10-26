import collections
from typing import List

import torch

from src.modeling_abstract import AbstractLlama, AbstractLoraLlamaVerifier
from src.tokenizer import LlamaTokenizer
from src.utils import sample_top_p, masked_mean


class Generator:
    def __init__(self, model: AbstractLlama, tokenizer: LlamaTokenizer):
        self.model = model
        self.max_seq_len = model.params.max_seq_len
        self.tokenizer = tokenizer

    def forward(self, instructions: List[str], t: float = 0.0, p: float = 1.0) -> List[dict]:
        bsz = len(instructions)
        prompt_tokens = []
        for x in instructions:
            x = self.tokenizer.encode(x, bos=True, eos=False)
            prompt_tokens.append(x[: self.max_seq_len])
        min_prompt_size = min([len(t) for t in prompt_tokens])
        tokens = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).cuda().long()
        for k, token in enumerate(prompt_tokens):
            tokens[k, : len(token)] = torch.tensor(token).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        unfinished_sequences = torch.ones(size=[bsz], dtype=torch.long).cuda()
        for cur_pos in range(start_pos, self.max_seq_len):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos, use_cache=True)[:, -1, :]
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
        for i, token in enumerate(tokens.tolist()):
            prompt_length = len(prompt_tokens[i])
            # cut to max gen len
            token = token[: self.max_seq_len]
            # cut to eos tok if any
            try:
                token = token[: token.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(dict(
                instruction=self.tokenizer.decode(token[:prompt_length]),
                output=self.tokenizer.decode(token[prompt_length:])
            ))
        self.model.flush()
        return decoded


class VerifierGenerator:
    def __init__(self, model: AbstractLoraLlamaVerifier, tokenizer: LlamaTokenizer):
        self.model = model
        self.max_seq_len = model.params.max_seq_len
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

    def _prepare_for_generation(self, instructions: List[str], outputs: List[str]):
        bsz = len(instructions)
        tokens = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).long()
        masks = torch.full((bsz, self.max_seq_len), False)
        for i, (instruction, output) in enumerate(zip(instructions, outputs)):
            instruction_ids = self.tokenizer.encode(instruction, bos=True, eos=False)
            output_ids = self.tokenizer.encode(output, bos=False, eos=True)
            instruction_ids, output_ids = self._truncating_strategy(instruction_ids, output_ids)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            tokens[i, :instr_len + output_len] = torch.tensor(instruction_ids + output_ids).long()
            masks[i, instr_len: instr_len + output_len] = True
        Output = collections.namedtuple('Outputs', ['tokens', 'masks'])
        return Output(tokens=tokens, masks=masks)

    def forward(self, instructions: List[str], chosen: List[str], rejected: List[str]):
        self.model.eval()
        c_examples = self._prepare_for_generation(instructions, chosen)
        r_examples = self._prepare_for_generation(instructions, rejected)
        with torch.no_grad():
            c_tokens_rewards = self.model.forward(c_examples.tokens).cpu()
            r_tokens_rewards = self.model.forward(r_examples.tokens).cpu()
        chosen_tokens_rewards = []
        rejected_tokens_rewards = []
        for i, (ctr, rtr) in enumerate(zip(c_tokens_rewards, r_tokens_rewards)):
            chosen_tokens_rewards.append(torch.masked_select(ctr, c_examples.masks[i]).tolist())
            rejected_tokens_rewards.append(torch.masked_select(rtr, r_examples.masks[i]).tolist())
        c_rewards = masked_mean(c_tokens_rewards, c_examples.masks).tolist()
        r_rewards = masked_mean(r_tokens_rewards, r_examples.masks).tolist()
        Output = collections.namedtuple('Output', [
            'chosen_rewards', 'rejected_rewards', 'chosen_tokens_rewards', 'rejected_tokens_rewards'])
        return Output(
            chosen_rewards=c_rewards,
            rejected_rewards=r_rewards,
            chosen_tokens_rewards=chosen_tokens_rewards,
            rejected_tokens_rewards=rejected_tokens_rewards
        )
