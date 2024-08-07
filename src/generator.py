import collections
from typing import List, Union

import torch

from src.models.modeling import ModelForCausalLM, ParallelModelForCausalLM, ParallelVerifier, Verifier
from src.tokenizers.tokenizer import Tokenizer
from src.utils import sample_top_p, masked_mean, truncate


class GeneratorForCausalLM:
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int,
            temperature: float = 0.0,
            top_p: float = 0.95
    ):
        self.model = model
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p

    def forward(self, instructions: List[str]) -> List[str]:
        self.model.eval()
        bsz = len(instructions)
        prompt_tokens = []
        for x in instructions:
            x = self.tokenizer.encode(x, bos=True, eos=False)
            prompt_tokens.append(x[: self.max_seq_len])
        min_prompt_size = min([len(t) for t in prompt_tokens])
        tokens = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).long()
        input_masks = torch.full((bsz, self.max_seq_len), False)
        unfinished_sequences = torch.ones(size=[bsz], dtype=torch.long)
        if torch.cuda.is_available():
            tokens = tokens.cuda()
            input_masks = input_masks.cuda()
            unfinished_sequences = unfinished_sequences.cuda()
        for k, tks in enumerate(prompt_tokens):
            tokens[k, :len(tks)] = torch.tensor(tks).long()
            input_masks[k, :len(tks)] = True
        # input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, self.max_seq_len):
            with torch.no_grad():
                outputs = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos, use_cache=True)
                logits = outputs.logits[:, -1, :]
            if self.temperature > 0:
                probs = torch.softmax(logits / self.temperature, dim=-1)
                next_token = sample_top_p(probs, self.top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_masks[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            unfinished_sequences = unfinished_sequences * (
                    (next_token != self.tokenizer.eos_id) | input_masks[:, cur_pos]
            )
            if unfinished_sequences.max() == 0:
                break
        decoded = []
        for i, tks in enumerate(tokens.tolist()):
            # cut to max gen len
            tks = tks[len(prompt_tokens[i]):]
            # cut to eos tok if any
            if self.tokenizer.eos_id in tks:
                tks = tks[: tks.index(self.tokenizer.eos_id)]
            decoded.append(self.tokenizer.decode(tks))
        self.model.flush()
        return decoded


class GeneratorForVerifier:
    def __init__(
            self,
            model: Union[Verifier, ParallelVerifier],
            tokenizer: Tokenizer,
            max_seq_len: int,
            reduce: str = "mean"
    ):
        self.model = model
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.reduce = reduce
        assert self.reduce in ["mean", "last"]

    # TODO: Duplicated code with src.trainer.ParallelVerifierTrainer.prepare_for_training()
    def prepare_for_generation(
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
            instruction_ids, output_ids = truncate(instruction_ids, output_ids, self.max_seq_len)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            tokens[i, :instr_len + output_len] = torch.tensor(instruction_ids + output_ids).long()
            masks[i, instr_len: instr_len + output_len] = True
        Output = collections.namedtuple('Outputs', ['tokens', 'masks'])
        return Output(tokens=tokens, masks=masks)

    def forward(self, instructions: Union[List[str], List[List[int]]], outputs: Union[List[str], List[List[int]]]):
        self.model.eval()
        examples = self.prepare_for_generation(instructions, outputs)
        with torch.no_grad():
            tokens_scores = self.model.forward(examples.tokens).scores
        tokens_scores = tokens_scores.detach().cpu()
        result_tokens_scores = []
        for i, score in enumerate(tokens_scores):
            result_tokens_scores.append(torch.masked_select(score, examples.masks[i]).tolist())
        if self.reduce == "mean":
            scores = masked_mean(tokens_scores, examples.masks).tolist()
        else:  # "last"
            scores = []
            for i, score in enumerate(tokens_scores):
                ids = examples.masks[i].nonzero()
                scores.append(score[ids[-1].item() if len(ids) > 0 else -1].item())
        Output = collections.namedtuple('Output', ['scores', 'tokens_scores'])
        return Output(scores=scores, tokens_scores=result_tokens_scores)
