import collections
from typing import List, Union

import numpy as np
import torch

from src.generator import GeneratorForVerifier
from src.modeling.llama_lora import LoraLlamaVerifier
from src.modeling.modeling import ModelForCausalLM, ParallelModelForCausalLM
from src.tokenizer import Tokenizer, LlamaTokenizer

ActionGeneratorOutputs = collections.namedtuple("ActionGeneratorOutputs", [
    'outputs', 'logits', 'hidden_states', 'obs', 'actions', 'action_logits', 'action_masks'
])


class ActionGeneratorForCausalLM:
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

    def _prepare_for_generation(self, prompts: List[str]):
        bsz = len(prompts)
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        tokens = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_masks = tokens != self.tokenizer.pad_id
        unfinished_sequences = torch.ones(size=[bsz], dtype=torch.long)
        Outputs = collections.namedtuple("Outputs", [
            'tokens', 'input_masks', 'min_prompt_size',
            'unfinished_sequences', 'start_pos'
        ])
        return Outputs(
            tokens=tokens.to(self.model.device()),
            input_masks=input_masks.to(self.model.device()),
            unfinished_sequences=unfinished_sequences.to(self.model.device()),
            min_prompt_size=min_prompt_size,
            start_pos=min_prompt_size,
        )

    def _model_forward(self, preparation):
        prev_pos = 0
        start_pos = preparation.start_pos
        tokens = preparation.tokens.clone()
        input_masks = preparation.input_masks
        unfinished_sequences = preparation.unfinished_sequences
        hidden_states = None
        logits = torch.zeros((*tokens.shape, self.vocab_size), dtype=torch.float32, device=self.model.device())
        tokens_logits = torch.zeros(tokens.shape, dtype=torch.float32, device=self.model.device())
        for cur_pos in range(start_pos, self.max_seq_len):
            with torch.no_grad():
                outputs = self.model.forward(
                    tokens[:, prev_pos: cur_pos], prev_pos, use_cache=True
                )
            if hidden_states is None:
                hidden_states = torch.zeros(
                    (*tokens.shape, outputs.hidden_states.shape[-1]),
                    dtype=torch.float32,
                    device=self.model.device()
                )
            hidden_states[:, prev_pos: cur_pos, :] = outputs.hidden_states
            logits[:, prev_pos: cur_pos, :] = outputs.logits
            next_tokens = torch.argmax(outputs.logits, dim=-1)
            tokens_logits[:, prev_pos: cur_pos] = torch.gather(
                outputs.logits, dim=-1, index=next_tokens.unsqueeze(-1)
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
        Outputs = collections.namedtuple("Outputs", [
            'tokens', 'hidden_states', 'logits', 'tokens_logits'])
        return Outputs(tokens=tokens, hidden_states=hidden_states, logits=logits, tokens_logits=tokens_logits)

    def _get_output_masks(self, tokens, prompt_lengths):
        output_masks = torch.full_like(tokens, fill_value=True)
        for i, t in enumerate(tokens.tolist()):
            output_masks[i][: prompt_lengths[i] - 1] = False
            if self.tokenizer.eos_id in t[1:]:
                # find index of eos
                end = t.index(self.tokenizer.eos_id, 1)
                output_masks[i][end:] = False
            else:
                output_masks[i][-1:] = False
        return output_masks.to(torch.bool)

    def _decode_response(self, tokens, output_masks):
        responses = []
        # shift right
        shifted_output_masks = torch.full_like(output_masks, fill_value=False)
        shifted_output_masks[:, 1:] = output_masks[:, :-1]
        for t, m in zip(tokens, shifted_output_masks):
            responses.append(self.tokenizer.decode(t[m].tolist()))
        return responses

    def forward(self, instructions: List[str]) -> ActionGeneratorOutputs:
        self.model.eval()
        prepare_outputs = self._prepare_for_generation(instructions)
        forward_outputs = self._model_forward(prepare_outputs)

        prompt_lengths = torch.sum(prepare_outputs.input_masks, dim=-1)
        output_masks = self._get_output_masks(forward_outputs.tokens, prompt_lengths)
        outputs = self._decode_response(forward_outputs.tokens, output_masks)
        # input tokens shift left to get output tokens
        output_tokens = torch.zeros_like(forward_outputs.tokens)
        output_tokens[:, :-1] = forward_outputs.tokens[:, 1:]
        return ActionGeneratorOutputs(
            outputs=outputs,
            logits=forward_outputs.logits,
            hidden_states=forward_outputs.hidden_states,
            obs=forward_outputs.tokens,
            actions=output_tokens,
            action_logits=forward_outputs.tokens_logits,
            action_masks=output_masks,
        )


class CriticismGeneratorForCausalLM:
    def __init__(self, verifier: LoraLlamaVerifier, tokenizer: LlamaTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.generator = GeneratorForVerifier(verifier, tokenizer, max_seq_len)

    def forward(self, obs: List[str], actions: np.ndarray, action_masks: np.ndarray) -> List[List[float]]:
        outputs = []
        for action, action_mask in zip(actions, action_masks):
            outputs.append(action[action_mask].tolist())
        return self.generator.forward(obs, outputs).tokens_rewards
