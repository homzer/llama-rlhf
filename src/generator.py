import collections
import random
from typing import List, Union

import torch

from src.models.modeling import ModelForCausalLM, ParallelModelForCausalLM, ParallelVerifier, Verifier
from src.parallel import set_model_parallel_barrier
from src.tokenizers.tokenizer import Tokenizer
from src.utils import sample_top_p, masked_mean, truncate


def sampling_strategy(logits: torch.Tensor, t: float, p: float):
    if len(logits.shape) == 2:
        logits = logits[:, None, :]  # [b, 1, v]
    assert len(logits.shape) == 3
    seq_length = logits.shape[1]
    # only perform sampling on the last token
    last_logits = logits[:, -1, :]  # [b, v]
    if t > 0:  # Top-p Sampling
        next_tokens = sample_top_p(torch.softmax(last_logits / t, dim=-1), p)
    else:  # Greedy Sampling
        next_tokens = torch.argmax(last_logits, dim=-1, keepdim=True)
    if seq_length > 1:
        next_tokens = torch.cat([torch.argmax(logits[:, :-1, :], dim=-1), next_tokens], dim=-1)
    return next_tokens  # [b, s]


class GeneratorForCausalLM:
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
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

    def prepare_for_generation(self, prompts: Union[List[str], List[List[int]]], eos: bool = False):
        bsz = len(prompts)
        if isinstance(prompts[0], str):
            prompt_tokens = []
            for x in prompts:
                x = self.tokenizer.encode(x, bos=True, eos=eos)
                prompt_tokens.append(x[: self.max_seq_len])
        elif isinstance(prompts[0], list) and isinstance(prompts[0][0], int):
            prompt_tokens = prompts
        else:
            raise TypeError(type(prompts))
        min_prompt_size = min([len(t) for t in prompt_tokens])
        tokens = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_masks = tokens != self.tokenizer.pad_id
        Outputs = collections.namedtuple("Outputs", [
            'tokens', 'input_masks', 'start_pos'
        ])
        return Outputs(
            tokens=tokens.to(self.model.device()),
            input_masks=input_masks.to(self.model.device()),
            start_pos=min_prompt_size,
        )

    def sampling(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        return sampling_strategy(logits, self.temperature, self.top_p)

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
                    torch.any(torch.stack([next_token != self.tokenizer.eos_id, input_masks[:, cur_pos]]), dim=0)
            ).long()
            if unfinished_sequences.max() == 0:
                break

        self.model.flush()
        Outputs = collections.namedtuple("Outputs", ['tokens', 'tokens_logits', 'tokens_logprobs'])
        return Outputs(tokens=tokens, tokens_logits=tokens_logits, tokens_logprobs=tokens_logprobs)

    def get_output_masks(self, tokens, input_masks):
        prompt_lengths = torch.sum(input_masks, dim=-1)
        output_masks = torch.full_like(tokens, fill_value=True)
        for i, t in enumerate(tokens.tolist()):
            output_masks[i][: prompt_lengths[i] - 1] = False
            if self.tokenizer.eos_id in t[prompt_lengths[i]:]:
                # find index of eos
                end = t.index(self.tokenizer.eos_id, prompt_lengths[i])
                output_masks[i][end:] = False
            else:
                output_masks[i][-1:] = False
        return output_masks.to(torch.bool)

    def decode_response(self, tokens, output_masks):
        responses = []
        # shift right
        shifted_output_masks = torch.full_like(output_masks, fill_value=False)
        shifted_output_masks[:, 1:] = output_masks[:, :-1]
        for t, m in zip(tokens, shifted_output_masks):
            responses.append(self.tokenizer.decode(t[m].tolist()))
        return responses

    def forward(self, instructions: Union[List[str], List[List[int]]]) -> List[str]:
        self.model.eval()
        prep_outputs = self.prepare_for_generation(instructions)
        forward_outputs = self.model_forward(
            prep_outputs.tokens, prep_outputs.input_masks, prep_outputs.start_pos
        )
        output_masks = self.get_output_masks(forward_outputs.tokens, prep_outputs.input_masks)
        responses = self.decode_response(forward_outputs.tokens, output_masks)
        return responses


class GroupGeneratorForCausalLM(GeneratorForCausalLM):
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int,
            num_samples_per_prompt: int = 1,
            temperature: float = 0.0,
            top_p: float = 1.0,
            diverse_prob: float = None
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

    def sampling(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        # check for recorded_tokens
        tokens, cur_pos = kwargs.get("tokens"), kwargs.get("cur_pos")
        logits = logits[:, -1, :]  # [b, v]
        if self.diverse_prob is not None:
            logits_masks = torch.full_like(logits, fill_value=False, dtype=torch.bool)  # [b, v]
            for i in range(tokens.shape[0]):
                for recorded_token in self.recorded_tokens[i]:
                    if (
                            random.random() < self.diverse_prob and
                            (recorded_token[: cur_pos] == tokens[i][: cur_pos]).all()
                    ):
                        logits_masks[i][recorded_token[cur_pos]] = True
            logits = logits - logits_masks * 10000.
        next_tokens = sampling_strategy(logits, self.temperature, self.top_p)
        return next_tokens

    def forward(self, instructions: Union[List[str], List[List[int]]]) -> List[List[str]]:
        self.model.eval()
        prep_outputs = self.prepare_for_generation(instructions)
        responses = [[] for _ in range(len(instructions))]
        self.recorded_tokens = [[] for _ in range(len(instructions))]
        for _ in range(self.num_samples_per_prompt):
            forward_outputs = self.model_forward(
                prep_outputs.tokens, prep_outputs.input_masks, prep_outputs.start_pos
            )
            for i, recorded_token in enumerate(self.recorded_tokens):
                recorded_token.append(forward_outputs.tokens[i])
            output_masks = self.get_output_masks(forward_outputs.tokens, prep_outputs.input_masks)
            for i, response in enumerate(self.decode_response(forward_outputs.tokens, output_masks)):
                responses[i].append(response)
            print(instructions[-1] + "\n" + responses[-1][-1])
        self.recorded_tokens = None
        return responses


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
            scores = masked_mean(tokens_scores, examples.masks, dim=-1).tolist()
        else:  # "last"
            scores = []
            for i, score in enumerate(tokens_scores):
                ids = examples.masks[i].nonzero()
                scores.append(score[ids[-1].item() if len(ids) > 0 else -1].item())
        Output = collections.namedtuple('Output', ['scores', 'tokens_scores'])
        return Output(scores=scores, tokens_scores=result_tokens_scores)


class ValueAugmentedSamplingGeneratorForCausalLM(GeneratorForCausalLM):
    def __init__(
            self,
            policy: Union[ModelForCausalLM, ParallelModelForCausalLM],
            critic: Union[Verifier, ParallelVerifier],
            tokenizer: Tokenizer,  # tokenizers of policy and verifier must be the same
            max_seq_len: int,
            beam_size: int = 1,
            span_size: int = 1,
            tree_size: int = 1,
            temperature: float = 1.0,
            top_p: float = 1.0,
            force_diversity: bool = False
    ):
        super().__init__(
            model=policy,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            temperature=temperature,
            top_p=top_p
        )
        self.policy = policy
        self.policy_device = policy.device()
        self.critic = critic
        self.critic_device = critic.device()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.beam_size = beam_size
        self.span_size = span_size
        self.tree_size = tree_size
        self.force_diversity = force_diversity

    def move_policy_to_cpu(self):
        self.policy.cpu()
        torch.cuda.empty_cache()

    def move_policy_to_gpu(self):
        self.policy.cuda(self.policy_device)

    def force_diversity_sampling(self, probs: torch.Tensor) -> torch.Tensor:
        # [batch_size * beam_size * tree_size, tree_size]
        sampled_tokens = sample_top_p(probs, p=self.top_p, num_samples=self.tree_size)

        # shuffle the tokens in the last dimension of each sample
        shuffled_indices = [torch.randperm(sampled_tokens.shape[1]) for _ in range(sampled_tokens.shape[0])]
        candidate_tokens = torch.stack(
            [sampled_tokens[i][shuffled_indices[i]] for i in range(sampled_tokens.shape[0])], dim=0
        ).to(sampled_tokens)
        candidate_tokens = torch.reshape(candidate_tokens[:, 0], shape=[-1, self.beam_size * self.tree_size])

        # check for duplication of the top-1 tokens
        next_tokens = []
        for i, top_tokens in enumerate(sampled_tokens[:, 0].reshape(-1, self.beam_size * self.tree_size)):
            unique_elements, counts = torch.unique(top_tokens, return_counts=True)
            duplicates = unique_elements[counts > 1]
            duplicate_masks = (top_tokens.unsqueeze(-1) == duplicates.unsqueeze(0).expand(
                top_tokens.shape[0], duplicates.shape[0]
            )).any(dim=-1)
            next_tokens.append(torch.where(duplicate_masks, candidate_tokens[i], top_tokens))
        return torch.stack(next_tokens, dim=0).to(sampled_tokens).reshape(-1)

    def simulate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        :param logits: [batch_size * beam_size * tree_size, seq_len, vocab_size]
        :return:
        """
        logits = logits[:, -1, :]
        probs = torch.softmax(logits / self.temperature, dim=-1)
        # [batch_size * beam_size * tree_size, 1]
        if not self.force_diversity:
            next_tokens = sample_top_p(probs, p=self.top_p)
            return torch.reshape(next_tokens, shape=(-1,))
        else:
            return self.force_diversity_sampling(probs)

    def expand(self, logits: torch.Tensor, tree_size: int = None) -> torch.Tensor:
        tree_size = tree_size or self.tree_size
        assert (self.beam_size * self.tree_size) % tree_size == 0
        beam_size = (self.beam_size * self.tree_size) // tree_size
        logits = logits[:, -1, :]
        logits = torch.reshape(logits, shape=[-1, beam_size, tree_size, logits.shape[-1]])
        # select master tree branch
        logits = logits[:, :, 0, :]  # [batch_size, beam_size, vocab_size]
        probs = torch.softmax(logits / self.temperature, dim=-1)
        next_tokens = sample_top_p(probs, num_samples=tree_size)  # [batch_size, beam_size, tree_size]
        return next_tokens.reshape(-1)

    def verify(self, tokens: torch.Tensor, unfinished_sequences: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        tokens = tokens.clone()
        unfinished_sequences = unfinished_sequences.bool()
        self.move_policy_to_cpu()
        set_model_parallel_barrier()
        with torch.no_grad():
            scores = self.critic.forward(tokens).scores  # [batch_size * beam_size * tree_size, seq_len]
        self.move_policy_to_gpu()
        set_model_parallel_barrier()
        select_scores = scores
        scores = scores[:, -self.span_size:]
        scores = torch.mean(scores, dim=-1)  # [b]
        scores = torch.where(unfinished_sequences, scores, float("-inf"))  # ignore finished sequences
        scores_values, scores_indices = torch.topk(
            scores.reshape(-1, self.beam_size * self.tree_size), k=self.beam_size
        )  # [batch_size, beam_size]
        scores_indices += (torch.arange(0, len(scores_indices)) * self.beam_size * self.tree_size).unsqueeze(-1).to(scores_indices)
        scores_indices = scores_indices[:, :, None].expand(
            scores_indices.shape[0], self.beam_size, self.tree_size
        ).reshape(-1)
        scores_indices = torch.where(  # skip finished sequences
            unfinished_sequences, scores_indices, torch.arange(0, scores_indices.shape[0]).to(scores_indices)
        )
        # pruning
        self.policy.rearrange_kv_cache(scores_indices)
        select_tokens = tokens[scores_indices]  # [batch_size * beam_size * tree_size, seq_len]
        select_scores = select_scores[scores_indices]
        select_scores[:, :-1] = select_scores[:, 1:]  # shift left
        return select_tokens, select_scores

    def expand_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """ expand the dimension of `batch_size` into `batch_size * beam_size * tree_size` """
        x = x.clone()
        b, seq_len = x.shape
        x = x[:, None, :].expand(b, self.beam_size * self.tree_size, seq_len).reshape(-1, seq_len)
        return x

    def model_forward(self, tokens: torch.Tensor, input_masks: torch.Tensor = None, start_pos: int = None):
        self.policy.eval()
        self.critic.eval()
        unfinished_sequences = torch.ones(
            size=[tokens.shape[0]], dtype=torch.long, device=self.model.device()
        )
        scores = None
        prev_pos = 0
        span_step = 0
        for cur_pos in range(start_pos, self.max_seq_len):
            with torch.no_grad():
                outputs = self.policy.forward(
                    tokens[:, prev_pos: cur_pos], prev_pos, use_cache=True
                )

            if span_step == 0:  # expansion
                next_tokens = self.expand(outputs.logits)
            else:  # simulation
                next_tokens = self.simulate(outputs.logits)

            #############################################################

            next_tokens = torch.where(input_masks[:, cur_pos], tokens[:, cur_pos], next_tokens)
            tokens[:, cur_pos] = next_tokens
            unfinished_sequences = unfinished_sequences * (next_tokens != self.tokenizer.eos_id).long()
            if unfinished_sequences.max() == 0:
                break

            if span_step == self.span_size - 1:  # verification
                tokens, scores = self.verify(tokens, unfinished_sequences)

            span_step = (span_step + 1) % self.span_size
            prev_pos = cur_pos

        self.policy.flush()
        Outputs = collections.namedtuple("Outputs", ['tokens', 'scores'])
        return Outputs(tokens=tokens, scores=scores)

    def select_best_tokens(self, tokens: torch.Tensor, scores: torch.Tensor, output_masks: torch.Tensor):
        scores = masked_mean(scores, output_masks, dim=-1)  # [batch_size * beam_size * tree_size]
        scores = torch.reshape(scores, [-1, self.beam_size * self.tree_size])
        tokens = torch.reshape(tokens, [-1, self.beam_size * self.tree_size, self.max_seq_len])
        output_masks = torch.reshape(output_masks, [-1, self.beam_size * self.tree_size, self.max_seq_len])
        scores_values, scores_indices = torch.topk(scores, k=1)
        select_tokens = []
        select_output_masks = []
        for i, t, m in zip(scores_indices, tokens, output_masks):
            select_tokens.append(t[i.item()])
            select_output_masks.append(m[i.item()])
        Outputs = collections.namedtuple("Output", ["tokens", "output_masks"])
        select_tokens = torch.stack(select_tokens).to(tokens)
        select_output_masks = torch.stack(select_output_masks).to(output_masks)
        return Outputs(tokens=select_tokens, output_masks=select_output_masks)

    def forward(self, instructions: Union[List[str], List[List[int]]]) -> (List[List[str]], List[List[float]]):
        prepare_outputs = self.prepare_for_generation(instructions)
        tokens = self.expand_tensor(prepare_outputs.tokens)
        input_masks = self.expand_tensor(prepare_outputs.input_masks)
        forward_outputs = self.model_forward(tokens, input_masks, prepare_outputs.start_pos)
        output_masks = self.get_output_masks(forward_outputs.tokens, input_masks)
        # select_outputs = self.select_best_tokens(forward_outputs.tokens, forward_outputs.scores, output_masks)
        # responses = self.decode_response(select_outputs.tokens, select_outputs.output_masks)
        responses = self.decode_response(forward_outputs.tokens, output_masks)
        responses = [responses[i: i + self.beam_size * self.tree_size]
                     for i in range(0, len(responses), self.beam_size * self.tree_size)]
        scores = masked_mean(forward_outputs.scores, output_masks, dim=-1)
        scores = torch.reshape(scores, shape=[-1, self.beam_size * self.tree_size]).tolist()
        return responses, scores
