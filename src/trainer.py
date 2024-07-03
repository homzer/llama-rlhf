import collections
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from src.criterion import PairwiseScoreLoss, KLDivLoss, DpoLoss, ReverseKLDivLoss, JSDivLoss, LastTokenScoreLoss
from src.models.modeling import Module, ParallelModule, ParallelModelForCausalLM, ParallelVerifier
from src.tokenizers import Tokenizer
from src.utils import set_barrier, truncate


class Trainer:
    def __init__(self, model: Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        # To avoid overflowing, will cause performance degradation !!!
        # if "eps" in self.optimizer.defaults:
        #     # get trainable dtype
        #     dtype = optimizer.param_groups[0]['params'][0].dtype
        #     if dtype == torch.float16:
        #         self.optimizer.defaults["eps"] = torch.finfo(dtype).tiny
        #         for group in self.optimizer.param_groups:
        #             group["eps"] = torch.finfo(dtype).tiny

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def save_optimizer(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        print(f'Saving optimizer to {save_path} ......')
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, f'optimizer.bin'))
        print(f'Saving done !')

    def load_optimizer(self, save_path: str):
        print(f'Loading optimizer from {save_path} .....')
        state_dict = torch.load(save_path)
        self.optimizer.load_state_dict(state_dict)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        print(f'Loading done !')

    def save_model(self, save_path: str):
        self.model.save(save_path)

    def load_model(self, save_path: str):
        self.model.load(save_path)

    def load(self, save_path: str):
        if save_path is None or save_path.lower() == "none":
            print("WARNING: Not loading model because `save_path` is None")
            return
        self.load_optimizer(save_path)
        self.load_model(save_path)

    def save(self, save_path: str):
        if save_path is None or save_path.lower() == "none":
            print("WARNING: Not saving model because `save_path` is None")
            return
        self.save_optimizer(save_path)
        self.save_model(save_path)


class ParallelTrainer(Trainer):
    def __init__(
            self,
            model: ParallelModule,
            optimizer: torch.optim.Optimizer
    ):
        super().__init__(model, optimizer)
        self.local_rank = model.local_rank
        self.world_size = model.world_size

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def save_optimizer(self, save_path: str):
        if self.local_rank == 0:
            os.makedirs(save_path, exist_ok=True)
        print(f'Saving optimizer to {save_path} ......')
        set_barrier()
        torch.save(self.optimizer.state_dict(), os.path.join(
            save_path, f'optimizer.0{self.local_rank}.bin'))
        set_barrier()
        print(f'Saving done !')

    def load_optimizer(self, save_path: str):
        checkpoints = sorted(Path(save_path).glob("optimizer.*.bin"))
        if len(checkpoints) == 0:
            return
        print(f'Loading optimizer from {save_path} .....')
        assert self.world_size == len(
            checkpoints
        ), f"Loading a optimizer for MP={len(checkpoints)} but world size is {self.world_size}"
        optim_file = checkpoints[self.local_rank]
        state_dict = torch.load(str(optim_file))
        self.optimizer.load_state_dict(state_dict)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        set_barrier()
        print(f'Loading done !')


class ParallelSolverTrainer(ParallelTrainer):
    def __init__(
            self,
            model: ParallelModelForCausalLM,
            tokenizer: Tokenizer,
            optimizer: torch.optim.Optimizer,
            max_seq_len: int,
            accumulation_steps: int = 1
    ):
        super().__init__(model, optimizer)
        self.model = model
        self.local_rank = model.local_rank
        self.world_size = model.world_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.accumulation_steps = accumulation_steps
        self.step = 0

    def _back_propagation(self, loss: torch.Tensor):
        self.step += 1
        loss = loss / self.accumulation_steps
        loss.backward()
        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def prepare_for_training(self, instructions, outputs):
        """ :return tokens, labels """
        bsz = len(instructions)
        tokens = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).long()
        labels = torch.full((bsz, self.max_seq_len), -100).long()
        for i, (instruction, output) in enumerate(zip(instructions, outputs)):
            instruction_ids = self.tokenizer.encode(instruction, bos=True, eos=False)
            output_ids = self.tokenizer.encode(output, bos=False, eos=True)
            # instruction_ids, output_ids = self._truncating_strategy(instruction_ids, output_ids)
            instruction_ids, output_ids = truncate(instruction_ids, output_ids, self.max_seq_len)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            tokens[i, :instr_len + output_len] = torch.tensor(instruction_ids + output_ids).long()
            labels[i, instr_len - 1: instr_len - 1 + output_len] = torch.tensor(output_ids).long()
        masks = (labels != -100)
        Output = collections.namedtuple('Outputs', ['tokens', 'labels', 'masks'])
        return Output(tokens=tokens, labels=labels, masks=masks)

    def predict(self, logits, instructions: List[str], outputs: List[str]):
        bzs = min(int(logits.shape[0]), 1)
        datalist = []
        for i in range(bzs):
            instruction_ids = self.tokenizer.encode(instructions[i], bos=True)
            output_ids = self.tokenizer.encode(outputs[i], eos=True)
            instruction_ids, output_ids = truncate(instruction_ids, output_ids, self.max_seq_len)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            predict_ids = torch.argmax(logits[i], dim=-1)[instr_len - 1: instr_len - 1 + output_len].tolist()
            datalist.append(dict(instruction=instructions[i], output=self.tokenizer.decode(predict_ids)))
        print(datalist[0]['instruction'] + datalist[0]['output'])

    def forward(self, instructions: List[str], outputs: List[str]):
        """ Instruction tuning """
        self.model.train()
        example = self.prepare_for_training(instructions=instructions, outputs=outputs)
        logits = self.model.forward(example.tokens).logits
        loss = self.criterion.forward(
            input=logits.view(-1, logits.size(-1)),
            target=example.labels.view(-1).to(logits.device)
        )
        self._back_propagation(loss)
        Output = collections.namedtuple('Output', ['loss', 'logits'])
        return Output(logits=logits, loss=loss)


class ParallelSolverDistillTrainer(ParallelSolverTrainer):
    def __init__(
            self,
            model: ParallelModelForCausalLM,
            tokenizer: Tokenizer,
            optimizer: torch.optim.Optimizer,
            max_seq_len: int,
            accumulation_steps: int = 1,
            use_reverse_kl: bool = False,
            use_js: bool = False
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            max_seq_len=max_seq_len,
            accumulation_steps=accumulation_steps
        )
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion_kl = KLDivLoss()
        if use_reverse_kl:
            self.criterion_kl = ReverseKLDivLoss()
        if use_js:
            self.criterion_kl = JSDivLoss()

    def distill(
            self,
            instructions: List[str],
            outputs: List[str],
            target_logits: torch.Tensor,
            alpha: float = 1.0,
            beta: float = 1.0,
            temperature: float = 1.0
    ):
        self.model.train()
        example = self.prepare_for_training(instructions=instructions, outputs=outputs)
        logits = self.model.forward(example.tokens).logits

        loss_ce = beta * self.criterion_ce.forward(
            input=logits.view(-1, logits.size(-1)),
            target=example.labels.view(-1).to(logits.device)
        )
        loss_kl = alpha * self.criterion_kl.forward(
            logits=logits,
            targets=target_logits,
            masks=example.masks,
            T=temperature
        )
        loss = loss_ce + loss_kl
        self._back_propagation(loss)
        Output = collections.namedtuple('Output', ['loss', 'logits', 'loss_kl', 'loss_ce'])
        return Output(logits=logits, loss=loss, loss_kl=loss_kl, loss_ce=loss_ce)


class ParallelSolverTripleDistillTrainer(ParallelSolverTrainer):
    def __init__(
            self,
            model: ParallelModelForCausalLM,
            tokenizer: Tokenizer,
            optimizer: torch.optim.Optimizer,
            max_seq_len: int,
            accumulation_steps: int = 1
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            max_seq_len=max_seq_len,
            accumulation_steps=accumulation_steps
        )
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion_kl = KLDivLoss()

    def distill(
            self,
            instructions: List[str],
            outputs: List[str],
            target_logits_a: torch.Tensor,
            target_logits_b: torch.Tensor,
            alpha: float = 1.0,
            beta: float = 1.0
    ):
        self.model.train()
        example = self.prepare_for_training(instructions=instructions, outputs=outputs)
        logits = self.model.forward(example.tokens).logits

        loss_ce = self.criterion_ce.forward(
            input=logits.view(-1, logits.size(-1)),
            target=example.labels.view(-1).to(logits.device)
        )
        loss_kl = self.criterion_kl.forward(
            logits=logits,
            targets=target_logits_a,
            masks=example.masks
        )
        loss_kl_ = self.criterion_kl.forward(
            logits=logits,
            targets=target_logits_b,
            masks=example.masks
        )
        loss = loss_ce + alpha * loss_kl + beta * loss_kl_
        self._back_propagation(loss)
        Output = collections.namedtuple('Output', ['loss', 'logits', 'loss_kl', 'loss_ce', 'loss_kl_'])
        return Output(logits=logits, loss=loss, loss_kl=loss_kl, loss_ce=loss_ce, loss_kl_=loss_kl_)


class ParallelSolverMccDistillTrainer(ParallelSolverTrainer):
    def __init__(
            self,
            model: ParallelModelForCausalLM,
            tokenizer: Tokenizer,
            optimizer: torch.optim.Optimizer,
            max_seq_len: int,
            accumulation_steps: int = 1
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            max_seq_len=max_seq_len,
            accumulation_steps=accumulation_steps
        )
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion_kl = KLDivLoss()

    def compute_kl_for_mcc(
            self,
            indices_a: List[List[torch.Tensor]],
            indices_b: List[List[torch.Tensor]],
            logits_a: torch.Tensor,
            logits_b: torch.Tensor,
            T: float
    ) -> torch.Tensor:
        """ It's gonna be a bit messy """
        indices_a, indices_b = indices_a[0], indices_b[0]
        indices_a = torch.cat([t.unsqueeze(0) for t in indices_a], dim=0).to(logits_a.device).T
        indices_b = torch.cat([t.unsqueeze(0) for t in indices_b], dim=0).to(logits_b.device).T
        bzs = indices_a.shape[0]
        vocab_size = logits_a.shape[-1]
        max_len1 = max(torch.sub(indices_a[:, 1], indices_a[:, 0]))
        max_len2 = max(torch.sub(indices_b[:, 1], indices_b[:, 0]))
        assert max_len1 == max_len2
        p = torch.full(size=(bzs, max_len1, vocab_size), fill_value=0.).float()
        q = torch.full(size=(bzs, max_len2, vocab_size), fill_value=0.).float()
        valid_batch_indices = []  # only count for those within `max_seq_len`
        for i in range(bzs):
            if indices_a[i, 1] >= self.max_seq_len or indices_b[i, 1] >= self.max_seq_len:
                print(f'WARNING: Escaping batch index because {max(indices_a[i, 1], indices_b[i, 1])} '
                      f'exceeding max length {self.max_seq_len}')
                continue
            p[i, : indices_a[i, 1] - indices_a[i, 0], :] = logits_a[i, indices_a[i, 0]: indices_a[i, 1], :]
            q[i, : indices_b[i, 1] - indices_b[i, 0], :] = logits_b[i, indices_b[i, 0]: indices_b[i, 1], :]
            valid_batch_indices.append(i)
        if len(valid_batch_indices) == 0:
            return torch.tensor(0.0)

        p = p[valid_batch_indices]
        q = q[valid_batch_indices]
        masks = (torch.sum(p, dim=-1) != 0)
        p_loss = self.criterion_kl.forward(p, q, masks=masks, T=T)
        q_loss = self.criterion_kl.forward(q, p, masks=masks, T=T)
        return (p_loss + q_loss) * 0.5

    def distill(
            self,
            instructions: List[str],
            outputs_a: List[str],
            outputs_b: List[str],
            indices_a: List[List[torch.Tensor]],
            indices_b: List[List[torch.Tensor]],
            alpha: float = 1.0,
            T: float = 1.0
    ):
        example_a = self.prepare_for_training(instructions, outputs_a)
        example_b = self.prepare_for_training(instructions, outputs_b)

        logits_a = self.model.forward(example_a.tokens).logits
        logits_b = self.model.forward(example_b.tokens).logits

        ce_loss_a = self.criterion_ce.forward(logits_a, example_a.tokens)
        ce_loss_b = self.criterion_ce.forward(logits_b, example_b.tokens)
        ce_loss = (ce_loss_a + ce_loss_b) * 0.5

        # Compute KL Div Loss
        kl_loss = alpha * self.compute_kl_for_mcc(indices_a, indices_b, logits_a, logits_b, T)

        loss = ce_loss + kl_loss
        self._back_propagation(loss)
        Output = collections.namedtuple('Output', ['logits_a', 'logits_b', 'loss', 'loss_kl', 'loss_ce'])
        return Output(logits_a=logits_a, logits_b=logits_b, loss=loss, loss_kl=kl_loss, loss_ce=ce_loss)


class ParallelSolverReferenceDistillTrainer(ParallelSolverTrainer):
    def __init__(
            self,
            model: ParallelModelForCausalLM,
            tokenizer: Tokenizer,
            optimizer: torch.optim.Optimizer,
            max_seq_len: int,
            accumulation_steps: int = 1
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            max_seq_len=max_seq_len,
            accumulation_steps=accumulation_steps
        )
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion_kl = KLDivLoss(return_scalar=False)
        self.eps = 1e-4

    def reference_kl_loss(self, logits, target_logits, target_logps, ref_logps, masks, temperature, scale):
        """ Compute KL loss with reference point weighting """
        loss = self.criterion_kl.forward(
            logits=logits,
            targets=target_logits,
            masks=masks,
            T=temperature
        )

        target_logps = target_logps.to(logits)
        masks = masks.to(logits.device)
        norm_factor = masks.sum(-1)  # [b]
        target_logps = ((target_logps * masks).sum(-1) / (norm_factor + self.eps)).type_as(target_logps)
        target_logps[norm_factor == 0] = -10000.0
        refs = torch.sigmoid((target_logps - ref_logps) * scale)
        loss = refs.unsqueeze(-1) * loss
        loss = torch.masked_select(loss.view(-1), masks.view(-1)).mean()
        Outputs = collections.namedtuple("Outputs", ['loss', 'refs'])
        return Outputs(loss=loss, refs=refs)

    def distill(
            self,
            instructions: List[str],
            outputs: List[str],
            target_logits: torch.Tensor,
            target_logps: torch.Tensor,
            ref_logps: float,
            ref_logps_scale: float,
            alpha: float = 1.0,
            beta: float = 1.0,
            temperature: float = 1.0
    ):
        """
        Knowledge distillation with weighted reference point
        :param ref_logps_scale: scaling factor
        :param instructions:
        :param outputs:
        :param target_logits: [b, s, v]
        :param target_logps: [b, s], log probs of label tokens
        :param ref_logps: average of target_logps over a mini-batch
        :param alpha: kl weight
        :param beta: ce weight
        :param temperature:
        :return:
        """
        self.model.train()
        example = self.prepare_for_training(instructions=instructions, outputs=outputs)
        logits = self.model.forward(example.tokens).logits

        loss_ce = beta * self.criterion_ce.forward(
            input=logits.view(-1, logits.size(-1)),
            target=example.labels.view(-1).to(logits.device)
        )
        kl_loss_outputs = self.reference_kl_loss(
            logits=logits,
            target_logits=target_logits,
            target_logps=target_logps,
            scale=ref_logps_scale,
            ref_logps=ref_logps,
            masks=example.masks,
            temperature=temperature
        )
        loss_kl = alpha * kl_loss_outputs.loss
        loss = loss_ce + loss_kl
        self._back_propagation(loss)
        Output = collections.namedtuple('Output', ['loss', 'logits', 'loss_kl', 'loss_ce', 'refs'])
        return Output(logits=logits, loss=loss, loss_kl=loss_kl, loss_ce=loss_ce, refs=kl_loss_outputs.refs)


class ParallelSolverDpoTrainer(ParallelSolverTrainer):
    def __init__(
            self,
            model: ParallelModelForCausalLM,
            tokenizer: Tokenizer,
            optimizer: torch.optim.Optimizer,
            max_seq_len: int,
            ce_coef: float = 1.0,
            accumulation_steps: int = 1
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            max_seq_len=max_seq_len,
            accumulation_steps=accumulation_steps
        )
        self.ce_coef = ce_coef
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion_dpo = DpoLoss()

    def dpo_forward(
            self,
            instructions: List[str],
            chosen: List[str],
            rejected: List[str],
            reference_chosen_log_probs: torch.Tensor = None,
            reference_rejected_log_probs: torch.Tensor = None,
            reference_chosen_logits: torch.Tensor = None,
            reference_rejected_logits: torch.Tensor = None
    ):
        chosen_examples = self.prepare_for_training(instructions, chosen)
        rejected_examples = self.prepare_for_training(instructions, rejected)

        chosen_logits = self.model.forward(chosen_examples.tokens).logits
        rejected_logits = self.model.forward(rejected_examples.tokens).logits

        dpo_loss = self.criterion_dpo.forward(
            chosen_logits=chosen_logits,
            rejected_logits=rejected_logits,
            chosen_labels=chosen_examples.labels,
            rejected_labels=rejected_examples.labels,
            chosen_masks=chosen_examples.masks,
            rejected_masks=rejected_examples.masks,
            ref_chosen_log_probs=reference_chosen_log_probs,
            ref_rejected_log_probs=reference_rejected_log_probs,
            ref_chosen_logits=reference_chosen_logits,
            ref_rejected_logits=reference_rejected_logits
        )

        ce_loss = self.ce_coef * self.criterion_ce.forward(
            input=chosen_logits.view(-1, chosen_logits.size(-1)),
            target=chosen_examples.labels.view(-1).to(chosen_logits.device)
        )

        loss = dpo_loss + ce_loss
        self._back_propagation(loss)

        Output = collections.namedtuple('Output', ['logits', 'loss', 'loss_dpo', 'loss_ce'])
        return Output(logits=chosen_logits, loss=loss, loss_dpo=dpo_loss, loss_ce=ce_loss)


class ParallelVerifierTrainer(ParallelTrainer):
    def __init__(
            self,
            model: ParallelVerifier,
            tokenizer: Tokenizer,
            optimizer: torch.optim.Optimizer,
            accumulation_steps: int = 1
    ):
        super().__init__(model, optimizer)
        self.model = model
        self.local_rank = model.local_rank
        self.world_size = model.world_size
        self.max_seq_len = self.model.args.max_seq_len
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step = 0

    def _back_propagation(self, loss: torch.Tensor):
        self.step += 1
        loss = loss / self.accumulation_steps
        loss.backward()
        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def prepare_for_training(self, instructions: List[str], outputs: List[str]):
        bsz = len(instructions)
        tokens = torch.full((bsz, self.max_seq_len), self.tokenizer.pad_id).long()
        masks = torch.full((bsz, self.max_seq_len), False)
        for i, (instruction, output) in enumerate(zip(instructions, outputs)):
            instruction_ids = self.tokenizer.encode(instruction, bos=True, eos=False)
            output_ids = self.tokenizer.encode(output, bos=False, eos=True)
            instruction_ids, output_ids = truncate(instruction_ids, output_ids, self.max_seq_len)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            tokens[i, :instr_len + output_len] = torch.tensor(instruction_ids + output_ids).long()
            masks[i, instr_len: instr_len + output_len] = True
        Output = collections.namedtuple('Outputs', ['tokens', 'masks'])
        return Output(tokens=tokens, masks=masks)


class ParallelVerifierPairwiseTrainer(ParallelVerifierTrainer):
    def __init__(
            self,
            model: ParallelVerifier,
            tokenizer: Tokenizer,
            optimizer: torch.optim.Optimizer,
            accumulation_steps: int = 1
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            accumulation_steps=accumulation_steps
        )
        self.criterion = PairwiseScoreLoss()

    def forward(self, instructions: List[str], chosen: List[str], rejected: List[str]):
        self.model.train()
        c_examples = self.prepare_for_training(instructions, chosen)
        r_examples = self.prepare_for_training(instructions, rejected)
        c_rewards = self.model.forward(c_examples.tokens)
        r_rewards = self.model.forward(r_examples.tokens)

        loss = self.criterion.forward(
            chosen_rewards=c_rewards.scores,
            rejected_rewards=r_rewards.scores,
            chosen_masks=c_examples.masks.to(c_rewards.scores.device),
            rejected_masks=r_examples.masks.to(r_rewards.scores.device)
        )
        self._back_propagation(loss)

        Output = collections.namedtuple('Output', ['loss'])
        return Output(loss=loss)


class ParallelVerifierPointwiseTrainer(ParallelVerifierTrainer):
    def __init__(
            self,
            model: ParallelVerifier,
            tokenizer: Tokenizer,
            optimizer: torch.optim.Optimizer,
            accumulation_steps: int = 1
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            accumulation_steps=accumulation_steps
        )
        self.criterion = LastTokenScoreLoss()

    def forward(self, instructions: List[str], outputs: List[str], labels: List[int]):
        self.model.train()
        examples = self.prepare_for_training(instructions, outputs)
        scores = self.model.forward(examples.tokens).scores
        loss = self.criterion.forward(
            scores=scores,
            masks=examples.masks.to(scores.device),
            labels=labels
        )
        self._back_propagation(loss)

        Output = collections.namedtuple('Output', ['loss'])
        return Output(loss=loss)

