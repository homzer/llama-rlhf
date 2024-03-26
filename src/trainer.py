import collections
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from src.criterion import RewardLoss, KLDivLoss
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

    # def _truncating_strategy(self, instruction_ids, output_ids):
    #     instruction_length = len(instruction_ids)
    #     output_length = len(output_ids)
    #     if instruction_length >= self.max_seq_len:
    #         print(f'WARNING: Length of instruction {instruction_length} '
    #               f'exceeds the max input length {self.max_seq_len}')
    #         instruction_ids = instruction_ids[:self.max_seq_len]
    #         instruction_length = len(instruction_ids)
    #     sequence_length = instruction_length + output_length
    #     if sequence_length > self.max_seq_len:
    #         exceed_length = sequence_length - self.max_seq_len
    #         output_ids = output_ids[:-exceed_length]
    #     return instruction_ids, output_ids

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

    def predict(self, logits, instructions: List[str], outputs: List[str]) -> List[dict]:
        bzs = int(logits.shape[0])
        datalist = []
        for i in range(bzs):
            instruction_ids = self.tokenizer.encode(instructions[i], bos=True)
            output_ids = self.tokenizer.encode(outputs[i], eos=True)
            instruction_ids, output_ids = truncate(instruction_ids, output_ids, self.max_seq_len)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            predict_ids = torch.argmax(logits[i], dim=-1)[instr_len - 1: instr_len - 1 + output_len].tolist()
            datalist.append(dict(instruction=instructions[i], output=self.tokenizer.decode(predict_ids)))
        return datalist

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
            target_logits: torch.Tensor,
            alpha: float = 1.0,
            beta: float = 1.0,
            T: float = 1.0
    ):
        self.model.train()
        example = self.prepare_for_training(instructions=instructions, outputs=outputs)
        logits = self.model.forward(example.tokens).logits

        loss_ce = alpha * self.criterion_ce.forward(
            input=logits.view(-1, logits.size(-1)),
            target=example.labels.view(-1).to(logits.device)
        )
        loss_kl = beta * self.criterion_kl.forward(
            logits=logits,
            targets=target_logits,
            masks=example.masks,
            T=T
        )
        loss = loss_ce + loss_kl
        self._back_propagation(loss)
        Output = collections.namedtuple('Output', ['loss', 'logits', 'loss_kl', 'loss_ce'])
        return Output(logits=logits, loss=loss, loss_kl=loss_kl, loss_ce=loss_ce)

    def distill_(
            self,
            instructions: List[str],
            outputs: List[str],
            target_logits: torch.Tensor,
            target_logits_: torch.Tensor,
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
            targets=target_logits,
            masks=example.masks
        )
        loss_kl_ = self.criterion_kl.forward(
            logits=logits,
            targets=target_logits_,
            masks=example.masks
        )
        loss = loss_ce + alpha * loss_kl + beta * loss_kl_
        self._back_propagation(loss)
        Output = collections.namedtuple('Output', ['loss', 'logits', 'loss_kl', 'loss_ce', 'loss_kl_'])
        return Output(logits=logits, loss=loss, loss_kl=loss_kl, loss_ce=loss_ce, loss_kl_=loss_kl_)


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
        self.criterion = RewardLoss()
        self.accumulation_steps = accumulation_steps
        self.step = 0

    def _back_propagation(self, loss: torch.Tensor):
        self.step += 1
        loss = loss / self.accumulation_steps
        loss.backward()
        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    # def _truncating_strategy(self, instruction_ids, output_ids):
    #     instruction_length = len(instruction_ids)
    #     output_length = len(output_ids)
    #     if instruction_length >= self.max_seq_len:
    #         print(f'WARNING: Length of instruction {instruction_length} '
    #               f'exceeds the max input length {self.max_seq_len}')
    #         instruction_ids = instruction_ids[:self.max_seq_len]
    #         instruction_length = len(instruction_ids)
    #     sequence_length = instruction_length + output_length
    #     if sequence_length > self.max_seq_len:
    #         exceed_length = sequence_length - self.max_seq_len
    #         output_ids = output_ids[:-exceed_length]
    #     return instruction_ids, output_ids

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
