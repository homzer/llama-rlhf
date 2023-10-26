import collections
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from src.criterion import KLDivLoss, RewardLoss
from src.modeling_abstract import DistributedModule, AbstractLlama
from src.modeling_lora import LoraLlamaVerifier
from src.tokenizer import LlamaTokenizer
from src.utils import barrier, reconstruct_logits_from_dicts


class DistributedTrainer:
    def __init__(
            self,
            model: DistributedModule,
            optimizer: torch.optim.Optimizer
    ):
        self.local_rank = model.local_rank
        self.world_size = model.world_size
        self.model = model
        self.optimizer = optimizer

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def save_distributed_optimizer(self, save_path: str):
        if self.local_rank == 0:
            os.makedirs(save_path, exist_ok=True)
        print(f'Saving optimizer to {save_path} ......')
        barrier()
        torch.save(self.optimizer.state_dict(), os.path.join(
            save_path, f'optimizer.0{self.local_rank}.bin'))
        barrier()
        print(f'Saving done !')

    def load_distributed_optimizer(self, save_path: str):
        checkpoints = sorted(Path(save_path).glob("optimizer.*.bin"))
        if len(checkpoints) == 0:
            return
        print(f'Loading optimizer from {save_path} .....')
        assert self.world_size == len(
            checkpoints
        ), f"Loading a optimizer for MP={len(checkpoints)} but world size is {self.world_size}"
        optim_file = checkpoints[self.local_rank]
        state_dict = torch.load(optim_file)
        self.optimizer.load_state_dict(state_dict)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        print(f'Loading done !')

    def save_distributed_model(self, save_path: str):
        self.model.save(save_path)

    def load_distributed_model(self, save_path: str):
        self.model.load(save_path)

    def load(self, save_path: str):
        if save_path is None or save_path.lower() == "none":
            print("WARNING: Not loading model because `save_path` is None")
            return
        self.load_distributed_optimizer(save_path)
        self.load_distributed_model(save_path)

    def save(self, save_path: str):
        if save_path is None or save_path.lower() == "none":
            print("WARNING: Not saving model because `save_path` is None")
            return
        self.save_distributed_optimizer(save_path)
        self.save_distributed_model(save_path)


class DistributedSolverTrainer(DistributedTrainer):
    def __init__(
            self,
            model: AbstractLlama,
            tokenizer: LlamaTokenizer,
            optimizer: torch.optim.Optimizer,
            accumulation_steps: int = 1
    ):
        super().__init__(model, optimizer)
        self.model = model
        self.local_rank = model.local_rank
        self.world_size = model.world_size
        self.max_seq_len = self.model.params.max_seq_len
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.accumulation_steps = accumulation_steps
        self.step = 0

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
            instruction_ids, output_ids = self._truncating_strategy(instruction_ids, output_ids)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            tokens[i, :instr_len + output_len] = torch.tensor(instruction_ids + output_ids).long()
            labels[i, instr_len - 1: instr_len - 1 + output_len] = torch.tensor(output_ids).long()
        Output = collections.namedtuple('Outputs', ['tokens', 'labels'])
        return Output(tokens=tokens, labels=labels)

    def predict(self, logits, instructions: List[str], outputs: List[str]) -> List[dict]:
        bzs = int(logits.shape[0])
        datalist = []
        for i in range(bzs):
            instruction_ids = self.tokenizer.tokenize(instructions[i], bos=True)
            output_ids = self.tokenizer.tokenize(outputs[i], eos=True)
            instruction_ids, output_ids = self._truncating_strategy(instruction_ids, output_ids)
            instr_len, output_len = len(instruction_ids), len(output_ids)
            predict_ids = torch.argmax(logits[i], dim=-1)[instr_len - 1: instr_len - 1 + output_len].tolist()
            datalist.append(dict(instruction=instructions[i], output=self.tokenizer.decode(predict_ids)))
        return datalist

    def forward(self, instructions: List[str], outputs: List[str]):
        """ Instruction tuning """
        example = self.prepare_for_training(instructions=instructions, outputs=outputs)
        logits = self.model.forward(example.tokens)
        loss = self.criterion.forward(
            input=logits.view(-1, logits.size(-1)),
            target=example.labels.view(-1).to(logits.device)
        )
        self._back_propagation(loss)
        Output = collections.namedtuple('Output', ['loss', 'logits'])
        return Output(logits=logits, loss=loss)


class DistributedVerifierTrainer(DistributedTrainer):
    def __init__(
            self,
            model: LoraLlamaVerifier,
            tokenizer: LlamaTokenizer,
            optimizer: torch.optim.Optimizer,
            accumulation_steps: int = 1
    ):
        super().__init__(model, optimizer)
        self.model = model
        self.local_rank = model.local_rank
        self.world_size = model.world_size
        self.max_seq_len = self.model.params.max_seq_len
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

    def prepare_for_training(self, instructions: List[str], outputs: List[str]):
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
        self.model.train()
        c_examples = self.prepare_for_training(instructions, chosen)
        r_examples = self.prepare_for_training(instructions, rejected)
        c_rewards = self.model.forward(c_examples.tokens)
        r_rewards = self.model.forward(r_examples.tokens)

        loss = self.criterion.forward(
            chosen_rewards=c_rewards,
            rejected_rewards=r_rewards,
            chosen_masks=c_examples.masks.to(c_rewards.device),
            rejected_masks=r_examples.masks.to(r_rewards.device)
        )
        self._back_propagation(loss)

        Output = collections.namedtuple('Output', ['loss'])
        return Output(loss=loss)


# class DistributedDistillingSolverTrainer(DistributedSolverTrainer):
#     def __init__(
#             self,
#             model: AbstractLlama,
#             tokenizer: LlamaTokenizer,
#             optimizer: torch.optim.Optimizer,
#             accumulation_steps: int = 1
#     ):
#         super().__init__(model, tokenizer, optimizer, accumulation_steps)
#         self.criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
#         self.criterion_kl = KLDivLoss()
#
#     def prepare_for_distilling(self, instructions, outputs, logits_dicts_list):
#         bsz = len(instructions)
#         labels = torch.full((bsz, self.max_seq_len), -100).long()
#         logits = torch.full((bsz, self.max_seq_len, self.tokenizer.n_words), 0).float()
#         for i, (instruction, output, logits_dicts) in enumerate(zip(instructions, outputs, logits_dicts_list)):
#             instruction_ids = self.tokenizer.encode(instruction, bos=True, eos=False)
#             output_ids = self.tokenizer.encode(output, bos=False, eos=True)
#             instruction_ids, output_ids = self._truncating_strategy(instruction_ids, output_ids)
#             instr_len, output_len = len(instruction_ids), len(output_ids)
#             logits_dicts = logits_dicts[: output_len]
#             assert output_len == len(logits_dicts)
#             logits[i, instr_len - 1: instr_len - 1 + output_len, :] = reconstruct_logits_from_dicts(
#                 logits_dicts, self.tokenizer.n_words)
#             labels[i, instr_len - 1: instr_len - 1 + output_len] = torch.tensor(output_ids).long()
#         label_masks = (labels != -100)
#         probs = torch.softmax(logits, dim=-1)
#         Output = collections.namedtuple('Outputs', ['teacher_logits', 'teacher_probs', 'label_masks'])
#         return Output(teacher_logits=logits, teacher_probs=probs, label_masks=label_masks)
#
#     def forward(
#             self,
#             instructions: List[str],
#             outputs: List[str],
#             logits_dicts_list: List[dict],
#             alpha: float,
#             logits_dicts_list2: List[dict] = None,
#             beta: float = None
#     ):
#         forward_example = self.prepare_for_training(instructions=instructions, outputs=outputs)
#         logits = self.model.forward(forward_example.tokens)
#         ce_loss = self.criterion_ce.forward(
#             input=logits.view(-1, logits.size(-1)),
#             target=forward_example.labels.view(-1).to(logits.device)
#         )
#
#         distill_example = self.prepare_for_distilling(
#             instructions, outputs, logits_dicts_list
#         )
#         distill_loss = self.criterion_kl.forward(
#             logits=logits,
#             targets=distill_example.teacher_probs.to(logits.device),
#             masks=distill_example.label_masks.to(logits.device)
#         )
#         loss = ce_loss + alpha * distill_loss
#
#         distill_loss2 = None
#         if logits_dicts_list2 is not None:
#             distill_example2 = self.prepare_for_distilling(
#                 instructions, outputs, logits_dicts_list2
#             )
#             distill_loss2 = self.criterion_kl.forward(
#                 logits=logits,
#                 targets=distill_example2.teacher_probs.to(logits.device),
#                 masks=distill_example2.label_masks.to(logits.device)
#             )
#             loss = loss + beta * distill_loss2
#
#         self._back_propagation(loss)
#         Output = collections.namedtuple('Output', ['loss', 'distill_loss', 'logits', 'distill_loss2'])
#         return Output(logits=logits, distill_loss=distill_loss, loss=ce_loss, distill_loss2=distill_loss2)
