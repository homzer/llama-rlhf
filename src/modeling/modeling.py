import collections
import os
from pathlib import Path

import torch
import torch.nn as nn

from src.utils import set_barrier

CausalLMOutputs = collections.namedtuple('CausalLMOutputs', ['logits', 'hidden_states'])
Seq2SeqLMOutputs = collections.namedtuple('Seq2SeqLMOutputs', ['logits'])
MaskedLMOutputs = collections.namedtuple('MaskedLMOutputs', ['logits'])


class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def device(self):
        return next(self.parameters()).device

    def load(self, ckpt_file: str, verbose: bool = True):
        if verbose:
            print(f'Loading model from {ckpt_file} .....')
        state_dict = torch.load(ckpt_file, map_location='cpu')
        outputs = self.load_state_dict(state_dict, strict=False)
        if verbose:
            for missing_key in outputs.missing_keys:
                print(f"MISSING KEY: {missing_key}")
            for unexpected_key in outputs.unexpected_keys:
                print(f"UNEXPECTED KEY: {unexpected_key}")
            print("Loading done!")
        return self

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f'Saving model to {save_dir} ......')
        torch.save(self.state_dict(), os.path.join(save_dir, f'pytorch_model.bin'))
        print(f'Saving done !')


# Decoder-only
class ModelForCausalLM(Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ) -> CausalLMOutputs:
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError


# Encoder-only
class ModelForMaskedLM(Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            tokens: torch.Tensor,
    ) -> MaskedLMOutputs:
        raise NotImplementedError


# Encoder-decoder
class ModelForSeq2SeqLM(Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            input_ids: torch.Tensor,
            decoder_input_ids: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            use_cache: bool = False,
            start_pos: int = None
    ) -> Seq2SeqLMOutputs:
        raise NotImplementedError


class ParallelModule(Module):
    def __init__(self, local_rank, world_size):
        super().__init__()
        self.local_rank = local_rank
        self.world_size = world_size

    def load(self, ckpt_dir: str, verbose: bool = True):
        if verbose:
            print(f'Loading model from {ckpt_dir} .....')
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert self.world_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {self.world_size}"
        ckpt_path = checkpoints[self.local_rank]
        state_dict = torch.load(ckpt_path, map_location="cpu")
        outputs = self.load_state_dict(state_dict, strict=False)
        self.cuda(self.local_rank)
        set_barrier()
        if verbose:
            for missing_key in outputs.missing_keys:
                print(f"MISSING KEY: {missing_key}")
            for unexpected_key in outputs.unexpected_keys:
                print(f"UNEXPECTED KEY: {unexpected_key}")
            print(f'Loading done !')

    def save(self, save_path):
        if self.local_rank == 0:
            os.makedirs(save_path, exist_ok=True)
        print(f'Saving model to {save_path} ......')
        set_barrier()
        torch.save(self.state_dict(), os.path.join(save_path, f'consolidated.0{self.local_rank}.pth'))
        set_barrier()
        print(f'Saving done !')


class ParallelModelForCausalLM(ParallelModule):
    def __init__(self, local_rank, world_size):
        super().__init__(local_rank, world_size)

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ) -> CausalLMOutputs:
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError


class ParallelModelForMaskedLM(ParallelModule):
    def __init__(self, local_rank, world_size):
        super().__init__(local_rank, world_size)

    def forward(
            self,
            tokens: torch.Tensor,
    ) -> MaskedLMOutputs:
        raise NotImplementedError


class ParallelModelForSeq2SeqLM(ParallelModule):
    def __init__(self, local_rank, world_size):
        super().__init__(local_rank, world_size)

    def forward(
            self,
            input_ids: torch.Tensor,
            decoder_input_ids: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            use_cache: bool = False,
            start_pos: int = None
    ) -> Seq2SeqLMOutputs:
        raise NotImplementedError
