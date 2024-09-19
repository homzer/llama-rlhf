import collections
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_world_size,
    get_model_parallel_rank,
    get_model_parallel_src_rank
)

from src.checkpoint import Checkpoint
from src.utils import load_safetensors
from src.parallel.utils import set_barrier

CausalLMOutputs = collections.namedtuple('CausalLMOutputs', ['logits', 'hidden_states'])
Seq2SeqLMOutputs = collections.namedtuple('Seq2SeqLMOutputs', ['logits', 'hidden_states'])
MaskedLMOutputs = collections.namedtuple('MaskedLMOutputs', ['logits', 'hidden_states'])
VerifierOutputs = collections.namedtuple('VerifierOutputs', ['scores'])


class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def device(self):
        return next(self.parameters()).device

    def init_weights(self):
        raise NotImplementedError

    def load(self, ckpt_file: str, **kwargs):
        if kwargs.get("verbose", True):
            print(f'Loading model from {ckpt_file} .....')
        if ckpt_file.endswith(".safetensors"):
            state_dict = load_safetensors(ckpt_file)
        else:
            state_dict = torch.load(ckpt_file, map_location='cpu')
        outputs = self.load_state_dict(state_dict, strict=False)
        if kwargs.get("verbose", True):
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

    def init_weights(self):
        raise NotImplementedError

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ) -> CausalLMOutputs:
        raise NotImplementedError

    def flush(self):
        """ clean KV cache """
        raise NotImplementedError

    def rearrange_kv_cache(self, indices: torch.Tensor):
        """ rearrange the order of the KV cache """
        pass


# Encoder-only
class ModelForMaskedLM(Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        raise NotImplementedError

    def forward(
            self,
            tokens: torch.Tensor,
    ) -> MaskedLMOutputs:
        raise NotImplementedError


# Encoder-decoder
class ModelForSeq2SeqLM(Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        raise NotImplementedError

    def forward(
            self,
            input_ids: torch.Tensor,
            decoder_input_ids: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            use_cache: bool = False,
            start_pos: int = None
    ) -> Seq2SeqLMOutputs:
        raise NotImplementedError


class Verifier(Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        raise NotImplementedError

    def forward(self, tokens: torch.Tensor) -> VerifierOutputs:
        raise NotImplementedError


class ParallelModule(Module):
    def __init__(self):
        super().__init__()
        self.global_rank = int(os.environ.get("RANK"))
        self.local_rank = int(os.environ.get("LOCAL_RANK"))
        self.world_size = int(os.environ.get("WORLD_SIZE"))
        self.model_parallel_world_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()  # rank in group
        self.model_parallel_src_rank = get_model_parallel_src_rank()

    def init_weights(self):
        raise NotImplementedError

    def load(self, ckpt_dir: str, **kwargs):
        if kwargs.get("verbose", True):
            print(f'Loading model from {ckpt_dir} .....')
        checkpoints = sorted(Path(ckpt_dir).glob("consolidated.*.pth"))
        assert self.model_parallel_world_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but model parallel size is {self.model_parallel_world_size}"
        ckpt_path = checkpoints[self.model_parallel_rank]
        loading_outputs = None
        if kwargs.get("sequential_load", False):  # For saving cpu memory
            for i in range(self.model_parallel_world_size):
                if i == self.model_parallel_rank:
                    state_dict = torch.load(str(ckpt_path), map_location="cpu")
                    if kwargs.get("merge_lora", False):
                        state_dict = Checkpoint.merge_lora_state_dict(state_dict)
                    loading_outputs = self.load_state_dict(state_dict, strict=False)
                    self.cuda(self.local_rank)
                set_barrier()
        else:
            state_dict = torch.load(str(ckpt_path), map_location="cpu")
            if kwargs.get("merge_lora", False):
                state_dict = Checkpoint.merge_lora_state_dict(state_dict)
            loading_outputs = self.load_state_dict(state_dict, strict=False)
            self.cuda(self.local_rank)
        set_barrier()
        if kwargs.get("verbose", True):
            for missing_key in loading_outputs.missing_keys:
                print(f"MISSING KEY: {missing_key}")
            for unexpected_key in loading_outputs.unexpected_keys:
                print(f"UNEXPECTED KEY: {unexpected_key}")
            print(f'Loading done !')

    def save(self, save_path):
        print(f'Saving model to {save_path} ......')
        if self.model_parallel_src_rank == 0:
            if self.model_parallel_rank == 0:
                os.makedirs(save_path, exist_ok=True)
            set_barrier()
            torch.save(self.state_dict(), os.path.join(save_path, 'consolidated.%02d.pth' % self.model_parallel_rank))
        set_barrier()
        print(f'Saving done !')


class ParallelModelForCausalLM(ParallelModule):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        raise NotImplementedError

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ) -> CausalLMOutputs:
        raise NotImplementedError

    def flush(self):
        """ clean KV cache """
        raise NotImplementedError

    def rearrange_kv_cache(self, indices: torch.Tensor):
        """ rearrange the order of the KV cache """
        pass


class ParallelModelForMaskedLM(ParallelModule):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        raise NotImplementedError

    def forward(
            self,
            tokens: torch.Tensor,
    ) -> MaskedLMOutputs:
        raise NotImplementedError


class ParallelModelForSeq2SeqLM(ParallelModule):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        raise NotImplementedError

    def forward(
            self,
            input_ids: torch.Tensor,
            decoder_input_ids: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            use_cache: bool = False,
            start_pos: int = None
    ) -> Seq2SeqLMOutputs:
        raise NotImplementedError


class ParallelVerifier(ParallelModule):
    def __init__(self):
        super().__init__()

    def forward(self, tokens: torch.Tensor) -> VerifierOutputs:
        raise NotImplementedError


class AttentionForCausalLM(nn.Module):
    def __init__(self, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.cache_k = None
        self.cache_v = None

    def apply_cache(self, xk, xv, start_pos):
        bsz, seq_len, n_heads, head_dim = xk.shape
        if self.cache_k is None:
            delattr(self, 'cache_k')
            self.register_buffer(
                name='cache_k',
                tensor=torch.zeros((bsz, self.max_seq_len, n_heads, head_dim)),
                persistent=False
            )
        if self.cache_v is None:
            delattr(self, 'cache_v')
            self.register_buffer(
                name='cache_v',
                tensor=torch.zeros((bsz, self.max_seq_len, n_heads, head_dim)),
                persistent=False
            )

        self.cache_k = self.cache_k.to(xk)
        self.cache_v = self.cache_v.to(xv)
        self.cache_k[:bsz, start_pos: start_pos + seq_len] = xk
        self.cache_v[:bsz, start_pos: start_pos + seq_len] = xv

        xk = self.cache_k[:bsz, : start_pos + seq_len]
        xv = self.cache_v[:bsz, : start_pos + seq_len]
        return xk, xv

    @staticmethod
    def apply_attention(xq, xk, xv, mask):
        bsz, seqlen, n_heads, head_dim = xq.shape
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        if scores.dtype == torch.float16:
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        else:
            scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, xv)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return output

    def flush(self):
        """ Clean cache for next inference. """
        self.cache_v = None
        self.cache_k = None

    def rearrange(self, indices: torch.Tensor):
        """
        rearrange the order of kv cache
        :param indices: [batch_size]
        :return:
        """
        assert self.cache_k is not None
        assert self.cache_v is not None
        assert len(indices.shape) == 1
        self.cache_k = self.cache_k[indices]
        self.cache_v = self.cache_v[indices]
