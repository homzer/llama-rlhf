import os
from collections import OrderedDict
from pathlib import Path
from typing import List

import safetensors
import torch

from src.parallel.utils import set_barrier


class Checkpoint:
    def __init__(self, col_parallel_names: List[str], row_parallel_names: List[str]):
        self.col_parallel_names = col_parallel_names
        self.row_parallel_names = row_parallel_names

    def is_col_parallel(self, name: str) -> bool:
        for col_parallel_name in self.col_parallel_names:
            if col_parallel_name in name:
                return True
        return False

    def is_row_parallel(self, name: str) -> bool:
        for row_parallel_name in self.row_parallel_names:
            if row_parallel_name in name:
                return True
        return False

    @classmethod
    def load_hf(cls, ckpt_files: List[str]) -> dict:
        state_dict = OrderedDict()
        for ckpt_file in ckpt_files:
            ckpt_file = str(ckpt_file)
            if ckpt_file.endswith(".safetensors"):
                with safetensors.safe_open(ckpt_file, "pt", device="cpu") as reader:
                    for k in reader.keys():
                        state_dict[k] = reader.get_tensor(k)
            elif ckpt_file.endswith(".bin"):
                reader = torch.load(ckpt_file, map_location="cpu")
                for k in reader.keys():
                    state_dict[k] = reader[k]
            else:
                raise TypeError(ckpt_file)
        return state_dict

    def split(self, state_dict: dict, n: int) -> List[dict]:
        new_state_dicts = [OrderedDict() for _ in range(n)]
        for name, param in state_dict.items():
            assert 'lora' not in name, 'can not split a lora checkpoint, merge it first'
            param = param.cpu()
            if self.is_col_parallel(name):
                dim0 = param.shape[0]
                assert dim0 % n == 0
                split = dim0 // n
                for i in range(n):
                    new_state_dicts[i][name] = param[i * split: (i + 1) * split].clone()
            elif self.is_row_parallel(name):
                dim1 = param.shape[1]
                assert dim1 % n == 0
                split = dim1 // n
                for i in range(n):
                    new_state_dicts[i][name] = param[:, i * split: (i + 1) * split].clone()
            else:
                for i in range(n):
                    new_state_dicts[i][name] = param.clone()
            # release memory
            state_dict[name] = None

        return new_state_dicts

    def merge(self, state_dict1: dict, state_dict2: dict) -> dict:
        new_state_dicts = OrderedDict()
        for name in state_dict1.keys():
            assert 'lora' not in name, 'can not split a lora checkpoint, merge it first'
            param1 = state_dict1[name]
            param2 = state_dict2[name]
            if self.is_col_parallel(name):
                new_state_dicts[name] = torch.cat([param1, param2], dim=0).clone()
            elif self.is_row_parallel(name):
                assert len(param1.shape) == 2
                assert len(param2.shape) == 2
                new_state_dicts[name] = torch.cat([param1, param2], dim=1).clone()
            else:
                new_state_dicts[name] = state_dict1[name].clone()
            # release memory
            state_dict1[name] = None
            state_dict2[name] = None

        return new_state_dicts

    def auto_merge_n_to_1(self, state_dicts: List[dict]) -> dict:
        if len(state_dicts) <= 1:
            return state_dicts[0]
        assert len(state_dicts) % 2 == 0, "the number of merged `state_dicts` must be even."
        merge_state_dicts = []
        for state_dict_1, state_dict_2 in zip(state_dicts[::2], state_dicts[1::2]):
            merge_state_dicts.append(self.merge(state_dict_1, state_dict_2))
        return self.auto_merge_n_to_1(merge_state_dicts)

    @classmethod
    def merge_lora_state_dict(cls, state_dict: dict) -> dict:
        res_state_dict = {}
        with torch.no_grad():
            for name, param in state_dict.items():
                if 'lora' not in name:
                    res_state_dict[name] = param
                elif 'lora_a_' in name:
                    origin = name.replace('lora_a_', '')
                    original_dtype = state_dict[origin].dtype
                    w = state_dict[origin].float()
                    wa = state_dict[name].float()
                    wb = state_dict[name.replace('lora_a_', 'lora_b_')].float()
                    res_state_dict[origin] = (w + wb @ wa).to(original_dtype)
                    state_dict[origin] = None  # free memory
        return res_state_dict

    def auto_split_huggingface_checkpoints(
            self,
            ckpt_dir: str,
            model_parallel_world_size: int,
            global_rank: int,
            verbose: bool = True
    ) -> str:
        pl_ckpt_dir = os.path.join(ckpt_dir, str(model_parallel_world_size))
        if global_rank == 0 and not os.path.exists(pl_ckpt_dir):
            if verbose:
                print(f'Parallel checkpoint dose not exist. Splitting into {pl_ckpt_dir} ...')
            if os.path.exists(os.path.join(ckpt_dir, "pytorch_model.bin")):
                split_files = [os.path.join(ckpt_dir, "pytorch_model.bin")]
            else:
                split_files = sorted(Path(ckpt_dir).glob("model*.safetensors"))
                if len(split_files) == 0:
                    split_files = sorted(Path(ckpt_dir).glob("pytorch_model*.bin"))
                    if len(split_files) == 0:
                        raise FileNotFoundError("Can not find any checkpoint file")
            state_dict = self.load_hf(split_files)
            new_state_dicts = self.split(state_dict, model_parallel_world_size)
            os.makedirs(pl_ckpt_dir, exist_ok=True)
            for i in range(model_parallel_world_size):
                torch.save(new_state_dicts[i], os.path.join(pl_ckpt_dir, 'consolidated.%02d.pth' % i))
            if verbose:
                print('Done!')
        return pl_ckpt_dir

    def auto_split_consolidate_checkpoints(
            self,
            ckpt_dir: str,
            model_parallel_world_size: int,
            global_rank: int,
            verbose: bool = True
    ) -> str:
        pl_ckpt_dir = os.path.join(ckpt_dir, str(model_parallel_world_size))
        if global_rank == 0 and not os.path.exists(pl_ckpt_dir):
            if verbose:
                print(f'Parallel checkpoint dose not exist. Splitting into {pl_ckpt_dir} ...')
            checkpoints = sorted(Path(ckpt_dir).glob("consolidated.*.pth"))
            assert len(checkpoints) > 0
            # merge to 1 first and then split to `model_parallel_world_size`
            state_dicts = [torch.load(ckpt_file, map_location="cpu") for ckpt_file in checkpoints]
            merge_state_dict = self.auto_merge_n_to_1(state_dicts)

            new_state_dicts = self.split(merge_state_dict, model_parallel_world_size)
            os.makedirs(pl_ckpt_dir, exist_ok=True)
            for i in range(model_parallel_world_size):
                torch.save(new_state_dicts[i], os.path.join(pl_ckpt_dir, 'consolidated.%02d.pth' % i))
            if verbose:
                print('Done!')
        return pl_ckpt_dir

    def auto_split_or_merge_checkpoints(
            self,
            ckpt_dir: str,
            model_parallel_world_size: int,
            global_rank: int
    ) -> str:
        checkpoints = sorted(Path(ckpt_dir).glob("consolidated.*.pth"))
        if len(checkpoints) == 0:  # splitting
            ckpt_dir = self.auto_split_huggingface_checkpoints(
                ckpt_dir,
                model_parallel_world_size=model_parallel_world_size,
                global_rank=global_rank
            )
        elif len(checkpoints) != model_parallel_world_size:
            ckpt_dir = self.auto_split_consolidate_checkpoints(
                ckpt_dir,
                model_parallel_world_size=model_parallel_world_size,
                global_rank=global_rank
            )
        set_barrier()
        return ckpt_dir


class CheckpointForLlama(Checkpoint):
    def __init__(self):
        col_parallel_names = [
            "wq.weight", "wk.weight", "wv.weight", "w1.weight", "w3.weight", "output.weight",
            "wq.bias", "wk.bias", "wv.bias", "w1.bias", "w3.bias", "output.bias",
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight", "lm_head.weight",
            "q_proj.bias", "k_proj.bias", "v_proj.bias", "gate_proj.bias", "up_proj.bias", "lm_head.bias"
        ]
        row_parallel_names = [
            "wo.weight", "w2.weight", "tok_embeddings.weight",
            "o_proj.weight", "down_proj.weight", "embed_tokens.weight",
        ]
        super().__init__(col_parallel_names, row_parallel_names)


class CheckpointForLlamaHf(Checkpoint):
    def __init__(self):
        col_parallel_names = [
            "wq.weight", "wk.weight", "wv.weight", "w1.weight", "w3.weight", "output.weight",
            "wq.bias", "wk.bias", "wv.bias", "w1.bias", "w3.bias", "output.bias",
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight",
            "q_proj.bias", "k_proj.bias", "v_proj.bias", "gate_proj.bias", "up_proj.bias",
        ]
        row_parallel_names = [
            "wo.weight", "w2.weight", "tok_embeddings.weight",
            "o_proj.weight", "down_proj.weight", "embed_tokens.weight", "lm_head.weight", "lm_head.bias"
        ]
        super().__init__(col_parallel_names, row_parallel_names)


class CheckpointForLlama3(Checkpoint):
    def __init__(self):
        col_parallel_names = [
            "wq.weight", "wk.weight", "wv.weight", "w1.weight", "w3.weight", "output.weight",
            "wq.bias", "wk.bias", "wv.bias", "w1.bias", "w3.bias", "output.bias",
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight", "lm_head.weight",
            "q_proj.bias", "k_proj.bias", "v_proj.bias", "gate_proj.bias", "up_proj.bias", "lm_head.bias",
            "tok_embeddings.weight", "embed_tokens.weight"
        ]
        row_parallel_names = [
            "wo.weight", "w2.weight", "o_proj.weight", "down_proj.weight"
        ]
        super().__init__(col_parallel_names, row_parallel_names)


class CheckpointForQwen(Checkpoint):
    def __init__(self):
        col_parallel_names = [
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight", "lm_head.weight",
            "q_proj.bias", "k_proj.bias", "v_proj.bias", "gate_proj.bias", "up_proj.bias", "lm_head.bias"
        ]
        row_parallel_names = [
            "o_proj.weight", "down_proj.weight", "embed_tokens.weight",
        ]
        super().__init__(col_parallel_names, row_parallel_names)


class CheckpointForInternLM(Checkpoint):
    def __init__(self):
        col_parallel_names = [
            "wq.weight", "wk.weight", "wv.weight", "w1.weight", "w3.weight", "output.weight",
            "wq.bias", "wk.bias", "wv.bias", "w1.bias", "w3.bias", "output.bias",
        ]
        row_parallel_names = [
            "wo.weight", "w2.weight", "tok_embeddings.weight",
        ]
        super().__init__(col_parallel_names, row_parallel_names)
        self.w_pack_name = "wqkv"

    def process_w_pack(self, state_dict: dict) -> dict:
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            if self.w_pack_name in name:
                h = param.shape[1]
                assert (param.shape[0] - h) % 2 == 0
                new_state_dict[name.replace(self.w_pack_name, "wq")] = param[: h].clone()
                new_state_dict[name.replace(self.w_pack_name, "wk")] = param[h: h + (param.shape[0] - h) // 2].clone()
                new_state_dict[name.replace(self.w_pack_name, "wv")] = param[h + (param.shape[0] - h) // 2:].clone()
            else:
                new_state_dict[name] = param
        return new_state_dict

    def split(self, state_dict: dict, n: int) -> List[dict]:
        state_dict = self.process_w_pack(state_dict)
        new_state_dicts = [OrderedDict() for _ in range(n)]
        for name, param in state_dict.items():
            assert 'lora' not in name, 'can not split a lora checkpoint, merge it first'
            param = param.cpu()
            if self.is_col_parallel(name):
                dim0 = param.shape[0]
                assert dim0 % n == 0
                split = dim0 // n
                for i in range(n):
                    new_state_dicts[i][name] = param[i * split: (i + 1) * split].clone()
            elif self.is_row_parallel(name):
                dim1 = param.shape[1]
                assert dim1 % n == 0
                split = dim1 // n
                for i in range(n):
                    new_state_dicts[i][name] = param[:, i * split: (i + 1) * split].clone()
            else:
                for i in range(n):
                    new_state_dicts[i][name] = param.clone()
            # release memory
            state_dict[name] = None

        return new_state_dicts


class CheckpointForBaichuan(Checkpoint):
    def __init__(self):
        col_parallel_names = [
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight",
            "q_proj.bias", "k_proj.bias", "v_proj.bias", "gate_proj.bias", "up_proj.bias"
        ]
        row_parallel_names = [
            "o_proj.weight", "down_proj.weight", "embed_tokens.weight",
        ]
        super().__init__(col_parallel_names, row_parallel_names)
        self.w_pack_name = "W_pack"

    def process_w_pack(self, state_dict: dict) -> dict:
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            if self.w_pack_name in name:
                h = param.shape[1]
                assert param.shape[0] / h == 3
                new_state_dict[name.replace(self.w_pack_name, "q_proj")] = param[: h].clone()
                new_state_dict[name.replace(self.w_pack_name, "k_proj")] = param[h: 2 * h].clone()
                new_state_dict[name.replace(self.w_pack_name, "v_proj")] = param[2 * h:].clone()
            else:
                new_state_dict[name] = param
        return new_state_dict

    def split(self, state_dict: dict, n: int) -> List[dict]:
        state_dict = self.process_w_pack(state_dict)
        new_state_dicts = [OrderedDict() for _ in range(n)]
        for name, param in state_dict.items():
            assert 'lora' not in name, 'can not split a lora checkpoint, merge it first'
            param = param.cpu()
            if self.is_col_parallel(name):
                dim0 = param.shape[0]
                assert dim0 % n == 0
                split = dim0 // n
                for i in range(n):
                    new_state_dicts[i][name] = param[i * split: (i + 1) * split].clone()
            elif self.is_row_parallel(name):
                dim1 = param.shape[1]
                assert dim1 % n == 0
                split = dim1 // n
                for i in range(n):
                    new_state_dicts[i][name] = param[:, i * split: (i + 1) * split].clone()
            else:
                for i in range(n):
                    new_state_dicts[i][name] = param.clone()
            # release memory
            state_dict[name] = None

        return new_state_dicts
