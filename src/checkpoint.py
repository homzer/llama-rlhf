import os
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import List

import fire
import safetensors
import torch
from safetensors.torch import save_file

from src.parallel.initialize import set_barrier
from src.utils import json_load, json_dump


class Checkpoint:
    def __init__(self, col_parallel_names: List[str], row_parallel_names: List[str]):
        self.col_parallel_names = col_parallel_names
        self.row_parallel_names = row_parallel_names

    def is_col_parallel(self, name: str) -> bool:
        for col_parallel_name in self.col_parallel_names:
            if re.search(f"{col_parallel_name}$", name):
                return True
        return False

    def is_row_parallel(self, name: str) -> bool:
        for row_parallel_name in self.row_parallel_names:
            if re.search(f"{row_parallel_name}$", name):
                return True
        return False

    @classmethod
    def load_hf(cls, ckpt_files: List[str]) -> dict:
        state_dict = OrderedDict()
        for ckpt_file in ckpt_files:
            ckpt_file = str(ckpt_file)
            print(f"Loading {ckpt_file} ...")
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

    @classmethod
    def save_hf(cls, state_dict: dict, save_dir: str):
        print(f"Saving to {save_dir} ...")
        save_file(state_dict, os.path.join(save_dir, "model.safetensors"))

    def split(self, state_dict: dict, n: int) -> List[dict]:
        if n == 1:
            return [state_dict]
        # Auto merge lora
        state_dict = self.merge_lora_state_dict(state_dict)
        new_state_dicts = [OrderedDict() for _ in range(n)]
        with torch.no_grad():
            for name, param in state_dict.items():
                # This case is unlikely to be reached. Keep the logic here just in case.
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
                        new_state_dicts[i][name] = param
                # release memory
                state_dict[name] = None

        return new_state_dicts

    def merge(self, state_dict1: dict, state_dict2: dict) -> dict:
        # Auto merge lora
        state_dict1 = self.merge_lora_state_dict(state_dict1)
        state_dict2 = self.merge_lora_state_dict(state_dict2)
        new_state_dict = OrderedDict()
        with torch.no_grad():
            for name in state_dict1.keys():
                # This case is unlikely to be reached. Keep the logic here just in case.
                assert 'lora' not in name, 'can not merge a lora checkpoint, merge it first'
                param1 = state_dict1[name]
                param2 = state_dict2[name]
                if self.is_col_parallel(name):
                    new_state_dict[name] = torch.cat([param1, param2], dim=0)
                elif self.is_row_parallel(name):
                    assert len(param1.shape) == 2
                    assert len(param2.shape) == 2
                    new_state_dict[name] = torch.cat([param1, param2], dim=1)
                else:
                    new_state_dict[name] = state_dict1[name]
                # release memory
                state_dict1[name] = None
                state_dict2[name] = None

        return new_state_dict

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
        results = {}
        with torch.no_grad():
            lora_a_keys = []
            for name in state_dict:
                if 'lora_a_' in name:
                    lora_a_keys.append(name)
            if len(lora_a_keys) == 0:
                return state_dict
            for name, param in state_dict.items():
                if "lora_a_" not in name and "lora_b_" not in name:
                    results[name] = param
                    state_dict[name] = None
            for lora_a_key in lora_a_keys:
                origin_key = lora_a_key.replace("lora_a_", "")
                lora_b_key = lora_a_key.replace("lora_a_", "lora_b_")
                wa = state_dict[lora_a_key]
                wb = state_dict[lora_b_key]
                results[origin_key] = results[origin_key] + (wb.float() @ wa.float()).to(results[origin_key].dtype)
                state_dict[lora_a_key] = None
                state_dict[lora_b_key] = None
        return results

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
            global_rank: int = 0,
            verbose: bool = True
    ) -> str:
        pl_ckpt_dir = os.path.join(ckpt_dir, str(model_parallel_world_size))
        if global_rank == 0 and not os.path.exists(pl_ckpt_dir):
            if verbose:
                print(f'Parallel checkpoint dose not exist. Splitting into {pl_ckpt_dir} ...')
            checkpoints = sorted(Path(ckpt_dir).glob("consolidated.*.pth"))
            assert len(checkpoints) > 0
            # merge to 1 first and then split to `model_parallel_world_size`
            print(f"Loading {len(checkpoints)} checkpoints from {ckpt_dir} ...")
            state_dicts = [torch.load(ckpt_file, map_location="cpu") for ckpt_file in checkpoints]
            print(f"Merging {len(checkpoints)} checkpoints to 1 checkpoint ...")
            state_dicts = self.auto_merge_n_to_1(state_dicts)
            print(f"Split into {model_parallel_world_size} checkpoints ...")

            state_dicts = self.split(state_dicts, model_parallel_world_size)
            os.makedirs(pl_ckpt_dir, exist_ok=True)
            for i in range(model_parallel_world_size):
                torch.save(state_dicts[i], os.path.join(pl_ckpt_dir, 'consolidated.%02d.pth' % i))
            if verbose:
                print('Done!')
        return pl_ckpt_dir

    def auto_split_or_merge_checkpoints(
            self,
            ckpt_dir: str,
            model_parallel_world_size: int | torch.Tensor,
            global_rank: int | torch.Tensor
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

    @classmethod
    def show(cls, checkpoint_file):
        if checkpoint_file.endswith(".safetensors"):
            state_dict = cls.load_hf([checkpoint_file])
        else:
            state_dict = torch.load(checkpoint_file, map_location="cpu")
        for name, param in state_dict.items():
            print(name, param.shape)


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
        self.replace_names = {
            "tok_embeddings.": "embed_tokens.",
            ".feed_forward.w1.": ".mlp.gate_proj.",
            ".feed_forward.w2.": ".mlp.down_proj.",
            ".feed_forward.w3.": ".mlp.up_proj.",
            ".attention_norm.": ".input_layernorm.",
            ".ffn_norm.": ".post_attention_layernorm.",
            ".attention.wq.": ".self_attn.q_proj.",
            ".attention.wk.": ".self_attn.k_proj.",
            ".attention.wv.": ".self_attn.v_proj.",
            ".attention.wo.": ".self_attn.o_proj.",
        }

    def _rename_consolidate_to_huggingface(self, state_dict, head_dim: int) -> dict:
        def permute(_w, _head_dim):
            _dim1, _dim2 = _w.shape
            return _w.view(_dim1 // _head_dim, _head_dim // 2, 2, _dim2).transpose(1, 2).reshape(_dim1, _dim2)
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            old_name = name
            if name == "output.weight":
                param = param.clone()
                name = "lm_head.weight"
            else:
                for source, target in self.replace_names.items():
                    name = name.replace(source, target)
                name = "model." + name
            print(old_name, "------>", name)
            if 'q_proj' in name or 'k_proj' in name:
                param = permute(param, head_dim)
            new_state_dict[name] = param
        return new_state_dict

    def convert_consolidate_to_huggingface(self, ckpt_dir: str, hf_config_dir: str, save_dir: str = None, head_dim: int = None):
        """
        Convert consolidated checkpoints into huggingface ones.
        :param head_dim: attention head dimension.
        :param ckpt_dir: path to the consolidate checkpoints dir.
        :param hf_config_dir: huggingface config dir.
        :param save_dir: default to `None`.
        """
        ckpt_dir = self.auto_split_consolidate_checkpoints(
            ckpt_dir=ckpt_dir,
            model_parallel_world_size=1,
            global_rank=0
        )
        config = json_load(os.path.join(hf_config_dir, "config.json"))
        state_dict = self._rename_consolidate_to_huggingface(
            torch.load(os.path.join(ckpt_dir, "consolidated.00.pth")),
            head_dim=config["head_dim"] if "head_dim" in config else head_dim
        )
        ckpt_dir = os.path.join(ckpt_dir, "hf")
        os.makedirs(ckpt_dir, exist_ok=True)
        save_dir = save_dir or ckpt_dir
        self.save_hf(state_dict, save_dir)
        import shutil
        config["tie_word_embeddings"] = False  # untie word embeddings
        json_dump(config, os.path.join(save_dir, "config.json"), indent=2)
        shutil.copy2(os.path.join(hf_config_dir, "generation_config.json"), save_dir)
        shutil.copy2(os.path.join(hf_config_dir, "special_tokens_map.json"), save_dir)
        shutil.copy2(os.path.join(hf_config_dir, "tokenizer_config.json"), save_dir)
        shutil.copy2(os.path.join(hf_config_dir, "tokenizer.json"), save_dir)


class CheckpointForGemma2(Checkpoint):
    def __init__(self):
        col_parallel_names = [
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight", "lm_head.weight",
            "q_proj.bias", "k_proj.bias", "v_proj.bias", "gate_proj.bias", "up_proj.bias", "lm_head.bias"
        ]
        row_parallel_names = [
            "o_proj.weight", "down_proj.weight", "embed_tokens.weight",
        ]
        super().__init__(col_parallel_names, row_parallel_names)

    @classmethod
    def load_hf(cls, ckpt_files: List[str]) -> dict:
        state_dict = Checkpoint.load_hf(ckpt_files)
        if "lm_head.weight" not in state_dict:  # for tie word embeddings
            print("`lm_head.weight` not found in checkpoint, copy `model.embed_tokens.weight` to replace it.")
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"].clone()
        return state_dict


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

    @classmethod
    def load_hf(cls, ckpt_files: List[str]) -> dict:
        state_dict = Checkpoint.load_hf(ckpt_files)
        if "lm_head.weight" not in state_dict:  # for tie word embeddings
            print("`lm_head.weight` not found in checkpoint, copy `model.embed_tokens.weight` to replace it.")
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"].clone()
        return state_dict


class CheckpointForQwen3Moe(Checkpoint):
    def __init__(self):
        col_parallel_names = [
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "lm_head.weight",
            "q_proj.bias", "k_proj.bias", "v_proj.bias", "lm_head.bias"
        ]
        row_parallel_names = [
            "o_proj.weight", "embed_tokens.weight",
        ]
        # TODO
        # col_parallel_names.extend(["gate_proj.weight", "up_proj.weight", "gate_proj.bias", "up_proj.bias"])
        # row_parallel_names.extend(["down_proj.weight"])
        super().__init__(col_parallel_names, row_parallel_names)


class CheckpointForQwenVL(Checkpoint):
    def __init__(self):
        col_parallel_names = [
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight", "lm_head.weight",
            "q_proj.bias", "k_proj.bias", "v_proj.bias", "gate_proj.bias", "up_proj.bias", "lm_head.bias",
            "qkv.weight", "qkv.bias"
        ]
        row_parallel_names = [
            "o_proj.weight", "attn.proj.weight", "down_proj.weight", "embed_tokens.weight",
        ]
        self.col_qkv_parallel_names = ["qkv.weight", "qkv.bias"]
        super().__init__(col_parallel_names, row_parallel_names)

    def is_col_qkv_parallel(self, name: str) -> bool:
        for col_qkv_parallel_name in self.col_qkv_parallel_names:
            if re.search(f"{col_qkv_parallel_name}$", name):
                return True
        return False

    @staticmethod
    def reshape_to_col_qkv_parallel(param: torch.Tensor) -> torch.Tensor:
        # reshape qkv weight from [
        #   [q1],
        #   [q2],
        #   [k1],
        #   [k2],
        #   [v1],
        #   [v2],
        # ] to [
        #   [q1, k1, v1],
        #   [q2, k2, v2],
        # ] in order to apply column parallel splitting
        num_dims = len(param.shape)
        param = torch.reshape(param.cpu(), (3, param.shape[0] // 3, -1)).transpose(-1, -2)
        param = torch.cat(param.unbind(0), dim=0).transpose(-1, -2)
        if num_dims == 1:
            param = param.reshape(-1)
        return param

    @ staticmethod
    def reshape_from_col_qkv_parallel(param: torch.Tensor) -> torch.Tensor:
        # reshape qkv weight from [
        #   [q1, k1, v1],
        #   [q2, k2, v2],
        # ] back to [
        #   [q1],
        #   [q2],
        #   [k1],
        #   [k2],
        #   [v1],
        #   [v2],
        # ] after column parallel splitting
        if len(param.shape) == 2:
            param = param.transpose(-1, -2)
            param = param.reshape(3, param.shape[0] // 3, -1).transpose(-1, -2)
            param = torch.cat(param.unbind(0), dim=0).contiguous()
        return param

    def split(self, state_dict: dict, n: int) -> List[dict]:
        # Auto merge lora
        state_dict = self.merge_lora_state_dict(state_dict)
        with torch.no_grad():
            for name in state_dict:
                if self.is_col_qkv_parallel(name):
                    state_dict[name] = self.reshape_to_col_qkv_parallel(state_dict[name])

        new_state_dicts = super().split(state_dict, n)
        with torch.no_grad():
            for name in new_state_dicts[0]:
                if self.is_col_qkv_parallel(name):
                    for new_state_dict in new_state_dicts:
                        new_state_dict[name] = self.reshape_from_col_qkv_parallel(new_state_dict[name])

        return new_state_dicts

    def merge(self, state_dict1: dict, state_dict2: dict) -> dict:
        # Auto merge lora
        state_dict1 = self.merge_lora_state_dict(state_dict1)
        state_dict2 = self.merge_lora_state_dict(state_dict2)
        with torch.no_grad():
            for name in state_dict1:
                if self.is_col_qkv_parallel(name):
                    state_dict1[name] = self.reshape_to_col_qkv_parallel(state_dict1[name])
                    state_dict2[name] = self.reshape_to_col_qkv_parallel(state_dict2[name])

        new_state_dict = super().merge(state_dict1, state_dict2)
        with torch.no_grad():
            for name in new_state_dict:
                if self.is_col_qkv_parallel(name):
                    new_state_dict[name] = self.reshape_from_col_qkv_parallel(new_state_dict[name])

        return new_state_dict

    @classmethod
    def load_hf(cls, ckpt_files: List[str]) -> dict:
        state_dict = Checkpoint.load_hf(ckpt_files)
        if "lm_head.weight" not in state_dict:  # for tie word embeddings
            print("`lm_head.weight` not found in checkpoint, copy `model.embed_tokens.weight` to replace it.")
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"].clone()
        return state_dict


class CheckpointForQwen3VL(CheckpointForQwenVL):
    def __init__(self):
        super().__init__()
        self.col_parallel_names = [
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight", "lm_head.weight",
            "q_proj.bias", "k_proj.bias", "v_proj.bias", "gate_proj.bias", "up_proj.bias", "lm_head.bias",
            "qkv.weight", "qkv.bias", "mlp.linear_fc1.weight", "mlp.linear_fc1.bias"
        ]
        self.row_parallel_names = [
            "o_proj.weight", "attn.proj.weight", "down_proj.weight", "embed_tokens.weight", "mlp.linear_fc2.weight"
        ]
        self.col_qkv_parallel_names = ["qkv.weight", "qkv.bias"]


class CheckpointForMinistral3(Checkpoint):
    def __init__(self):
        col_parallel_names = [
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight", "lm_head.weight"
        ]
        row_parallel_names = [
            "o_proj.weight", "down_proj.weight", "embed_tokens.weight",
        ]
        super().__init__(col_parallel_names, row_parallel_names)

    @classmethod
    def load_hf(cls, ckpt_files: List[str]) -> dict:
        state_dict = Checkpoint.load_hf(ckpt_files)
        if "lm_head.weight" not in state_dict:  # for tie word embeddings
            print("`lm_head.weight` not found in checkpoint, copy `model.embed_tokens.weight` to replace it.")
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"].clone()
        return state_dict


class CheckpointForMistral3(Checkpoint):
    def __init__(self):
        col_parallel_names = [
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight", "lm_head.weight"
        ]
        row_parallel_names = [
            "o_proj.weight", "down_proj.weight", "embed_tokens.weight",
        ]
        super().__init__(col_parallel_names, row_parallel_names)

    @classmethod
    def load_hf(cls, ckpt_files: List[str]) -> dict:
        state_dict = Checkpoint.load_hf(ckpt_files)
        if "language_model.lm_head.weight" not in state_dict:  # for tie word embeddings
            print("`lm_head.weight` not found in checkpoint, copy `model.embed_tokens.weight` to replace it.")
            state_dict["language_model.lm_head.weight"] = state_dict["language_model.model.embed_tokens.weight"].clone()
        return state_dict


class CheckpointForMistral(Checkpoint):
    def __init__(self):
        col_parallel_names = [
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight", "lm_head.weight"
        ]
        row_parallel_names = [
            "o_proj.weight", "down_proj.weight", "embed_tokens.weight",
        ]
        super().__init__(col_parallel_names, row_parallel_names)


class CheckpointForInternLM3(Checkpoint):
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


def rename_llama3(ckpt_dir: str, hf_config_dir: str, save_dir: str = None, head_dim: int = None):
    checkpoint = CheckpointForLlama3()
    checkpoint.convert_consolidate_to_huggingface(ckpt_dir, hf_config_dir, save_dir, head_dim)


def show(ckpt_file):
    Checkpoint.show(ckpt_file)


def check_equal(ckpt_file1="../../models/llama3-hf/llama-3.2-1b-instruct-hf/original/consolidated.00.pth",
                ckpt_file2="../../models/llama3-hf/llama-3.2-1b-instruct-hf/model.safetensors"):
    state_dict1 = torch.load(ckpt_file1, map_location="cpu") if ckpt_file1.endswith(".pth") else Checkpoint.load_hf([ckpt_file1])
    state_dict2 = torch.load(ckpt_file2, map_location="cpu") if ckpt_file2.endswith(".pth") else Checkpoint.load_hf([ckpt_file2])
    replace_names = {
        "tok_embeddings.": "embed_tokens.",
        ".feed_forward.w1.": ".mlp.gate_proj.",
        ".feed_forward.w2.": ".mlp.down_proj.",
        ".feed_forward.w3.": ".mlp.up_proj.",
        ".attention_norm.": ".input_layernorm.",
        ".ffn_norm.": ".post_attention_layernorm.",
        ".attention.wq.": ".self_attn.q_proj.",
        ".attention.wk.": ".self_attn.k_proj.",
        ".attention.wv.": ".self_attn.v_proj.",
        ".attention.wo.": ".self_attn.o_proj.",
    }
    def replace(_name):
        if _name == "output.weight":
            _name = "lm_head.weight"
        else:
            for source, target in replace_names.items():
                _name = _name.replace(source, target)
            _name = "model." + _name
        return _name

    for name, param in state_dict1.items():
        if name in state_dict2:
            hf_name = name
        else:
            if name == "output.weight":
                continue
            hf_name = replace(name)
        i = random.randint(0, param.shape[0] - 1)
        if hf_name not in state_dict2:
            continue
        print(name, hf_name, param[i].tolist() == state_dict2[hf_name][i].tolist())
        # assert param[i].tolist() == state_dict2[hf_name][i].tolist()


if __name__ == '__main__':
    fire.Fire()

