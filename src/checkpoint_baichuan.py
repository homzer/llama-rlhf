import os
from collections import OrderedDict
from pathlib import Path
from typing import Union

import safetensors
import torch

from src.checkpoint import splitting__


def process_w_pack(state_dict: OrderedDict) -> OrderedDict:
    new_state_dict = OrderedDict()
    for name, param in state_dict.items():
        if 'W_pack' in name:
            h = param.shape[1]
            assert param.shape[0] / h == 3
            new_state_dict[name.replace("W_pack", "q_proj")] = param[: h].clone()
            new_state_dict[name.replace("W_pack", "k_proj")] = param[h: 2 * h].clone()
            new_state_dict[name.replace("W_pack", "v_proj")] = param[2 * h:].clone()
        else:
            new_state_dict[name] = param
    return new_state_dict


# Copied from src.checkpoint.splitting
def splitting(
        ckpt_file: Union[str, list] = 'config/13B/consolidated.00.pth',
        save_path: str = 'config/13B/4/',
        n: int = 4
):
    assert isinstance(ckpt_file, str) or isinstance(ckpt_file, list)
    if isinstance(ckpt_file, str):
        state_dict = torch.load(ckpt_file, map_location="cpu")
    else:
        state_dict = OrderedDict()
        for f in ckpt_file:
            f = str(f)
            if f.endswith(".safetensors"):
                with safetensors.safe_open(f, "pt", device="cpu") as reader:
                    for k in reader.keys():
                        state_dict[k] = reader.get_tensor(k)
            else:
                reader = torch.load(f, map_location="cpu")
                for k in reader.keys():
                    state_dict[k] = reader[k]
    state_dict = process_w_pack(state_dict)
    new_state_dicts = splitting__(state_dict, n)
    os.makedirs(save_path, exist_ok=True)
    for i in range(n):
        torch.save(new_state_dicts[i], os.path.join(save_path, f'consolidated.0{i}.pth'))


# Copied from src.checkpoint.auto_split_huggingface_checkpoints
def auto_split_huggingface_checkpoints(ckpt_dir: str, world_size: int, local_rank: int, verbose: bool = True) -> str:
    pl_ckpt_dir = os.path.join(ckpt_dir, str(world_size))
    if local_rank == 0 and not os.path.exists(pl_ckpt_dir):
        if verbose:
            print(f'Parallel checkpoint dose not exist. Splitting into {pl_ckpt_dir} ...')
        if os.path.exists(os.path.join(ckpt_dir, "pytorch_model.bin")):
            split_file = os.path.join(ckpt_dir, "pytorch_model.bin")
        else:
            split_file = sorted(Path(ckpt_dir).glob("*.safetensors"))
            if len(split_file) == 0:
                split_file = sorted(Path(ckpt_dir).glob("pytorch_model*.bin"))
                if len(split_file) == 0:
                    raise FileNotFoundError("Can not find any checkpoint file")
        splitting(split_file, pl_ckpt_dir, n=world_size)
        if verbose:
            print('Done!')
    return pl_ckpt_dir
