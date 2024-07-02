import os
from collections import OrderedDict
from pathlib import Path
from typing import Union

import fire
import safetensors
import torch


def load_safetensors(f: str) -> OrderedDict:
    state_dict = OrderedDict()
    assert f.endswith(".safetensors")
    with safetensors.safe_open(f, "pt", device="cpu") as reader:
        for k in reader.keys():
            state_dict[k] = reader.get_tensor(k)
    return state_dict


def is_parallel(name):
    return ('wq.wei' in name or 'q_proj.wei' in name or 'wq.bias' in name or 'q_proj.bias' in name) or \
        ('wk.wei' in name or 'k_proj.wei' in name or 'wk.bias' in name or 'k_proj.bias' in name) or \
        ('wv.wei' in name or 'v_proj.wei' in name or 'wv.bias' in name or 'v_proj.bias' in name) or \
        ('wo.wei' in name or 'o_proj.wei' in name) or \
        ('w1.wei' in name or 'gate_proj.wei' in name or 'w1.bias' in name or 'gate_proj.bias' in name) or \
        ('w2.wei' in name or 'down_proj.wei' in name) or \
        ('w3.wei' in name or 'up_proj.wei' in name or 'w3.bias' in name or 'up_proj.bias' in name) or \
        ('tok_embeddings.wei' in name or 'embed_tokens.wei' in name)


def is_col_parallel(name):
    return ('wq' in name or 'q_proj' in name) or \
        ('wk' in name or 'k_proj' in name) or \
        ('wv' in name or 'v_proj' in name) or \
        ('w1' in name or 'gate_proj' in name) or \
        ('w3' in name or 'up_proj' in name)


def splitting__(state_dict, n) -> list:
    new_state_dicts = [OrderedDict() for _ in range(n)]
    for name, param in state_dict.items():
        assert 'lora' not in name, 'can not split a lora checkpoint, merge it first'
        param = param.cpu()
        if is_parallel(name):
            params = []
            if is_col_parallel(name):
                dim0 = param.shape[0]
                assert dim0 % n == 0
                split = dim0 // n
                for i in range(n):
                    params.append(param[i * split: (i + 1) * split])
            else:
                dim1 = param.shape[1]
                assert dim1 % n == 0
                split = dim1 // n
                for i in range(n):
                    params.append(param[:, i * split: (i + 1) * split])
            for i in range(n):
                new_state_dicts[i][name] = params[i].clone()
        else:
            for i in range(n):
                new_state_dicts[i][name] = param.clone()
    return new_state_dicts


def merging__(state_dict1, state_dict2) -> dict:
    new_state_dicts = OrderedDict()

    for name in state_dict1.keys():
        assert 'lora' not in name, 'can not split a lora checkpoint, merge it first'
        if is_parallel(name):
            param1 = state_dict1[name]
            param2 = state_dict2[name]
            assert len(param1.shape) == 2
            if is_col_parallel(name):
                param = torch.cat([param1, param2], dim=0)
            else:
                param = torch.cat([param1, param2], dim=1)
            new_state_dicts[name] = param.clone()
        else:
            new_state_dicts[name] = state_dict1[name].clone()

    return new_state_dicts


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


def merging(
        ckpt_file1: str = 'config/13B/consolidated.00.pth',
        ckpt_file2: str = 'config/13B/consolidated.01.pth',
        save_file: str = 'consolidated.pth'
):
    state_dict1 = torch.load(ckpt_file1, map_location='cpu')
    state_dict2 = torch.load(ckpt_file2, map_location='cpu')

    new_state_dicts = merging__(state_dict1, state_dict2)

    for name, param in new_state_dicts.items():
        print(name, param.shape)

    save_dir = '/'.join(save_file.split('/')[:-1])
    os.makedirs(save_dir, exist_ok=True)
    torch.save(new_state_dicts, save_file)


def auto_merge_8_to_1(
        ckpt_dir: str,
        save_file: str = "pytorch_model.bin",
):
    os.makedirs(os.path.split(save_file)[0], exist_ok=True)
    state_dict = merging__(
        merging__(
            merging__(
                torch.load(os.path.join(ckpt_dir, "consolidated.00.pth"), map_location="cpu"),
                torch.load(os.path.join(ckpt_dir, "consolidated.01.pth"), map_location="cpu")
            ),
            merging__(
                torch.load(os.path.join(ckpt_dir, "consolidated.02.pth"), map_location="cpu"),
                torch.load(os.path.join(ckpt_dir, "consolidated.03.pth"), map_location="cpu")
            )
        ),
        merging__(
            merging__(
                torch.load(os.path.join(ckpt_dir, "consolidated.04.pth"), map_location="cpu"),
                torch.load(os.path.join(ckpt_dir, "consolidated.05.pth"), map_location="cpu")
            ),
            merging__(
                torch.load(os.path.join(ckpt_dir, "consolidated.06.pth"), map_location="cpu"),
                torch.load(os.path.join(ckpt_dir, "consolidated.07.pth"), map_location="cpu")
            )
        )
    )
    torch.save(state_dict, save_file)


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


def show(
        ckpt_file: str = "config/orca-1-13b/pytorch_model.bin"
):
    if ckpt_file.endswith(".safetensors"):
        state_dicts = load_safetensors(ckpt_file)
    else:
        state_dicts = torch.load(ckpt_file, map_location="cpu")
    for key, value in state_dicts.items():
        print(key, value.shape)


if __name__ == '__main__':
    fire.Fire()
