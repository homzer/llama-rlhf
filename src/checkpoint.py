import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Union, Dict

import fire
import safetensors
import safetensors.torch as safe_torch
import torch
from tqdm import tqdm


def load_safetensors(f: str) -> OrderedDict:
    state_dict = OrderedDict()
    assert f.endswith(".safetensors")
    with safetensors.safe_open(f, "pt", device="cpu") as reader:
        for k in reader.keys():
            state_dict[k] = reader.get_tensor(k)
    return state_dict


def save_safetensors(state_dict: Dict[str, torch.Tensor], f: str):
    safe_torch.save_file(state_dict, f)


def is_parallel(name):
    return ('wq.wei' in name or 'q_proj.wei' in name or 'wq.bias' in name or 'q_proj.bias' in name) or \
           ('wk.wei' in name or 'k_proj.wei' in name or 'wk.bias' in name or 'k_proj.bias' in name) or \
           ('wv.wei' in name or 'v_proj.wei' in name or 'wv.bias' in name or 'v_proj.bias' in name) or \
           ('wo.wei' in name or 'o_proj.wei' in name) or \
           ('w1.wei' in name or 'gate_proj.wei' in name or 'w1.bias' in name or 'gate_proj.bias' in name) or \
           ('w2.wei' in name or 'down_proj.wei' in name) or \
           ('w3.wei' in name or 'up_proj.wei' in name or 'w3.bias' in name or 'up_proj.bias' in name) or \
           ('tok_embeddings.wei' in name or 'embed_tokens.wei' in name) or \
           ('output.wei' in name or 'lm_head.wei' in name or 'output.bias' in name or 'lm_head.bias' in name)


def is_col_parallel(name):
    return ('wq' in name or 'q_proj' in name) or \
           ('wk' in name or 'k_proj' in name) or \
           ('wv' in name or 'v_proj' in name) or \
           ('w1' in name or 'gate_proj' in name) or \
           ('w3' in name or 'up_proj' in name) or \
           ('output' in name or 'lm_head' in name)


def get_layer_id(name):
    matches = re.findall(r'layers\.(\d+)\.', name)
    return matches[0]


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

    new_state_dicts = splitting__(state_dict, n)
    os.makedirs(save_path, exist_ok=True)
    for i in range(n):
        torch.save(new_state_dicts[i], os.path.join(save_path, f'consolidated.0{i}.pth'))


def auto_split_2_to_8(
        ckpt_dir: str = "config/13B/2/",
        save_dir: str = 'config/13B/8/'
):
    state_dicts = []
    for i in range(2):
        state_dict = torch.load(os.path.join(ckpt_dir, f"consolidated.0{i}.pth"), map_location="cpu")
        state_dicts.extend(splitting__(state_dict, 4))
    os.makedirs(save_dir, exist_ok=True)
    for i in range(8):
        torch.save(state_dicts[i], os.path.join(save_dir, f'consolidated.0{i}.pth'))


def auto_split_4_to_8(
        ckpt_dir: str = 'config/13B/4/',
        save_dir: str = 'config/13B/8/'
):
    state_dicts = []
    for i in range(4):
        state_dict = torch.load(os.path.join(ckpt_dir, f"consolidated.0{i}.pth"), map_location="cpu")
        state_dicts.extend(splitting__(state_dict, 2))
    os.makedirs(save_dir, exist_ok=True)
    for i in range(8):
        torch.save(state_dicts[i], os.path.join(save_dir, f'consolidated.0{i}.pth'))


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


def auto_merge_8_to_4(
        ckpt_dir: str = 'config/13B/8/',
        save_dir: str = 'config/13B/4/'
):
    state_dicts = []
    for i in [0, 2, 4, 6]:
        state_dict1 = torch.load(os.path.join(ckpt_dir, f"consolidated.0{i}.pth"), map_location="cpu")
        state_dict2 = torch.load(os.path.join(ckpt_dir, f"consolidated.0{i+1}.pth"), map_location="cpu")
        state_dicts.append(merging__(state_dict1, state_dict2))
    os.makedirs(save_dir, exist_ok=True)
    assert len(state_dicts) == 4
    for i in range(4):
        torch.save(state_dicts[i], os.path.join(save_dir, f'consolidated.0{i}.pth'))


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


def merge_hf_checkpoints(
        folder_path: str = "config/orca-1-13b/"
):
    checkpoints = sorted(Path(folder_path).glob("pytorch_model*.bin"))
    results = None
    for ckpt in checkpoints:
        c = torch.load(str(ckpt), map_location='cpu')
        if results is None:
            results = c
            continue
        for name, param in tqdm(c.items()):
            results[name] = param.clone()
    torch.save(results, os.path.join(folder_path, "pytorch_model.bin"))
    for key, value in results.items():
        print(key, value.shape)


def rename_hf_ckpt_to_llama(
        ckpt_file: str = "config/orca-1-13b/pytorch_model.bin"
):
    state_dicts = torch.load(ckpt_file, map_location="cpu")
    new_state_dicts = OrderedDict()
    for name, param in state_dicts.items():
        name = str(name).replace("model.layers.", "layers.")
        name = name.replace("model.norm.", "norm.")
        name = name.replace("lm_head.weight", "output.weight")
        name = name.replace("model.embed_tokens.", "tok_embeddings.")
        name = name.replace(".self_attn.q_proj.", ".attention.wq.")
        name = name.replace(".self_attn.k_proj.", ".attention.wk.")
        name = name.replace(".self_attn.v_proj.", ".attention.wv.")
        name = name.replace(".self_attn.o_proj.", ".attention.wo.")
        name = name.replace(".mlp.gate_proj.", ".feed_forward.w1.")
        name = name.replace(".mlp.down_proj.", ".feed_forward.w2.")
        name = name.replace(".mlp.up_proj.", ".feed_forward.w3.")
        name = name.replace(".input_layernorm.", ".attention_norm.")
        name = name.replace(".self_attn.rotary_emb.", ".attention.rotary_emb.")
        name = name.replace(".post_attention_layernorm.", ".ffn_norm.")
        new_state_dicts[name] = param.clone()

    torch.save(new_state_dicts, ckpt_file.replace(".bin", "_renamed.bin"))
    for key, value in new_state_dicts.items():
        print(key, value.shape)


def auto_split_4_to_8_for_30b(
        ckpt_dir: str = "config/30B/4/",
        save_dir: str = "config/30B/8/"
):
    os.makedirs(save_dir, exist_ok=True)
    for mp in range(4):
        n = 2
        state_dict = torch.load(os.path.join(ckpt_dir, f"consolidated.0{mp}.pth"), map_location="cpu")
        new_state_dicts = [OrderedDict() for _ in range(n)]
        for name, param in state_dict.items():
            param = param.cpu()
            if is_parallel(name):
                params = []
                if is_col_parallel(name):
                    if 'wq' in name or 'wk' in name or 'wv' in name:
                        layer_id = int(get_layer_id(name))
                        if layer_id < 30:
                            params.append(param[:896])
                            params.append(param[896:])
                        else:
                            params.append(param[:768])
                            params.append(param[768:])
                    else:
                        dim0 = param.shape[0]
                        assert dim0 % n == 0
                        split = dim0 // n
                        for i in range(n):
                            params.append(param[i * split: (i + 1) * split])
                else:  # row parallel
                    if 'wo' in name:
                        layer_id = int(get_layer_id(name))
                        if layer_id < 30:
                            params.append(param[:, :896])
                            params.append(param[:, 896:])
                        else:
                            params.append(param[:, :768])
                            params.append(param[:, 768:])
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
        for i in range(n):
            torch.save(new_state_dicts[i], os.path.join(save_dir, f'consolidated.0{(mp*2)+i}.pth'))


def show(
        ckpt_file: str = "config/orca-1-13b/pytorch_model.bin"
):
    state_dicts = torch.load(ckpt_file, map_location="cpu")
    for key, value in state_dicts.items():
        print(key, value.shape)


def merge_lora(
        ckpt_dir: str = 'config/13B/8/',
        save_dir: str = 'config/13B/8/merge'
):
    os.makedirs(save_dir, exist_ok=True)
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    for i, ckpt_file in enumerate(checkpoints):
        result_dict = {}
        state_dict = torch.load(str(ckpt_file), map_location="cpu")
        for name, param in state_dict.items():
            if 'lora' not in name:
                result_dict[name] = param.clone()
            elif 'lora_a_' in name:
                origin = name.replace('lora_a_', '')
                w = state_dict[origin]
                wa = state_dict[name]
                wb = state_dict[name.replace('lora_a_', 'lora_b_')]
                result_dict[origin] = (w + wb @ wa).clone().to(w.dtype)
        torch.save(result_dict, os.path.join(save_dir, f'consolidated.0{i}.pth'))


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


if __name__ == '__main__':
    fire.Fire()
