import os
from collections import OrderedDict
from pathlib import Path

import fire
import torch
from tqdm import tqdm


def is_parallel(name):
    return ('wq.wei' in name or 'q_proj.wei' in name) or \
           ('wk.wei' in name or 'k_proj.wei' in name) or \
           ('wv.wei' in name or 'v_proj.wei' in name) or \
           ('wo.wei' in name or 'o_proj.wei' in name) or \
           ('w1.wei' in name or 'gate_proj.wei' in name) or \
           ('w2.wei' in name or 'down_proj.wei' in name) or \
           ('w3.wei' in name or 'up_proj.wei' in name) or \
           ('tok_embeddings.wei' in name or 'embed_tokens.wei' in name) or \
           ('output.wei' in name or 'lm_head.wei' in name)


def is_col_parallel(name):
    return ('wq.wei' in name or 'q_proj.wei' in name) or \
           ('wk.wei' in name or 'k_proj.wei' in name) or \
           ('wv.wei' in name or 'v_proj.wei' in name) or \
           ('w1.wei' in name or 'gate_proj.wei' in name) or \
           ('w3.wei' in name or 'up_proj.wei' in name) or \
           ('output.weight' in name or 'lm_head.wei' in name)


def splitting(
        ckpt_file: str = 'config/13B/consolidated.00.pth',
        save_path: str = 'config/13B/4/',
        n: int = 4
):
    state_dict = torch.load(ckpt_file, map_location="cpu")
    new_state_dicts = [OrderedDict() for _ in range(n)]
    for name, param in state_dict.items():
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
    new_state_dicts = OrderedDict()

    for name in state_dict1.keys():
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
        c = torch.load(ckpt, map_location='cpu')
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


def show(
        ckpt_file: str = "config/orca-1-13b/pytorch_model.bin"
):
    state_dicts = torch.load(ckpt_file, map_location="cpu")
    for key, value in state_dicts.items():
        print(key, value.shape)


def remove_added_tokens(
        ckpt_file: str = "config/orca-2-13b/pytorch_model_renamed.bin"
):
    state_dict = torch.load(ckpt_file, map_location='cpu')
    state_dict['output.weight'] = state_dict['output.weight'][:-2, ...].clone()
    state_dict['tok_embeddings.weight'] = state_dict['tok_embeddings.weight'][:-2, ...].clone()

    torch.save(state_dict, ckpt_file.replace(".bin", "_removed.bin"))
    for key, value in state_dict.items():
        print(key, value.shape)


if __name__ == '__main__':
    fire.Fire(remove_added_tokens)
