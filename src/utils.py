import collections
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Tuple, List, Union, Callable

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    get_model_parallel_world_size,
    get_model_parallel_rank,
    get_model_parallel_src_rank,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_model_parallel_group, get_data_parallel_group,
)
from torch.distributed import init_process_group
from tqdm import trange


def get_torch_dtype(dtype: str):
    if dtype == "float32" or dtype == "fp32":
        return torch.float32
    elif dtype == 'float16' or dtype == "fp16":
        return torch.float16
    elif dtype == "bfloat16" or dtype == "bf16":
        return torch.bfloat16
    elif dtype == "int8":
        return torch.int8
    else:
        raise ValueError(dtype)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [b, s, n_heads, head_dim / 2]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# Copied from Huggingface
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from Huggingface
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from Huggingface
def compute_position_ids(start_pos: int, seq_length: int):
    position_ids = torch.arange(
        start_pos, seq_length + start_pos, dtype=torch.long
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    return position_ids


def json_dump(obj, f, indent=None, ensure_ascii=False):
    if str(f).endswith(".json"):
        with open(f, 'w', encoding='utf-8') as writer:
            writer.write(json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii))
    elif str(f).endswith(".jsonl"):
        with open(f, 'w', encoding='utf-8') as writer:
            assert type(obj) is list
            for data in obj:
                writer.write(json.dumps(data, ensure_ascii=ensure_ascii) + '\n')
    else:
        raise ValueError(f"Unexpected file type: {str(f)}")


def json_load(f):
    """Load a .json file into a dictionary."""
    if str(f).endswith(".json"):
        with open(f, 'r', encoding='utf-8') as reader:
            datalist = json.load(reader)
    elif str(f).endswith(".jsonl"):
        datalist = []
        with open(f, 'r', encoding='utf-8') as reader:
            for line in reader:
                datalist.append(json.loads(line))
    else:
        raise ValueError(f"Unexpected file type: {str(f)}")
    return datalist


def pickle_load(f):
    with open(f, "rb") as r:
        objects = pickle.load(r)
    return objects


def pickle_dump(obj, f):
    with open(f, "wb") as f:
        pickle.dump(obj, f)
    return f


def get_data_parallel_src_rank() -> int:
    """Calculate the global rank corresponding to a local rank zero
    in the data parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_data_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


ParallelInfos = collections.namedtuple("ParallelInfos", [
    "global_rank",
    "local_rank",
    "world_size",
    "model_parallel_world_size",
    "model_parallel_rank",
    "model_parallel_src_rank",
    "data_parallel_world_size",
    "data_parallel_rank",
    "data_parallel_src_rank"
])


def setup_model_parallel(
        model_parallel_size: int = None, seed: int = None
) -> ParallelInfos:

    global_rank: int = int(os.environ.get("RANK"))
    local_rank: int = int(os.environ.get("LOCAL_RANK"))
    world_size: int = int(os.environ.get("WORLD_SIZE"))

    init_process_group("nccl")
    initialize_model_parallel(model_parallel_size or world_size)

    model_parallel_world_size: int = get_model_parallel_world_size()
    model_parallel_rank: int = get_model_parallel_rank()
    model_parallel_src_rank: int = get_model_parallel_src_rank()
    data_parallel_world_size: int = get_data_parallel_world_size()
    data_parallel_rank: int = get_data_parallel_rank()
    data_parallel_src_rank: int = get_data_parallel_src_rank()

    if global_rank != model_parallel_src_rank:
        sys.stdout = open(os.devnull, "w")

    torch.cuda.set_device(local_rank)
    # seed must be the same in all processes
    set_seed(seed or 1)

    return ParallelInfos(
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        model_parallel_world_size=model_parallel_world_size,
        model_parallel_rank=model_parallel_rank,
        model_parallel_src_rank=model_parallel_src_rank,
        data_parallel_world_size=data_parallel_world_size,
        data_parallel_rank=data_parallel_rank,
        data_parallel_src_rank=data_parallel_src_rank
    )


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def cross_entropy(logits, labels, weights=None, keepdim=False):
    """
    Compute Cross-Entropy Loss..
    :param logits: [batch_size, ..., num_classes] tensor.
    :param labels: [batch_size, ...] tensor. LongTensor.
    Same shape with 0th - (last - 1)th of logits.
    :param weights: [batch_size, ...] tensor, where `1` donates validate and
     `0` donates invalidate. Same shape with 0th - (last - 1)th of logits.
    :param keepdim: bool, whether not to perform reduce sum of the final result.
    :return: The mean of all examples' loss.
    """
    bzs = logits.shape[0]
    logits = logits.float()
    labels = labels.to(logits.device)
    if weights is None:
        weights = torch.ones_like(labels)
    weights = weights.float().to(logits.device)
    weights = torch.reshape(weights, [bzs, -1])
    num_classes = int(logits.size()[-1])
    logits = torch.reshape(logits, shape=[bzs, -1, num_classes])
    log_probs = F.log_softmax(logits, dim=-1)
    labels = torch.reshape(labels, [bzs, -1]).long()
    labels = F.one_hot(labels, num_classes=num_classes)
    loss = - torch.sum(log_probs * labels, dim=[-1])  # [b, s]
    if not keepdim:
        nrt = torch.sum(weights * loss, dim=-1)
        dnm = torch.sum(weights, dim=-1) + 1e-8
        loss = torch.mean(nrt / dnm, dim=0)
    return loss


def set_barrier():
    """ make sure that all other processes cannot continue until reach this op. """
    torch.distributed.barrier()


def set_data_parallel_barrier():
    """ make sure that all other processes in data parallel group cannot continue until reach this op. """
    torch.distributed.barrier(get_data_parallel_group())


def set_model_parallel_barrier():
    """ make sure that all other processes in model parallel group cannot continue until reach this op. """
    torch.distributed.barrier(get_model_parallel_group())


def extract_logits(logits, p=0.8, max_n=10, min_n=5):
    """
    For extracting teacher distribution to distill.
    We don't need to record the whole vocabulary's probabilities which is space-consuming.
    Instead, we only retain those indices summed up to `p`.
    :param: logits, two dim tensor [s, v].
    :return: List[dict]
    """
    assert min_n <= max_n
    assert len(logits.shape) == 2
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=max_n)
    results = []
    for vas, ins, lgs in zip(values, indices, logits):
        end_index = 0
        accum = 0
        for v in vas:
            accum += v
            end_index += 1
            if accum >= p:
                break
        if end_index < min_n:
            end_index = min_n
        result = {}
        for i in range(end_index):
            result[int(ins[i])] = round(float(lgs[ins[i]]), 6)
        results.append(result)
    return results


def reconstruct_logits_from_dict(logits_dict: dict, vocab_size: int = 32000) -> torch.Tensor:
    # logits = torch.zeros(size=(vocab_size,))
    logits = torch.full(size=(vocab_size,), fill_value=-1e4, dtype=torch.float16)
    for index, lgt in logits_dict.items():
        index, lgt = int(index), float(lgt)
        logits[index] = lgt
    # normalize
    # logits = logits / torch.sum(logits)
    return logits  # [v]


def reconstruct_logits_from_dicts(
        logits_dicts: List[dict], vocab_size: int = 32000
) -> torch.Tensor:
    """ Reconstruct logits with return shape [seq_length, vocab_size] """
    seq_len = len(logits_dicts)
    logits = torch.zeros(size=(seq_len, vocab_size))
    for i, logits_dict in enumerate(logits_dicts):
        logits[i, :] = reconstruct_logits_from_dict(logits_dict, vocab_size)
    return logits


def powmax(tensor, exponent=1, dim=-1, eps=7e-5):
    """ Similar to softmax, perform power max on vectors along one specific dimension. """
    numerator = torch.pow(tensor, exponent=exponent)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / (denominator + eps)


def masked_mean(x, mask=None, dim: int = -1, keepdim: bool = False, eps: float = 1e-12):
    if type(x) is torch.Tensor:
        if mask is None:
            mask = torch.full_like(x, fill_value=True)
        assert x.shape == mask.shape
        mask = mask.to(x.dtype)
        return torch.sum(
            x * mask, dim=dim, keepdim=keepdim
        ) / (torch.sum(mask, dim=dim, keepdim=keepdim) + eps)
    elif type(x) is np.ndarray:
        if mask is None:
            mask = np.full_like(x, fill_value=True)
        assert x.shape == mask.shape
        mask = mask.astype(x.dtype)
        return np.sum(
            x * mask, axis=dim, keepdims=keepdim
        ) / (np.sum(mask, axis=dim, keepdims=keepdim) + eps)
    else:
        raise TypeError


def masked_std(x, mask=None, dim: int = -1, keepdim: bool = False, eps: float = 1e-12):
    if type(x) is torch.Tensor:
        if mask is None:
            mask = torch.full_like(x, fill_value=True)
        mu = masked_mean(x, mask, dim, keepdim=True, eps=eps)
        x = (x - mu) ** 2
        mask = mask.to(x.dtype)
        std = torch.sqrt(
            torch.sum(
                x * mask, dim=dim, keepdim=keepdim
            ) / (torch.sum(mask, dim=dim, keepdim=keepdim) + eps)
        )
        return std
    elif type(x) is np.ndarray:
        if mask is None:
            mask = np.full_like(x, fill_value=True)
        mu = masked_mean(x, mask, dim, keepdim=True, eps=eps)
        x = (x - mu) ** 2
        mask = mask.astype(x.dtype)
        std = np.sqrt(
            np.sum(
                x * mask, axis=dim, keepdims=keepdim
            ) / (np.sum(mask, axis=dim, keepdims=keepdim) + eps)
        )
        return std
    else:
        raise TypeError


def logits_normalize(x: torch.Tensor, dim=-1):
    """ Avoid overflowing """
    return x - torch.max(x, dim=dim, keepdim=True)[0]


def load_safetensors(f: str) -> dict:
    state_dict = collections.OrderedDict()
    assert f.endswith(".safetensors")
    with safetensors.safe_open(f, "pt", device="cpu") as reader:
        for k in reader.keys():
            state_dict[k] = reader.get_tensor(k)
    return state_dict


def merge_lora_state_dict(state_dict: dict) -> dict:
    res_state_dict = {}
    with torch.no_grad():
        for name, param in state_dict.items():
            if 'lora' not in name:
                res_state_dict[name] = param.clone()
            elif 'lora_a_' in name:
                origin = name.replace('lora_a_', '')
                original_dtype = state_dict[origin].dtype
                w = state_dict[origin].float()
                wa = state_dict[name].float()
                wb = state_dict[name.replace('lora_a_', 'lora_b_')].float()
                res_state_dict[origin] = (w + wb @ wa).clone().to(original_dtype)
    return res_state_dict


def merge_lora_checkpoints(
        ckpt_dir,
        world_size=8,
        layers=32
):
    """ Merge lora checkpoint. """
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    state_dicts = []
    for checkpoint in checkpoints:
        state_dicts.append(torch.load(checkpoint, map_location='cpu'))

    with torch.no_grad():
        for i in range(layers):
            for j in range(world_size):
                for wx in ['wq', 'wk', 'wv', 'w1', 'w3', 'wo', 'w2']:
                    module = 'feed_forward' if wx in ['w1', 'w2', 'w3'] else 'attention'
                    w = state_dicts[j][f"layers.{i}.{module}.{wx}.weight"]
                    a = state_dicts[j].pop(f"layers.{i}.{module}.lora_a_{wx}.weight")
                    b = state_dicts[j].pop(f"layers.{i}.{module}.lora_b_{wx}.weight")
                    state_dicts[j][f"layers.{i}.{module}.{wx}.weight"] = (w + b @ a).clone().to(w.dtype)
        # output
        for j in range(world_size):
            w = state_dicts[j][f"output.weight"]
            a = state_dicts[j].pop(f"lora_a_output.weight")
            b = state_dicts[j].pop(f"lora_b_output.weight")
            state_dicts[j][f"output.weight"] = (w + b @ a).clone().to(w.dtype)

    saved_path = os.path.join(ckpt_dir, 'merged')
    print(f"Saving checkpoint to {saved_path} ......")
    for rank in trange(world_size):
        os.makedirs(saved_path, exist_ok=True)
        torch.save(state_dicts[rank], os.path.join(ckpt_dir, 'merged', f'consolidated.0{rank}.pth'))


def masked_normalize(x: Union[torch.Tensor, np.ndarray], masks: Union[torch.Tensor, np.ndarray] = None):
    """ mean to be 0, std to be 1 """
    if type(x) is torch.Tensor:
        bzs = x.shape[0]
        if masks is None:
            masks = torch.full_like(x, fill_value=True, dtype=torch.bool)
        for i in range(bzs):
            data = x[i][masks[i]]
            x[i][masks[i]] = (data - data.mean()) / (data.std() + 1e-8)
        return x
    elif type(x) is np.ndarray:
        bzs = x.shape[0]
        if masks is None:
            masks = np.full_like(x, fill_value=True, dtype=np.bool_)
        for i in range(bzs):
            data = x[i][masks[i]]
            x[i][masks[i]] = (data - data.mean()) / (data.std() + 1e-8)
        return x
    else:
        raise TypeError('Unknown type: ', type(x))


def clamp(x: torch.Tensor, disable: bool = False) -> torch.Tensor:
    """
    Clamp inf values to enable fp16 training.
    Will slow down speed, disable it when you don't need it.
    """
    if disable or not x.requires_grad:
        return x
    if x.dtype == torch.float16:
        clamp_value = torch.where(
            torch.isinf(x).any(),
            torch.finfo(x.dtype).max - 1000,
            torch.finfo(x.dtype).max
        ).item()
        x = torch.clamp(x, min=-clamp_value, max=clamp_value)
    return x


def apply_lora(x: torch.Tensor, lora_a: torch.nn.Module, lora_b: torch.nn.Module):
    return lora_b(lora_a(x.type_as(next(lora_a.parameters()))).type_as(next(lora_b.parameters()))).type_as(x)


def truncate(instruction_ids: list, output_ids: list, max_seq_len: int) -> (list, list):
    instruction_length = len(instruction_ids)
    output_length = len(output_ids)
    if instruction_length >= max_seq_len:
        print(f'WARNING: Length of instruction {instruction_length} '
              f'exceeds the max input length {max_seq_len}')
        instruction_ids = instruction_ids[:max_seq_len]
        instruction_length = len(instruction_ids)
    sequence_length = instruction_length + output_length
    if sequence_length > max_seq_len:
        exceed_length = sequence_length - max_seq_len
        output_ids = output_ids[:-exceed_length]
    return instruction_ids, output_ids


def convert_dataloader_data_to_list(data: dict) -> List[dict]:
    """
    :param data: data from `DataLoader`
    :return: List[dict]
    """
    batch_size = None
    for k, v in data.items():
        if batch_size is None:
            batch_size = len(v)
        else:
            assert batch_size == len(v)
        if isinstance(v, torch.Tensor):
            data[k] = v.tolist()
    result_dicts = []
    for i in range(batch_size):
        result_dict = {}
        for k, v in data.items():
            result_dict[k] = v[i]
        result_dicts.append(result_dict)
    return result_dicts


# ===============================================================


def jaccard(_set1: set, _set2: set) -> float:
    return len(_set1 & _set2) / len(_set1 | _set2)


def deduplicate_texts(iterable: list, threshold: float = 0.8, key: Callable = None) -> list:
    results = []
    if key is None:
        def key(x):
            return x
    for i in trange(len(iterable)):
        results.append(iterable[i])
        for j in range(i + 1, len(iterable)):
            if threshold == 1:
                sim = 1 if key(iterable[i]) == key(iterable[j]) else 0
            else:
                sim = jaccard(set(key(iterable[i]).split(' ')), set(key(iterable[j]).split(' ')))
            if sim >= threshold:
                results.pop(-1)
                break

    return results
