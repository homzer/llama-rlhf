import collections
import inspect
import json
import os
import pickle
import random
from typing import Tuple, List, Callable

import numpy as np
import safetensors
import torch
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


def apply_rotary_emb_(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


# Copied from Huggingface
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from Huggingface
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])  # [1, 1, seq_len, head_dim]
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_(x, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])  # [1, 1, seq_len, head_dim]
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)  # [1, 1, seq_len, head_dim]
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)  # [1, 1, seq_len, head_dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


# Copied from Huggingface
def compute_position_ids(start_pos: int, seq_length: int):
    position_ids = torch.arange(
        start_pos, seq_length + start_pos, dtype=torch.long
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    return position_ids  # [1, seq_len]


def json_dump(obj, file, indent=None, ensure_ascii=False):
    if str(file).endswith(".json"):
        with open(file, 'w', encoding='utf-8') as writer:
            writer.write(json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii))
    elif str(file).endswith(".jsonl"):
        with open(file, 'w', encoding='utf-8') as writer:
            assert type(obj) is list
            for data in obj:
                writer.write(json.dumps(data, ensure_ascii=ensure_ascii) + '\n')
    else:
        raise ValueError(f"Unexpected file type: {str(file)}")


def json_load(file):
    """Load a .json file into a dictionary."""
    if str(file).endswith(".json"):
        with open(file, 'r', encoding='utf-8') as reader:
            datalist = json.load(reader)
    elif str(file).endswith(".jsonl"):
        datalist = []
        with open(file, 'r', encoding='utf-8') as reader:
            for line in reader:
                datalist.append(json.loads(line))
    else:
        raise ValueError(f"Unexpected file type: {str(file)}")
    return datalist


def pickle_load(f):
    with open(f, "rb") as r:
        objects = pickle.load(r)
    return objects


def pickle_dump(obj, f):
    with open(f, "wb") as f:
        pickle.dump(obj, f)
    return f


def sample_top_p(probs: torch.Tensor, p: float = 1.0, num_samples: int = 1):
    """
    perform top-p sampling on the last dim
    :param probs: any tensor with shape [..., vocab_size], probability distribution
    :param p: top probability
    :param num_samples: number of sampled tokens, default to 1.
    :return: next_tokens with shape [..., num_samples]
    """
    origin_shape = probs.shape[:-1]
    probs = torch.reshape(probs, shape=[-1, probs.shape[-1]])
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    try:
        next_tokens = torch.multinomial(probs_sort, num_samples=num_samples)
    except RuntimeError:
        print(probs_sort)
        if int(os.environ.get("RANK")) == 0:
            torch.save(probs_sort.cpu(), "./probs_sort_debug.bin")
        exit(0)
    next_tokens = torch.gather(probs_idx, -1, next_tokens)
    next_tokens = torch.reshape(next_tokens, shape=[*origin_shape, num_samples])
    return next_tokens


def powmax(tensor, exponent=1, dim=-1, eps=1e-12):
    """ Similar to softmax, perform power max on vectors along one specific dimension. """
    numerator = torch.pow(tensor, exponent=exponent)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return (numerator.float() / torch.clamp(denominator.float(), min=eps)).type_as(tensor)


def masked_sum(x, mask=None, dim: int = None, keepdim: bool = False):
    if type(x) is torch.Tensor:
        if mask is None:
            mask = torch.full_like(x, fill_value=True)
        assert x.shape == mask.shape
        return torch.sum(x * mask, dim=dim, keepdim=keepdim)
    elif type(x) is np.ndarray:
        if mask is None:
            mask = np.full_like(x, fill_value=True)
        assert x.shape == mask.shape
        return np.sum(x * mask, axis=dim, keepdims=keepdim)
    else:
        raise TypeError(type(x))


def masked_mean(x, mask=None, dim: int = None, keepdim: bool = False):
    if type(x) is torch.Tensor:
        if mask is None:
            mask = torch.full_like(x, fill_value=True)
        assert x.shape == mask.shape
        mask_sum = torch.sum(mask, dim=dim, keepdim=keepdim)
        mask_sum = torch.where(mask_sum == 0, 0.0001, mask_sum)
        mask_sum = mask_sum.to(x.dtype)
        return torch.sum(x * mask, dim=dim, keepdim=keepdim) / mask_sum
    elif type(x) is np.ndarray:
        if mask is None:
            mask = np.full_like(x, fill_value=True)
        assert x.shape == mask.shape
        mask_sum = np.sum(mask, axis=dim, keepdims=keepdim)
        mask_sum = np.where(mask_sum == 0, 0.0001, mask_sum)
        mask_sum = mask_sum.astype(x.dtype)
        return np.sum(x * mask, axis=dim, keepdims=keepdim) / mask_sum
    else:
        raise TypeError(type(x))


def masked_std(x, mask=None, dim: int = -1, keepdim: bool = False, eps: float = 1e-12):
    if type(x) is torch.Tensor:
        if mask is None:
            mask = torch.full_like(x, fill_value=True)
        mu = masked_mean(x, mask, dim, keepdim=True)
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
        mu = masked_mean(x, mask, dim, keepdim=True)
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


def normalize(x, dim: int = None):
    if type(x) is torch.Tensor:
        return (x - torch.mean(x, dim=dim, keepdim=True)) / torch.std(x, dim=dim, keepdim=True)
    elif type(x) is np.ndarray:
        return (x - np.mean(x, axis=dim, keepdims=True)) / np.std(x, axis=dim, keepdims=True)
    else:
        raise TypeError


def logits_normalize(x: torch.Tensor, dim=-1):
    """ Avoid overflow and underflow """
    return x - torch.max(x, dim=dim, keepdim=True)[0].detach()


class _Log1mSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        a = torch.softmax(input_, dim=-1)
        a = torch.clamp(a, max=1 - 2e-3)  # for bfloat16
        ctx.save_for_backward(a)
        return torch.log(1 - a)

    @staticmethod
    def backward(ctx, *grad_outputs):
        a, = ctx.saved_tensors
        grad_output = grad_outputs[0]
        grad_input = (a / (a - 1) * grad_output).sum(dim=-1, keepdim=True) + grad_output / (1 - a)
        grad_input = - a * grad_input
        return grad_input


def log1m_softmax(logits: torch.Tensor):
    """
    Stably compute `log(1 - softmax(logits))` along the last dim.
    Args:
        logits (torch.Tensor): Input tensor containing logits.
    Returns:
        torch.Tensor: A tensor containing `log(1 - softmax(logits))` values.
    """
    return _Log1mSoftmax.apply(logits)


@torch.no_grad()
def create_target_distribution(
        logits: torch.Tensor,
        actions: torch.Tensor,
        rho: float = 1.0,
        min_rho_prob: float = 0.80,
        max_rho_prob: float = 0.99
):
    """
    Construct a target distribution based on `old_action_logprobs` by adjusting
    the probability values at the positions indexed by `actions`.
    :param logits: [..., vocab_size] policy logits tensor.
    :param actions: [...,] action ids indicating positions to modify.
    :param rho: should be greater than 0. It is the ratio between the target probability and
    the original probability of a sampled action. A value closer to 1 results in higher similarity.
    :param min_rho_prob: is used when `rho` < 1 to set a lower bound for the probability.
    When the calculated `rho_probs` is less than `min_rho_prob`, we use the original probability instead of `rho_probs`.
    :param max_rho_prob: is used when `rho` > 1 to set a upper bound for the probability.
    When the calculated `rho_probs` is more than `max_rho_prob`, we use the `max_rho_prob` instead of `rho_probs`.
    :return: log target distribution with adjusted probabilities.
    """
    assert rho > 0
    actions = actions.unsqueeze(-1)
    action_logprobs = torch.gather(torch.log_softmax(logits, dim=-1), dim=-1, index=actions).squeeze(-1)
    a = torch.logsumexp(
        torch.scatter_add(
            logits,
            dim=-1,
            index=actions,
            src=torch.full_like(actions, fill_value=float("-inf"), dtype=logits.dtype)
        ),
        dim=-1,
        keepdim=True
    )
    b = torch.gather(logits, dim=-1, index=actions)
    rho_probs = rho * action_logprobs.float().exp()
    if min_rho_prob is not None and rho < 1:  # only consider the case when rho < 1
        assert 0 <= min_rho_prob <= 1
        rho_probs = torch.where(rho_probs < min_rho_prob, action_logprobs.float().exp(), rho_probs)
    if max_rho_prob is not None and rho > 1:  # only consider the case when rho > 1
        assert 0 <= max_rho_prob <= 1
        rho_probs = torch.where(rho_probs > max_rho_prob, max_rho_prob, rho_probs)
    c = torch.where(
        1 - rho_probs > 0, torch.log(rho_probs / (1 - rho_probs)), 1000.
    ).unsqueeze(-1).to(logits.dtype)
    logits = torch.scatter_add(logits, dim=-1, index=actions, src=(a - b + c))
    return torch.log_softmax(logits, dim=-1)


@torch.no_grad()
def create_lco_log_target(old_logits: torch.Tensor, advantages: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Create LCO log targets for KL loss.
    :param old_logits: [..., vocab_size]
    :param advantages: [..., vocab_size], same shape with `old_logits`
    :param beta: control original KL penalty, a larger value for greater penalty
    :return: tensor with shape [..., vocab_size]
    """
    assert beta > 0
    log_target = torch.log_softmax(old_logits, dim=-1) + advantages / beta
    log_target = log_target - torch.logsumexp(log_target, dim=-1, keepdim=True)  # [..., 1]
    return log_target


@torch.no_grad()
def estimate_vocab_adv(
        old_logits: torch.Tensor, advantages: torch.Tensor, advantage_indices: torch.Tensor
) -> torch.Tensor:
    """
    Estimate the missing values of vocab-size advantages according minimizing derivative principle.
    :param old_logits: [..., vocab_size]
    :param advantages: [..., known_size]
    :param advantage_indices: [..., known_size]
    :return: estimated advantages with shape of [..., vocab_size]
    """
    assert len(old_logits.shape) == len(advantages.shape)
    assert advantage_indices.shape == advantages.shape
    old_probs = torch.gather(torch.softmax(old_logits, dim=-1), dim=-1, index=advantage_indices)
    advantages = advantages * (1 - old_probs)  # smoothing
    estimated_adv = - torch.sum(old_probs * advantages, dim=-1) / torch.clamp(1 - old_probs.sum(-1), min=0.1)
    estimated_adv = torch.repeat_interleave(estimated_adv.unsqueeze(-1), repeats=old_logits.shape[-1], dim=-1)
    estimated_adv[torch.arange(estimated_adv.shape[0])[:, None], advantage_indices] = advantages
    return estimated_adv


def load_safetensors(f: str) -> dict:
    state_dict = collections.OrderedDict()
    assert f.endswith(".safetensors")
    with safetensors.safe_open(f, "pt", device="cpu") as reader:
        for k in reader.keys():
            state_dict[k] = reader.get_tensor(k)
    return state_dict


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


def print_current_func_args():
    frame = inspect.currentframe().f_back
    args, _, _, values = inspect.getargvalues(frame)
    param_str = '\n'.join(['%30s = %s' % (arg, values[arg]) for arg in sorted(args)])
    print('\n%30s   %s\n%s\n%s\n' % ('ATTRIBUTE', 'VALUE', '_' * 60, param_str))


def can_convert_to_tensor(x: np.ndarray) -> bool:
    supported_dtypes = [
        np.float32, np.float64, np.float16,
        np.int64, np.int32, np.int16, np.int8, np.uint8,
        np.bool_
    ]
    return x.dtype in supported_dtypes


def reshape_to_matrix(x: np.ndarray | torch.Tensor):
    return x.reshape(-1, x.shape[-1])


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
