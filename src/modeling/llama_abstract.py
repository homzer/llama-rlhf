import math
from typing import Optional

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling.modeling import ParallelModule, ParallelModelForCausalLM, CausalLMOutputs
from src.modeling.modeling_args import LlamaArgs, LoraLlamaArgs
from src.utils import apply_rotary_emb, precompute_freqs_cis, set_barrier, logits_normalize


class AbstractAttention(nn.Module):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        self.args = args
        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = None
        self.wk = None
        self.wv = None
        self.wo = None

        self.cache_k = None
        self.cache_v = None

    def forward(self,
                x: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
                use_cache=False):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if use_cache:
            if self.cache_k is None:
                self.cache_k = torch.zeros(
                    (bsz, self.args.max_seq_len, self.n_local_heads, self.head_dim)
                ).cuda()
            if self.cache_v is None:
                self.cache_v = torch.zeros(
                    (bsz, self.args.max_seq_len, self.n_local_heads, self.head_dim)
                ).cuda()

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

    def flush(self):
        """ Clean self.cache for next inference. """
        self.cache_v = None
        self.cache_k = None


class AbstractFeedForward(nn.Module):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        hidden_dim = int(2 * (4 * args.dim) / 3)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.hidden_dim = hidden_dim
        self.dim = args.dim
        self.w1 = None
        self.w2 = None
        self.w3 = None

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AbstractTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LlamaArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = AbstractAttention(args)
        self.feed_forward = AbstractFeedForward(args)
        self.layer_id = layer_id
        self.attention_norm = None
        self.ffn_norm = None

    def forward(self,
                x: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
                use_cache):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, use_cache)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class AbstractLlama(ParallelModelForCausalLM):
    def __init__(self, args: LlamaArgs):
        super().__init__(args.local_rank, args.world_size)
        self.params = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = None

        self.layers = torch.nn.ModuleList()
        self.norm = None
        self.output = None

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor, start_pos=0, use_cache=False):
        tokens = tokens.to(next(self.parameters()).device)
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask, use_cache)
        h = self.norm(h)
        output = self.output(h)
        return CausalLMOutputs(logits=logits_normalize(output.float()), hidden_states=h.float())

    def flush(self):
        """ Clean cache in `Attention` module """
        for i in range(self.params.n_layers):
            self.layers[i].attention.flush()
        set_barrier()


class AbstractLoraAttention(AbstractAttention):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)

        self.lora_a_wq = None
        self.lora_b_wq = None
        self.lora_a_wk = None
        self.lora_b_wk = None
        self.lora_a_wv = None
        self.lora_b_wv = None
        self.lora_a_wo = None
        self.lora_b_wo = None

    def forward(self,
                x: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
                use_cache=False):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x) + self.lora_b_wq(self.lora_a_wq(x.float())).to(x.dtype)
        xk = self.wk(x) + self.lora_b_wk(self.lora_a_wk(x.float())).to(x.dtype)
        xv = self.wv(x) + self.lora_b_wv(self.lora_a_wv(x.float())).to(x.dtype)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if use_cache:
            if self.cache_k is None:
                self.cache_k = torch.zeros(
                    (bsz, self.args.max_seq_len, self.n_local_heads, self.head_dim)
                ).cuda()
            if self.cache_v is None:
                self.cache_v = torch.zeros(
                    (bsz, self.args.max_seq_len, self.n_local_heads, self.head_dim)
                ).cuda()

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output) + self.lora_b_wo(self.lora_a_wo(output.float())).to(output.dtype)


class AbstractLoraFeedForward(AbstractFeedForward):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.r = args.r

        self.lora_a_w1 = None
        self.lora_b_w1 = None
        self.lora_a_w2 = None
        self.lora_b_w2 = None
        self.lora_a_w3 = None
        self.lora_b_w3 = None

    def forward(self, x):
        w1_x = self.w1(x) + self.lora_b_w1(self.lora_a_w1(x.float())).to(x.dtype)
        w3_x = self.w3(x) + self.lora_b_w3(self.lora_a_w3(x.float())).to(x.dtype)
        out = F.silu(w1_x) * w3_x
        return self.w2(out) + self.lora_b_w2(self.lora_a_w2(out.float())).to(out.dtype)


class AbstractLoraTransformerBlock(AbstractTransformerBlock):
    def __init__(self, layer_id: int, args: LoraLlamaArgs):
        super().__init__(layer_id, args)
        self.attention = AbstractLoraAttention(args)
        self.feed_forward = AbstractLoraFeedForward(args)


class AbstractLoraLlama(AbstractLlama):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.lora_a_output = None
        self.lora_b_output = None

    def forward(self, tokens: torch.Tensor, start_pos=0, use_cache=False):
        tokens = tokens.to(next(self.parameters()).device)
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask, use_cache)
        h = self.norm(h)
        output = self.output(h) + self.lora_b_output(self.lora_a_output(h.float())).to(h.dtype)
        return CausalLMOutputs(logits=logits_normalize(output.float()), hidden_states=h.float())

    def _freeze(self):
        """ Freeze all parameters but lora ones. """
        frozen_names = []
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad_(False)
                frozen_names.append(name)


class AbstractLoraLlamaVerifier(ParallelModule):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args.local_rank, args.world_size)
        self.params = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = None

        self.layers = torch.nn.ModuleList()
        self.norm = None

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        self.v_head = None

    def forward(self, tokens: torch.Tensor):
        start_pos = 0
        use_cache = False
        tokens = tokens.to(next(self.parameters()).device)
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask, use_cache)
        h = self.norm(h).float()
        scores = self.v_head(h).squeeze(-1)  # [b, s]
        return scores

    def _freeze(self):
        """ Freeze all parameters but lora ones. """
        frozen_names = []
        for name, param in self.named_parameters():
            if 'lora' not in name and 'v_head' not in name:
                param.requires_grad_(False)
                frozen_names.append(name)
