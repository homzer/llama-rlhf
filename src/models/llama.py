from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding
)

from src.models.modeling import ParallelModelForCausalLM, CausalLMOutputs, AttentionForCausalLM, \
    ParallelVerifier, VerifierOutputs
from src.models.modeling_acts import RMSNorm
from src.models.modeling_args import LlamaArgs, LoraLlamaArgs
from src.utils import apply_rotary_emb, precompute_freqs_cis, set_barrier, logits_normalize, clamp, apply_lora


class Attention(AttentionForCausalLM):
    def __init__(self, args: LlamaArgs):
        super().__init__(args.max_seq_len)
        self.args = args
        self.n_local_heads = args.n_heads // args.world_size
        self.head_dim = args.dim // args.n_heads

        self.wq = None
        self.wk = None
        self.wv = None
        self.wo = None

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = clamp(self.wq(x)), clamp(self.wk(x)), clamp(self.wv(x))

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)
        output = self.apply_attention(xq, xk, xv, mask)
        return clamp(self.wo(output))

    def init_weights(self):
        self.wq = ColumnParallelLinear(
            self.args.dim,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            self.args.dim,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            self.args.dim,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            self.args.n_heads * self.head_dim,
            self.args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )


class FeedForward(nn.Module):
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
        return clamp(self.w2(clamp(F.silu(self.w1(x)) * self.w3(x))))

    def init_weights(self):
        self.w1 = ColumnParallelLinear(
            self.dim,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            self.hidden_dim,
            self.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            self.dim,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LlamaArgs):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = None
        self.ffn_norm = None

    def forward(self,
                x: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
                use_cache):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, use_cache)
        h = h + self.feed_forward.forward(self.ffn_norm(h))
        return h

    def init_weights(self):
        self.attention.init_weights()
        self.feed_forward.init_weights()
        self.attention_norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        self.ffn_norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)


class Llama(ParallelModelForCausalLM):
    def __init__(self, args: LlamaArgs):
        super().__init__(args.local_rank, args.world_size)
        self.args = args

        self.tok_embeddings = None
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))
        self.norm = None
        self.output = None

        self.freqs_cis = precompute_freqs_cis(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2
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
        output = clamp(self.output(h))
        return CausalLMOutputs(logits=logits_normalize(output), hidden_states=h)

    def init_weights(self):
        self.tok_embeddings = ParallelEmbedding(
            self.args.vocab_size, self.args.dim, init_method=lambda x: x
        )
        for layer in self.layers:
            layer.init_weights()
        self.norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        self.output = ColumnParallelLinear(
            self.args.dim, self.args.vocab_size, bias=False, init_method=lambda x: x
        )

    def load(self, ckpt_dir: str, verbose: bool = True, merge_lora: bool = False):
        super().load(ckpt_dir, verbose, merge_lora=merge_lora)

    def flush(self):
        """ Clean cache in `Attention` module """
        for i in range(self.args.n_layers):
            self.layers[i].attention.flush()
        set_barrier()


class LlamaVerifier(ParallelVerifier):
    def __init__(self, args: LlamaArgs):
        super().__init__(args.local_rank, args.world_size)
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = None
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))
        self.norm = None
        self.v_head = None

        self.freqs_cis = precompute_freqs_cis(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor) -> VerifierOutputs:
        tokens = tokens.to(next(self.parameters()).device)
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[: seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).type_as(h)

        for layer in self.layers:
            h = layer(h, 0, freqs_cis, mask, use_cache=False)
        h = self.norm(h)
        scores = self.v_head(h.type_as(self.v_head.weight)).squeeze(-1)  # [b, s]
        return VerifierOutputs(scores=scores)

    def init_weights(self):
        self.tok_embeddings = ParallelEmbedding(
            self.args.vocab_size, self.args.dim, init_method=lambda x: x
        )
        for layer in self.layers:
            layer.init_weights()
        self.norm = RMSNorm(self.args.dim, eps=self.args.norm_eps)
        self.v_head = nn.Linear(self.args.dim, 1, bias=False)

    def load(self, ckpt_dir: str, verbose: bool = True, merge_lora: bool = False):
        super().load(ckpt_dir, verbose, merge_lora=merge_lora)


class LoraAttention(Attention):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.args = args
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
        xq = clamp(self.wq(x) + apply_lora(x, self.lora_a_wq, self.lora_b_wq))
        # xq = clamp(self.wq(x) + self.lora_b_wq(self.lora_a_wq(x.float())).to(x.dtype))
        xk = clamp(self.wk(x) + apply_lora(x, self.lora_a_wk, self.lora_b_wk))
        # xk = clamp(self.wk(x) + self.lora_b_wk(self.lora_a_wk(x.float())).to(x.dtype))
        xv = clamp(self.wv(x) + apply_lora(x, self.lora_a_wv, self.lora_b_wv))
        # xv = clamp(self.wv(x) + self.lora_b_wv(self.lora_a_wv(x.float())).to(x.dtype))

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)
        output = self.apply_attention(xq, xk, xv, mask)
        # return clamp(self.wo(output) + self.lora_b_wo(self.lora_a_wo(output.float())).to(output.dtype))
        return clamp(self.wo(output) + apply_lora(output, self.lora_a_wo, self.lora_b_wo))

    def init_weights(self):
        super().init_weights()

        self.lora_a_wq = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).float()
        self.lora_b_wq = ColumnParallelLinear(
            self.args.r,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()
        self.lora_a_wk = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).float()
        self.lora_b_wk = ColumnParallelLinear(
            self.args.r,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()
        self.lora_a_wv = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).float()
        self.lora_b_wv = ColumnParallelLinear(
            self.args.r,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()
        self.lora_a_wo = RowParallelLinear(
            self.args.n_heads * self.head_dim,
            self.args.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).float()
        self.lora_b_wo = nn.Linear(
            self.args.r,
            self.args.dim,
            bias=False
        ).float()
        init.zeros_(self.lora_b_wo.weight)


class LoraFeedForward(FeedForward):
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
        # self.lora_b_w1(self.lora_a_w1(x.float())).to(x.dtype)
        w1_x = self.w1(x) + apply_lora(x, self.lora_a_w1, self.lora_b_w1)
        # self.lora_b_w3(self.lora_a_w3(x.float())).to(x.dtype)
        w3_x = self.w3(x) + apply_lora(x, self.lora_a_w3, self.lora_b_w3)
        out = clamp(F.silu(w1_x) * w3_x)
        # self.lora_b_w2(self.lora_a_w2(out.float())).to(out.dtype)
        return clamp(self.w2(out) + apply_lora(out, self.lora_a_w2, self.lora_b_w2))

    def init_weights(self):
        super().init_weights()

        self.lora_a_w1 = nn.Linear(
            self.dim,
            self.r,
            bias=False
        ).float()
        self.lora_b_w1 = ColumnParallelLinear(
            self.r,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()
        self.lora_a_w2 = RowParallelLinear(
            self.hidden_dim,
            self.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).float()
        self.lora_b_w2 = nn.Linear(
            self.r,
            self.dim,
            bias=False
        ).float()
        init.zeros_(self.lora_b_w2.weight)
        self.lora_a_w3 = nn.Linear(
            self.dim,
            self.r,
            bias=False
        ).float()
        self.lora_b_w3 = ColumnParallelLinear(
            self.r,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).float()


class LoraTransformerBlock(TransformerBlock):
    def __init__(self, layer_id: int, args: LoraLlamaArgs):
        super().__init__(layer_id, args)
        self.attention = LoraAttention(args)
        self.feed_forward = LoraFeedForward(args)


class LoraLlama(Llama):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.args = args
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(LoraTransformerBlock(layer_id, args))
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
        # self.lora_b_output(self.lora_a_output(h.float())).to(h.dtype)
        output = clamp(self.output(h) + apply_lora(h, self.lora_a_output, self.lora_b_output))
        return CausalLMOutputs(logits=logits_normalize(output), hidden_states=h)

    def init_weights(self):
        super().init_weights()

        self.lora_a_output = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).float()
        self.lora_b_output = ColumnParallelLinear(
            self.args.r,
            self.args.vocab_size,
            bias=False,
            gather_output=True,
            init_method=init.zeros_
        ).float()

        # Freeze parameters
        self._freeze()

    # lora op
    def _freeze(self):
        """ Freeze all parameters but lora ones. """
        frozen_names = []
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad_(False)
                frozen_names.append(name)


class LoraLlamaVerifier(LlamaVerifier):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.args = args
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(LoraTransformerBlock(layer_id, args))

    def init_weights(self):
        super().init_weights()

        # Freeze parameters
        self._freeze()

    def _freeze(self):
        """ Freeze all parameters but lora ones. """
        frozen_names = []
        for name, param in self.named_parameters():
            if 'lora' not in name and 'v_head' not in name:
                param.requires_grad_(False)
                frozen_names.append(name)
