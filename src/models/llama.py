from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.checkpoint import CheckpointForLlama
from src.models.modeling import (
    ParallelModelForCausalLM,
    CausalLMOutputs,
    AttentionForCausalLM,
    ParallelVerifier,
    VerifierOutputs
)
from src.models.modeling_acts import RMSNorm, Clamp, LogitsNormalize
from src.models.modeling_args import LlamaArgs, LoraLlamaArgs
from src.parallel.initialize import set_model_parallel_barrier
from src.parallel.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding
)
from src.parallel.sequence_parallel.mappings import (
    scatter_to_sequence_parallel_region,
    gather_from_sequence_parallel_region
)
from src.utils import precompute_freqs_cis, apply_lora, apply_rotary_emb_


class LlamaAttention(AttentionForCausalLM):
    def __init__(self, args: LlamaArgs):
        super().__init__(args.max_seq_len)
        self.args = args
        self.n_kv_heads = args.n_kv_heads or args.n_heads
        self.n_local_heads = args.n_heads // args.model_parallel_world_size
        self.n_local_kv_heads = self.n_kv_heads // args.model_parallel_world_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = None
        self.wk = None
        self.wv = None
        self.wo = None
        self.wq_fn = lambda x: self.wq(x)
        self.wk_fn = lambda x: self.wk(x)
        self.wv_fn = lambda x: self.wv(x)
        self.wo_fn = lambda x: self.wo(x)

    def init_weights(self):
        self.wq = ColumnParallelLinear(
            self.args.dim,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            ).type(self.args.dtype)
        self.wk = ColumnParallelLinear(
            self.args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.wv = ColumnParallelLinear(
            self.args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
            ).type(self.args.dtype)
        self.wo = RowParallelLinear(
            self.args.n_heads * self.head_dim,
            self.args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
            ).type(self.args.dtype)

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq_fn(x), self.wk_fn(x), self.wv_fn(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        local_freqs_cis = freqs_cis
        if not use_cache:
            xk = gather_from_sequence_parallel_region(xk)
            xv = gather_from_sequence_parallel_region(xv)
            local_freqs_cis = scatter_to_sequence_parallel_region(freqs_cis.unsqueeze(0)).squeeze(0)

        xq = apply_rotary_emb_(xq, freqs_cis=local_freqs_cis)
        xk = apply_rotary_emb_(xk, freqs_cis=freqs_cis)

        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)

        xk, xv = self.repeat_kv(xk, xv, self.n_rep)

        output = self.apply_attention(xq, xk, xv, mask)
        return self.wo_fn(output)


class LlamaFeedForward(nn.Module):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        self.args = args
        self.ffn_dim_multiplier = args.ffn_dim_multiplier or 1
        hidden_dim = int(2 * (4 * args.dim) / 3)
        hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.hidden_dim = hidden_dim
        self.dim = args.dim
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.w1_fn = lambda x: self.w1(x)
        self.w2_fn = lambda x: self.w2(x)
        self.w3_fn = lambda x: self.w3(x)

    def init_weights(self):
        self.w1 = ColumnParallelLinear(
            self.dim,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.w2 = RowParallelLinear(
            self.hidden_dim,
            self.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        ).type(self.args.dtype)
        self.w3 = ColumnParallelLinear(
            self.dim,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        ).type(self.args.dtype)

    def forward(self, x):
        return self.w2_fn(F.silu(self.w1_fn(x)) * self.w3_fn(x))


class LlamaTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LlamaArgs):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.attention = LlamaAttention(args)
        self.feed_forward = LlamaFeedForward(args)
        self.clamp = Clamp(enable=args.use_clamp)

        self.attention_norm = None
        self.ffn_norm = None

    def forward(self,
                x: torch.Tensor,
                start_pos: int,
                freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor],
                use_cache):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, use_cache)
        h = self.clamp.forward(h)
        h = h + self.feed_forward.forward(self.ffn_norm(h))
        h = self.clamp.forward(h)
        return h

    def init_weights(self):
        self.attention.init_weights()
        self.feed_forward.init_weights()
        self.attention_norm = RMSNorm(self.args.dim, eps=self.args.norm_eps).type(self.args.dtype)
        self.ffn_norm = RMSNorm(self.args.dim, eps=self.args.norm_eps).type(self.args.dtype)


class Llama(ParallelModelForCausalLM):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        self.args = args

        self.tok_embeddings = None
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(LlamaTransformerBlock(layer_id, args))
        self.norm = None
        self.output = None
        self.output_fn = lambda x: self.output(x)
        self.logits_norm = LogitsNormalize(enable=self.args.use_logits_normalize)

        self.freqs_cis = precompute_freqs_cis(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, self.args.rope_theta
        )  # [s * 2, head_dim / 2]
        self.checkpoint = CheckpointForLlama()

    def forward(self, tokens: torch.Tensor, start_pos=0, use_cache=False):
        tokens = tokens.to(next(self.parameters()).device)
        _bsz, seq_len = tokens.shape
        freqs_cis = self.freqs_cis[start_pos: start_pos + seq_len].to(tokens.device)

        if not use_cache:
            tokens = scatter_to_sequence_parallel_region(tokens)

        h = self.tok_embeddings(tokens)

        mask = None
        if seq_len > 1:
            mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

            if not use_cache:
                mask = scatter_to_sequence_parallel_region(mask.transpose(1, 2)).transpose(1, 2)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask, use_cache)
        h = self.norm(h)
        logits = self.output_fn(h)

        if not use_cache:
            logits = gather_from_sequence_parallel_region(logits)

        return CausalLMOutputs(logits=self.logits_norm.forward(logits), hidden_states=h)

    def init_weights(self):
        self.tok_embeddings = ParallelEmbedding(
            self.args.vocab_size, self.args.dim, init_method=lambda x: x
        ).type(self.args.dtype)
        for layer in self.layers:
            layer.init_weights()
        self.norm = RMSNorm(self.args.dim, eps=self.args.norm_eps).type(self.args.dtype)
        self.output = ColumnParallelLinear(
            self.args.dim, self.args.vocab_size, bias=False, init_method=lambda x: x
        ).type(self.args.dtype)

    # Copied from llama_hf.LlamaHf.load
    def load(self, ckpt_dir: str, verbose: bool = True, **kwargs):
        ckpt_dir = self.checkpoint.auto_split_or_merge_checkpoints(
            ckpt_dir=ckpt_dir,
            model_parallel_world_size=self.model_parallel_world_size,
            global_rank=self.global_rank
        )
        merge_lora = kwargs.get("merge_lora", True)
        super().load(ckpt_dir, verbose=verbose, merge_lora=merge_lora)

    def flush(self):
        """ Clean cache in `LlamaAttention` module """
        for i in range(self.args.n_layers):
            self.layers[i].attention.flush()
        set_model_parallel_barrier()

    def rearrange_kv_cache(self, indices: torch.Tensor):
        """ Rearrange the order of the KV cache in `LlamaAttention` module """
        for i in range(self.args.n_layers):
            self.layers[i].attention.rearrange(indices)
        set_model_parallel_barrier()


class LlamaVerifier(ParallelVerifier):
    def __init__(self, args: LlamaArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = None
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(LlamaTransformerBlock(layer_id, args))
        self.norm = None
        self.v_head = None
        self.v_head_fn = lambda x: self.v_head(x)

        self.freqs_cis = precompute_freqs_cis(
            self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, self.args.rope_theta
        )
        self.checkpoint = CheckpointForLlama()

    def forward(self, tokens: torch.Tensor) -> VerifierOutputs:
        tokens = tokens.to(next(self.parameters()).device)
        _bsz, seq_len = tokens.shape
        freqs_cis = self.freqs_cis[: seq_len].to(tokens.device)

        # Sequence parallel op.
        tokens = scatter_to_sequence_parallel_region(tokens)

        h = self.tok_embeddings(tokens)

        mask = None
        if seq_len > 1:
            mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).type_as(h)

            # Sequence parallel op.
            mask = scatter_to_sequence_parallel_region(mask.transpose(1, 2)).transpose(1, 2)

        for layer in self.layers:
            h = layer(h, 0, freqs_cis, mask, use_cache=False)
        h = self.norm(h)
        scores = self.v_head_fn(h).squeeze(-1)  # [b, s]

        # Sequence parallel op.
        scores = gather_from_sequence_parallel_region(scores)

        return VerifierOutputs(scores=scores)

    def init_weights(self):
        self.tok_embeddings = ParallelEmbedding(
            self.args.vocab_size, self.args.dim, init_method=lambda x: x
        ).type(self.args.dtype)
        for layer in self.layers:
            layer.init_weights()
        self.norm = RMSNorm(self.args.dim, eps=self.args.norm_eps).type(self.args.dtype)
        self.v_head = nn.Linear(self.args.dim, 1, bias=False).type(self.args.dtype)

    def load(self, ckpt_dir: str, verbose: bool = True, **kwargs):
        ckpt_dir = self.checkpoint.auto_split_or_merge_checkpoints(
            ckpt_dir=ckpt_dir,
            model_parallel_world_size=self.model_parallel_world_size,
            global_rank=self.global_rank
        )
        merge_lora = kwargs.get("merge_lora", True)
        super().load(ckpt_dir, verbose=verbose, merge_lora=merge_lora)


class LoraLlamaAttention(LlamaAttention):
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
        self.wq_fn = lambda x: self.wq(x) + apply_lora(x, self.lora_a_wq, self.lora_b_wq)
        self.wk_fn = lambda x: self.wk(x) + apply_lora(x, self.lora_a_wk, self.lora_b_wk)
        self.wv_fn = lambda x: self.wv(x) + apply_lora(x, self.lora_a_wv, self.lora_b_wv)
        self.wo_fn = lambda x: self.wo(x) + apply_lora(x, self.lora_a_wo, self.lora_b_wo)

    def init_weights(self):
        super().init_weights()

        self.lora_a_wq = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_wq = ColumnParallelLinear(
            self.args.r,
            self.args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_wk = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_wk = ColumnParallelLinear(
            self.args.r,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_wv = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_wv = ColumnParallelLinear(
            self.args.r,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_wo = RowParallelLinear(
            self.args.n_heads * self.head_dim,
            self.args.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).type(self.args.lora_dtype)
        self.lora_b_wo = nn.Linear(
            self.args.r,
            self.args.dim,
            bias=False
        ).type(self.args.lora_dtype)
        init.zeros_(self.lora_b_wo.weight)


class LoraLlamaFeedForward(LlamaFeedForward):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.args = args
        self.r = args.r

        self.lora_a_w1 = None
        self.lora_b_w1 = None
        self.lora_a_w2 = None
        self.lora_b_w2 = None
        self.lora_a_w3 = None
        self.lora_b_w3 = None
        self.w1_fn = lambda x: self.w1(x) + apply_lora(x, self.lora_a_w1, self.lora_b_w1)
        self.w2_fn = lambda x: self.w2(x) + apply_lora(x, self.lora_a_w2, self.lora_b_w2)
        self.w3_fn = lambda x: self.w3(x) + apply_lora(x, self.lora_a_w3, self.lora_b_w3)

    def init_weights(self):
        super().init_weights()

        self.lora_a_w1 = nn.Linear(
            self.dim,
            self.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_w1 = ColumnParallelLinear(
            self.r,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)
        self.lora_a_w2 = RowParallelLinear(
            self.hidden_dim,
            self.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_,
        ).type(self.args.lora_dtype)
        self.lora_b_w2 = nn.Linear(
            self.r,
            self.dim,
            bias=False
        ).type(self.args.lora_dtype)
        init.zeros_(self.lora_b_w2.weight)
        self.lora_a_w3 = nn.Linear(
            self.dim,
            self.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_w3 = ColumnParallelLinear(
            self.r,
            self.hidden_dim,
            bias=False,
            gather_output=False,
            init_method=init.zeros_,
        ).type(self.args.lora_dtype)


class LoraLlamaTransformerBlock(LlamaTransformerBlock):
    def __init__(self, layer_id: int, args: LoraLlamaArgs):
        super().__init__(layer_id, args)
        self.attention = LoraLlamaAttention(args)
        self.feed_forward = LoraLlamaFeedForward(args)


class LoraLlama(Llama):
    def __init__(self, args: LoraLlamaArgs):
        super().__init__(args)
        self.args = args
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(LoraLlamaTransformerBlock(layer_id, args))
        self.lora_a_output = None
        self.lora_b_output = None
        self.output_fn = lambda x: self.output(x) + apply_lora(x, self.lora_a_output, self.lora_b_output)

    def load(self, ckpt_dir: str, verbose: bool = True, merge_lora: bool = False):
        super().load(ckpt_dir=ckpt_dir, verbose=verbose, merge_lora=merge_lora)

    def init_weights(self):
        super().init_weights()

        self.lora_a_output = nn.Linear(
            self.args.dim,
            self.args.r,
            bias=False
        ).type(self.args.lora_dtype)
        self.lora_b_output = ColumnParallelLinear(
            self.args.r,
            self.args.vocab_size,
            bias=False,
            gather_output=True,
            init_method=init.zeros_
        ).type(self.args.lora_dtype)

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
            self.layers.append(LoraLlamaTransformerBlock(layer_id, args))

    def load(self, ckpt_dir: str, verbose: bool = True, merge_lora: bool = False):
        super().load(ckpt_dir=ckpt_dir, verbose=verbose, merge_lora=merge_lora)

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
