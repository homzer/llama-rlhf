""" Attention-Based Credit for PPO """
import collections
import gc
import math
from typing import Optional, List, Union

import fire
import numpy as np
import torch

from policy_train_ppo import collect_actor_buffer, collect_reference_buffer, collect_critic_buffer, train_actor, \
    train_critic
from src.dataset import JsonDataset
from src.entities import Timer
from src.generator import GeneratorForVerifier
from src.models import Llama3Verifier
from src.models.llama import LlamaAttention, LlamaTransformerBlock, LlamaVerifier
from src.models.modeling_args import LlamaArgs
from src.parallel.utils import setup_model_parallel, set_barrier
from src.ppo.buffer import CriticRolloutBuffer, RolloutBuffer, ActorRolloutBuffer
from src.tokenizers import Tokenizer, LlamaTokenizer, Llama3Tokenizer
from src.utils import apply_rotary_emb
from src.utils import masked_mean, json_load


def apply_attention(xq, xk, xv, mask):
    bsz, seqlen, n_heads, head_dim = xq.shape
    xq = xq.transpose(1, 2)
    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)
    scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(head_dim)
    if mask is not None:
        scores = scores + mask
    scores = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(scores, xv)
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return output, scores


class LlamaAttentionForAttentionBasedCredit(LlamaAttention):
    def __init__(self, args: LlamaArgs, layer_id: int):
        super().__init__(args)
        self.attention_scores = None
        self.layer_id = layer_id

    def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
            use_cache=False
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        if use_cache:
            xk, xv = self.apply_cache(xk, xv, start_pos)

        xk, xv = self.repeat_kv(xk, xv, self.n_rep)

        output, scores = apply_attention(xq, xk, xv, mask)
        if self.args.n_layers - 1 == self.layer_id:  # only store the last layer
            self.attention_scores = torch.mean(scores, dim=1)  # [b, s, s]
        return self.wo(output)


class LlamaTransformerBlockForAttentionBasedCredit(LlamaTransformerBlock):
    def __init__(self, layer_id: int, args: LlamaArgs):
        super().__init__(layer_id, args)
        self.attention = LlamaAttentionForAttentionBasedCredit(args, layer_id)


class LlamaVerifierForAttentionBasedCredit(LlamaVerifier):
    def __init__(self, args: LlamaArgs):
        super().__init__(args)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(LlamaTransformerBlockForAttentionBasedCredit(layer_id, args))

    def fetch_last_layer_attention_scores(self) -> torch.Tensor:
        return self.layers[-1].attention.attention_scores


class Llama3VerifierForAttentionBasedCredit(Llama3Verifier):
    def __init__(self, args: LlamaArgs):
        super().__init__(args)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(LlamaTransformerBlockForAttentionBasedCredit(layer_id, args))

    def fetch_last_layer_attention_scores(self) -> torch.Tensor:
        return self.layers[-1].attention.attention_scores


class PairwiseVerifierStrategyForAttentionBasedCredit:
    @staticmethod
    def generator_forward(
            scores: torch.Tensor, masks: torch.Tensor, attention_scores: torch.Tensor
    ) -> List[List[float]]:
        scores = scores.detach().cpu()  # [b, s]
        bsz = scores.shape[0]
        attention_scores = attention_scores.detach().cpu()  # [b, s, s]
        tokens_scores = []
        for i in range(bsz):
            check_end = masks[i].nonzero()
            if len(check_end) == 0:
                print("Warming: instruction len out of range. Setting reward score to 0.")
                tokens_scores.append([])
                continue
            eos_pos = check_end[-1].item()
            eos_reward = scores[i][eos_pos].item()
            eos_attention = attention_scores[i][eos_pos][masks[i]]  # [s]
            tokens_scores.append((eos_attention * eos_reward).tolist())
        return tokens_scores


class VerifierGeneratorForAttentionBasedCredit(GeneratorForVerifier):
    def __init__(
            self,
            model: Union[LlamaVerifierForAttentionBasedCredit, Llama3VerifierForAttentionBasedCredit],
            tokenizer: Tokenizer,
            max_seq_len: int,
    ):
        super().__init__(model=model, tokenizer=tokenizer, max_seq_len=max_seq_len)
        self.strategy = PairwiseVerifierStrategyForAttentionBasedCredit()

    def forward(self, instructions: Union[List[str], List[List[int]]], outputs: Union[List[str], List[List[int]]]):
        self.model.eval()
        examples = self.prepare_for_generation(instructions, outputs)
        with torch.no_grad():
            scores = self.model.forward(examples.tokens).scores
        Outputs = collections.namedtuple("Outputs", ["tokens_scores"])
        return Outputs(tokens_scores=self.strategy.generator_forward(
            scores=scores, masks=examples.masks, attention_scores=self.model.fetch_last_layer_attention_scores()
        ))


class CriticBufferCollectorForAttentionBasedCredit:
    def __init__(
            self,
            critic: Union[LlamaVerifierForAttentionBasedCredit, Llama3VerifierForAttentionBasedCredit],
            tokenizer: Tokenizer,
            max_seq_len: int
    ):
        self.generator = VerifierGeneratorForAttentionBasedCredit(
            model=critic, tokenizer=tokenizer, max_seq_len=max_seq_len
        )

    def forward(self, instructions: List[str], actions: np.ndarray, action_masks: np.ndarray) -> CriticRolloutBuffer:
        responses = []
        for action, action_mask in zip(actions, action_masks):
            responses.append(action[action_mask].tolist())
        token_scores = self.generator.forward(instructions, responses).tokens_scores
        return CriticRolloutBuffer(token_scores, action_masks)


def collect_verifier_buffer(
        verifier_model_type: str,
        verifier_config_file: str,
        max_seq_len: int,
        verifier_tokenizer_file: str,
        dtype: str,
        verifier_ckpt_dir: str,
        actor_rollout_buffer: ActorRolloutBuffer,
        max_forward_batch_size: int,
) -> CriticRolloutBuffer:
    assert verifier_model_type in ["llama", "llama3"]
    if verifier_model_type == "llama":
        verifier_tokenizer = LlamaTokenizer(verifier_tokenizer_file)
        verifier = LlamaVerifierForAttentionBasedCredit(LlamaArgs(
            max_seq_len=max_seq_len,
            dtype=dtype,
            use_clamp=False,
            use_logits_normalize=True
        ).from_json(verifier_config_file))
    else:
        verifier_tokenizer = Llama3Tokenizer(verifier_tokenizer_file)
        verifier = Llama3VerifierForAttentionBasedCredit(LlamaArgs(
            max_seq_len=max_seq_len,
            dtype=dtype,
            use_clamp=False,
            use_logits_normalize=True
        ).from_json(verifier_config_file))
    verifier.init_weights()
    verifier.load(verifier_ckpt_dir)
    verifier_buffer_collector = CriticBufferCollectorForAttentionBasedCredit(verifier, verifier_tokenizer, max_seq_len)
    verifier_rollout_buffer = CriticRolloutBuffer()
    print('Reward buffer collecting ...')
    timer = Timer(total=len(actor_rollout_buffer) // max_forward_batch_size, episode=10)
    for data in actor_rollout_buffer.get(max_forward_batch_size):
        timer.step()
        verifier_rollout_buffer.extend(
            verifier_buffer_collector.forward(
                data.instructions, data.actions, data.action_masks
            )
        )

    verifier.cpu()
    del verifier
    del verifier_buffer_collector
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    return verifier_rollout_buffer


def run(
        train_file: str,
        actor_ckpt_dir: str,
        actor_model_type: str,
        actor_save_dir: str,
        critic_ckpt_dir: str,
        critic_model_type: str,
        critic_save_dir: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,
        actor_config_file: str = None,
        actor_tokenizer_file: str = None,
        critic_config_file: str = None,
        critic_tokenizer_file: str = None,
        verifier_config_file: str = None,
        verifier_tokenizer_file: str = None,
        reference_ckpt_dir: str = None,
        actor_lora_rank: int = -1,
        actor_lora_dtype: str = "bfloat16",
        critic_lora_rank: int = -1,
        critic_lora_dtype: str = "bfloat16",
        actor_max_batch_size: int = 1,
        critic_max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 24,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-6,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        kl_coef: float = 0.1,
        clip_range: float = 0.2,
        use_chat_template: bool = False
):
    setup_model_parallel()
    actor_config_file = actor_config_file or actor_ckpt_dir
    actor_tokenizer_file = actor_tokenizer_file or actor_ckpt_dir
    critic_config_file = critic_config_file or critic_ckpt_dir
    critic_tokenizer_file = critic_tokenizer_file or critic_ckpt_dir
    verifier_config_file = verifier_config_file or verifier_ckpt_dir
    verifier_tokenizer_file = verifier_tokenizer_file or verifier_ckpt_dir

    datalist = json_load(train_file)
    chunk_size = chunk_size or len(datalist)
    local_epochs = len(datalist) // chunk_size
    begin_global_epoch = begin_epoch // local_epochs
    begin_local_epoch = begin_epoch % local_epochs
    for global_epoch in range(begin_global_epoch, epochs):
        for local_epoch in range(begin_local_epoch, local_epochs):
            epoch = local_epoch + global_epoch * local_epochs
            print(f"Epoch - {epoch} of {local_epochs * epochs}")
            dataset = JsonDataset(f=datalist[local_epoch * chunk_size: (local_epoch + 1) * chunk_size])
            if len(dataset) == 0:
                continue

            # Collecting actor buffer
            actor_rollout_buffer = collect_actor_buffer(
                actor_model_type=actor_model_type,
                actor_config_file=actor_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=actor_tokenizer_file,
                dtype=dtype,
                actor_ckpt_dir=actor_ckpt_dir,
                epoch=epoch,
                actor_save_dir=actor_save_dir,
                use_chat_template=use_chat_template,
                dataset=dataset,
                max_generate_batch_size=max_generate_batch_size,
                temperature=temperature,
                top_p=top_p,
                num_samples_per_prompt=num_samples_per_prompt
            )

            reference_rollout_buffer = None
            if reference_ckpt_dir is not None:
                # Collecting reference logprobs
                reference_rollout_buffer = collect_reference_buffer(
                    actor_model_type=actor_model_type,
                    actor_config_file=actor_config_file,
                    max_seq_len=max_seq_len,
                    actor_tokenizer_file=actor_tokenizer_file,
                    dtype=dtype,
                    reference_ckpt_dir=reference_ckpt_dir,
                    actor_rollout_buffer=actor_rollout_buffer,
                    max_forward_batch_size=max_forward_batch_size
                )

            # Collecting critic buffer
            critic_rollout_buffer = collect_critic_buffer(
                critic_model_type=critic_model_type,
                critic_config_file=critic_config_file,
                max_seq_len=max_seq_len,
                critic_tokenizer_file=critic_tokenizer_file,
                dtype=dtype,
                critic_ckpt_dir=critic_ckpt_dir,
                epoch=epoch,
                critic_save_dir=critic_save_dir,
                actor_rollout_buffer=actor_rollout_buffer,
                max_forward_batch_size=max_forward_batch_size
            )

            # Collecting verifier buffer
            verifier_rollout_buffer = collect_verifier_buffer(
                verifier_model_type=verifier_model_type,
                verifier_config_file=verifier_config_file,
                max_seq_len=max_seq_len,
                verifier_tokenizer_file=verifier_tokenizer_file,
                dtype=dtype,
                verifier_ckpt_dir=verifier_ckpt_dir,
                actor_rollout_buffer=actor_rollout_buffer,
                max_forward_batch_size=max_forward_batch_size
            )

            print("Average Rewards: ", masked_mean(verifier_rollout_buffer.scores, actor_rollout_buffer.action_masks))

            rollout_buffer = RolloutBuffer(
                obs=actor_rollout_buffer.obs,
                actions=actor_rollout_buffer.actions,
                rewards=verifier_rollout_buffer.scores,
                values=critic_rollout_buffer.scores,
                action_logits=actor_rollout_buffer.action_logits,
                action_masks=actor_rollout_buffer.action_masks,
                action_logprobs=actor_rollout_buffer.action_logprobs,
                ref_action_logprobs=reference_rollout_buffer.output_tokens_logps if (
                        reference_rollout_buffer is not None
                ) else None,
                kl_coef=kl_coef
            )

            # Actor training
            train_actor(
                actor_model_type=actor_model_type,
                actor_config_file=actor_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=actor_tokenizer_file,
                actor_lora_rank=actor_lora_rank,
                dtype=dtype,
                actor_lora_dtype=actor_lora_dtype,
                lr=lr,
                epoch=epoch,
                actor_ckpt_dir=actor_ckpt_dir,
                actor_save_dir=actor_save_dir,
                rollout_buffer=rollout_buffer,
                actor_max_batch_size=actor_max_batch_size,
                inner_epochs=inner_epochs,
                clip_range=clip_range
            )

            train_critic(
                critic_model_type=critic_model_type,
                critic_config_file=critic_config_file,
                max_seq_len=max_seq_len,
                critic_tokenizer_file=critic_tokenizer_file,
                critic_lora_rank=critic_lora_rank,
                dtype=dtype,
                lr=lr,
                critic_lora_dtype=critic_lora_dtype,
                critic_ckpt_dir=critic_ckpt_dir,
                epoch=epoch,
                critic_save_dir=critic_save_dir,
                rollout_buffer=rollout_buffer,
                critic_max_batch_size=critic_max_batch_size,
                inner_epochs=inner_epochs
            )


if __name__ == '__main__':
    fire.Fire(run)
