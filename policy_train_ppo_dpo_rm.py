import fire
import torch

from policy_train_ppo import (
    collect_actor_buffer,
    collect_reference_buffer,
    collect_critic_buffer,
    train_actor,
    train_critic
)
from src.dataset import JsonDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.models.modeling import ParallelVerifier, VerifierOutputs, ParallelModelForCausalLM
from src.parallel.utils import setup_model_parallel, set_barrier
from src.ppo.buffer import RolloutBuffer, ActorRolloutBuffer, CriticRolloutBuffer
from src.ppo.collector import CriticBufferCollector
from src.utils import masked_mean, json_load


class VerifierForDPO(ParallelVerifier):
    def __init__(self, model: ParallelModelForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> VerifierOutputs:
        labels = torch.full_like(tokens, fill_value=0)
        labels[:, :-1] = tokens[:, 1:]
        logits = self.model.forward(tokens).logits
        log_probs = torch.log_softmax(
            logits.float() if logits.dtype == torch.float16 else logits, dim=-1
        ).type_as(logits)
        labels = labels.to(logits.device)
        log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        return VerifierOutputs(scores=log_probs)


def collect_verifier_buffer(
        verifier_model_type: str,
        verifier_config_file: str,
        max_seq_len: int,
        verifier_tokenizer_file: str,
        dtype: str,
        verifier_ckpt_dir: str,
        verifier_reference_ckpt_dir: str,
        actor_rollout_buffer: ActorRolloutBuffer,
        max_forward_batch_size: int,
        beta: float,
) -> CriticRolloutBuffer:
    verifier, verifier_tokenizer = get_parallel_model(
        model_type=verifier_model_type,
        config_file=verifier_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=verifier_tokenizer_file,
        dtype=dtype
    )
    verifier.load(verifier_ckpt_dir)
    verifier = VerifierForDPO(verifier)
    verifier_buffer_collector = CriticBufferCollector(verifier, verifier_tokenizer, max_seq_len)
    verifier_rollout_buffer = CriticRolloutBuffer()
    print("Reward buffer collecting ...")
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
    set_barrier()

    reference, reference_tokenizer = get_parallel_model(
        model_type=verifier_model_type,
        config_file=verifier_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=verifier_tokenizer_file,
        dtype=dtype
    )
    reference.load(verifier_reference_ckpt_dir)
    reference = VerifierForDPO(reference)
    reference_buffer_collector = CriticBufferCollector(reference, reference_tokenizer, max_seq_len)
    reference_rollout_buffer = CriticRolloutBuffer()
    timer = Timer(total=len(actor_rollout_buffer) // max_forward_batch_size, episode=10)
    for data in actor_rollout_buffer.get(max_forward_batch_size):
        timer.step()
        reference_rollout_buffer.extend(
            reference_buffer_collector.forward(
                data.instructions, data.actions, data.action_masks
            )
        )

    reference.cpu()
    del reference
    del reference_buffer_collector
    torch.cuda.empty_cache()
    set_barrier()

    # beta * (log_probs - ref_log_probs)
    verifier_rollout_buffer.scores = beta * (verifier_rollout_buffer.scores - reference_rollout_buffer.scores)

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
        verifier_reference_ckpt_dir: str,
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
        max_forward_batch_size: int = 36,
        max_seq_len: int = 1024,
        chunk_size: int = None,
        epochs: int = 1,
        inner_epochs: int = 3,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        kl_ceof: float = 0.1,
        beta: float = 0.1,
        clip_range: float = 0.2,
        use_chat_template: bool = False,
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
                max_generate_batch_size=max_generate_batch_size
            )

            reference_rollout_buffer = None
            if reference_ckpt_dir is not None:
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

            verifier_rollout_buffer = collect_verifier_buffer(
                verifier_model_type=verifier_model_type,
                verifier_config_file=verifier_config_file,
                max_seq_len=max_seq_len,
                verifier_tokenizer_file=verifier_tokenizer_file,
                dtype=dtype,
                verifier_ckpt_dir=verifier_ckpt_dir,
                verifier_reference_ckpt_dir=verifier_reference_ckpt_dir,
                actor_rollout_buffer=actor_rollout_buffer,
                max_forward_batch_size=max_forward_batch_size,
                beta=beta
            )

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
                kl_coef=kl_ceof
            )

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


if __name__ == "__main__":
    fire.Fire(run)
