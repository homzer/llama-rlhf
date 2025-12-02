import gc
import os

import fire
import numpy as np
import torch

from policy_train_lco import collect_logits_buffer, train_lco
from policy_train_ppo import collect_actor_buffer
from src.dataset import JsonDataset
from src.entities import Timer, IterationHandler
from src.modeling import get_parallel_model
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.ppo.buffer import RolloutBuffer, LogitsRolloutBuffer
from src.ppo.generator import ActorLogitsGeneratorForCausalLM
from src.utils import json_load, print_current_func_args, normalize


def collect_verifier_buffer(
        logits_rollout_buffer: LogitsRolloutBuffer,
        verifier_model_type: str,
        verifier_ckpt_dir: str,
        verifier_config_file: str,
        verifier_tokenizer_file: str,
        reference_ckpt_dir: str,
        max_seq_len: int,
        max_forward_batch_size: int,
        dtype: str,
) -> RolloutBuffer:
    verifier, verifier_tokenizer = get_parallel_model(
        model_type=verifier_model_type,
        config_file=verifier_config_file,
        tokenizer_file=verifier_tokenizer_file,
        max_seq_len=max_seq_len,
        dtype=dtype
    )
    verifier.load(verifier_ckpt_dir)
    verifier_buffer_generator = ActorLogitsGeneratorForCausalLM(
        model=verifier,
        tokenizer=verifier_tokenizer,
        max_seq_len=max_seq_len
    )
    verifier_rollout_buffer = RolloutBuffer()
    timer = Timer(total=logits_rollout_buffer.size() // max_forward_batch_size, episode=10)
    for data in logits_rollout_buffer.get(max_forward_batch_size):
        timer.step()
        generator_outputs = verifier_buffer_generator.forward(data.instructions, data.responses)
        verifier_rollout_buffer.extend(RolloutBuffer(
            advantages=np.take_along_axis(
                torch.log_softmax(generator_outputs.logits, dim=-1).cpu().numpy(),
                indices=data["logits_indices"],
                axis=-1
            ),
            advantage_indices=data["logits_indices"],
            actions=data["actions"],
            action_masks=data["action_masks"]
        ))

    verifier.cpu()
    del verifier
    del verifier_buffer_generator
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    reference, reference_tokenizer = get_parallel_model(
        model_type=verifier_model_type,
        config_file=verifier_config_file,
        tokenizer_file=verifier_tokenizer_file,
        max_seq_len=max_seq_len,
        dtype=dtype
    )
    reference.load(reference_ckpt_dir)
    reference_buffer_generator = ActorLogitsGeneratorForCausalLM(
        model=reference,
        tokenizer=reference_tokenizer,
        max_seq_len=max_seq_len
    )
    reference_rollout_buffer = RolloutBuffer()
    timer = Timer(total=logits_rollout_buffer.size() // max_forward_batch_size, episode=10)
    for data in logits_rollout_buffer.get(max_forward_batch_size):
        timer.step()
        generator_outputs = reference_buffer_generator.forward(data.instructions, data.responses)
        reference_rollout_buffer.extend(RolloutBuffer(
            advantages=np.take_along_axis(
                torch.log_softmax(generator_outputs.logits, dim=-1).cpu().numpy(),
                indices=data["logits_indices"],
                axis=-1
            ),
        ))

    verifier_rollout_buffer["advantages"] -= reference_rollout_buffer["advantages"]
    # advantage normalization
    verifier_rollout_buffer["advantages"] = normalize(verifier_rollout_buffer["advantages"], dim=-1)
    return verifier_rollout_buffer


def run(
        train_file: str,
        save_dir: str,
        log_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,
        reference_ckpt_dir: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        verifier_config_file: str = None,
        verifier_tokenizer_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 24,
        max_seq_len: int = 1024,
        beta: float = 10.0,
        logits_topk: int = 5,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
        save_optim: bool = False,
        accumulation_steps: int = 1,
        max_num_ckpts: int = None,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
):
    parallel_infos = setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        log_mode="w" if begin_epoch == 0 else "a",
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir
    verifier_config_file = verifier_config_file or verifier_ckpt_dir
    verifier_tokenizer_file = verifier_tokenizer_file or verifier_ckpt_dir

    for epoch, datalist in IterationHandler(json_load(train_file), epochs, chunk_size, begin_epoch):
        dataset = JsonDataset(datalist)
        if len(dataset) == 0:
            continue

        # collecting policy buffer
        policy_rollout_buffer = collect_actor_buffer(
            actor_model_type=policy_model_type,
            actor_config_file=policy_config_file,
            actor_tokenizer_file=policy_tokenizer_file,
            actor_ckpt_dir=policy_ckpt_dir,
            actor_save_dir=save_dir,
            max_seq_len=max_seq_len,
            dtype=dtype,
            epoch=epoch,
            use_chat_template=use_chat_template,
            dataset=dataset,
            max_generate_batch_size=max_generate_batch_size,
            temperature=temperature,
            top_p=top_p,
            num_samples_per_prompt=num_samples_per_prompt
        )

        # collecting logits buffer
        logits_rollout_buffer = collect_logits_buffer(
            policy_rollout_buffer=policy_rollout_buffer,
            policy_model_type=policy_model_type,
            policy_ckpt_dir=policy_ckpt_dir,
            policy_config_file=policy_config_file,
            policy_tokenizer_file=policy_tokenizer_file,
            max_seq_len=max_seq_len,
            max_forward_batch_size=max_forward_batch_size,
            dtype=dtype,
            logits_topk=logits_topk
        )

        # collecting verifier buffer
        verifier_rollout_buffer = collect_verifier_buffer(
            logits_rollout_buffer=logits_rollout_buffer,
            verifier_model_type=verifier_model_type,
            verifier_ckpt_dir=verifier_ckpt_dir,
            verifier_config_file=verifier_config_file,
            verifier_tokenizer_file=verifier_tokenizer_file,
            reference_ckpt_dir=reference_ckpt_dir,
            max_seq_len=max_seq_len,
            max_forward_batch_size=max_forward_batch_size,
            dtype=dtype
        )

        logits_rollout_buffer["advantages"] = verifier_rollout_buffer["advantages"]
        logits_rollout_buffer["advantage_indices"] = verifier_rollout_buffer["advantage_indices"]

        train_lco(
            rollout_buffer=logits_rollout_buffer,
            policy_ckpt_dir=policy_ckpt_dir,
            policy_model_type=policy_model_type,
            policy_config_file=policy_config_file,
            policy_tokenizer_file=policy_tokenizer_file,
            max_seq_len=max_seq_len,
            dtype=dtype,
            lora_rank=lora_rank,
            lora_dtype=lora_dtype,
            lr=lr,
            epoch=epoch,
            inner_epochs=inner_epochs,
            save_dir=save_dir,
            max_batch_size=max_batch_size,
            beta=beta,
            max_num_ckpts=max_num_ckpts,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )

        if parallel_infos.global_rank == 0:
            logits_rollout_buffer.save(os.path.join(log_dir, "epoch-%03d" % (epoch + 1)))


if __name__ == '__main__':
    fire.Fire(run)
