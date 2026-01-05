import gc
import os

import fire
import numpy as np
import torch

from policy_train_lco_with_qrm import train_lco
from policy_train_ppo import collect_actor_buffer
from policy_train_ppo_with_evaluate import evaluate_actor
from src.dataset import JsonDataset
from src.entities import Timer, IterationHandler
from src.modeling import get_parallel_model
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.ppo.buffer import RolloutBuffer, LogitsRolloutBuffer
from src.ppo.generator import ActorLogitsGeneratorForCausalLM
from src.utils import json_load, print_current_func_args


def collect_verifier_buffer(
        policy_rollout_buffer: RolloutBuffer,
        verifier_model_type: str,
        verifier_ckpt_dir: str,
        verifier_config_file: str,
        verifier_tokenizer_file: str,
        max_seq_len: int,
        max_forward_batch_size: int,
        dtype: str,
        logits_topk: int,
) -> LogitsRolloutBuffer:
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
    verifier_rollout_buffer = LogitsRolloutBuffer()
    print("Verifier buffer collecting ...")
    timer = Timer(total=policy_rollout_buffer.size() // max_forward_batch_size, episode=10)
    for data in policy_rollout_buffer.get(max_forward_batch_size):
        timer.step()
        generator_outputs = verifier_buffer_generator.forward(data.instructions, data.responses)
        buffer = LogitsRolloutBuffer(
            instructions=data.instructions,
            obs=generator_outputs.obs.cpu().numpy(),
            actions=generator_outputs.actions.cpu().numpy(),
            action_masks=generator_outputs.action_masks.cpu().numpy(),
            action_logprobs=generator_outputs.action_logprobs.float().cpu().numpy(),
            logits=generator_outputs.logits,
            responses=data.responses,
            logits_topk=logits_topk
        )
        buffer["advantages"] = np.take_along_axis(
            torch.log_softmax(generator_outputs.logits, dim=-1).half().cpu().numpy(),
            indices=buffer["logits_indices"],
            axis=-1
        )
        buffer["advantage_indices"] = buffer["logits_indices"]
        verifier_rollout_buffer.extend(buffer)

    verifier.cpu()
    del verifier
    del verifier_buffer_generator
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    return verifier_rollout_buffer


def collect_logits_buffer(
        verifier_rollout_buffer: LogitsRolloutBuffer,
        policy_model_type: str,
        policy_ckpt_dir: str,
        policy_config_file: str,
        policy_tokenizer_file: str,
        max_seq_len: int,
        max_forward_batch_size: int,
        epoch: int,
        save_dir: str,
        dtype: str,
) -> LogitsRolloutBuffer:
    policy, policy_tokenizer = get_parallel_model(
        model_type=policy_model_type,
        config_file=policy_config_file,
        tokenizer_file=policy_tokenizer_file,
        max_seq_len=max_seq_len,
        dtype=dtype
    )
    policy.load(policy_ckpt_dir if epoch == 0 else os.path.join(save_dir, "epoch-%03d" % epoch))
    print("Logits buffer collecting ...")
    logits_buffer_generator = ActorLogitsGeneratorForCausalLM(
        model=policy,
        tokenizer=policy_tokenizer,
        max_seq_len=max_seq_len
    )
    logits_rollout_buffer = RolloutBuffer()
    timer = Timer(total=verifier_rollout_buffer.size() // max_forward_batch_size, episode=10)
    for data in verifier_rollout_buffer.get(max_forward_batch_size):
        timer.step()
        generator_outputs = logits_buffer_generator.forward(data.instructions, data.responses)
        logits_rollout_buffer.extend(RolloutBuffer(
            logits_values=np.take_along_axis(
                generator_outputs.logits.half().cpu().numpy(),
                indices=data.logits_indices,
                axis=-1
            ),
            logits_indices=data.logits_indices,
            vocab_sizes=np.full(shape=generator_outputs.logits.shape[0], fill_value=generator_outputs.logits.shape[-1])
        ))

    policy.cpu()
    del policy
    del logits_buffer_generator
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    vocab_sizes = verifier_rollout_buffer["vocab_sizes"]
    if not np.array_equal(verifier_rollout_buffer["vocab_sizes"], logits_rollout_buffer["vocab_sizes"]):
        print(f"Warming: vocab size is not equal, setting to policy's vocab size")
        print(f"Verifier vocab size: {verifier_rollout_buffer['vocab_sizes'][0]}")
        print(f"Policy vocab size: {logits_rollout_buffer['vocab_sizes'][0]}")
        vocab_sizes = logits_rollout_buffer["vocab_sizes"]

    return LogitsRolloutBuffer(
        instructions=verifier_rollout_buffer["instructions"],
        obs=verifier_rollout_buffer["obs"],
        actions=verifier_rollout_buffer["actions"],
        action_masks=verifier_rollout_buffer["action_masks"],
        action_logprobs=verifier_rollout_buffer["action_logprobs"],
        responses=verifier_rollout_buffer["responses"],
        logits_values=logits_rollout_buffer["logits_values"],
        logits_indices=logits_rollout_buffer["logits_indices"],
        vocab_sizes=vocab_sizes,
        advantages=verifier_rollout_buffer["advantages"],
        advantage_indices=verifier_rollout_buffer["advantage_indices"],
    )


def run(
        train_file: str,
        save_dir: str,
        log_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        verifier_config_file: str = None,
        verifier_tokenizer_file: str = None,
        label_file: str = None,
        task: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 24,
        max_seq_len: int = 1024,
        beta: float = 2.0,
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

        # collecting verifier buffer
        verifier_rollout_buffer = collect_verifier_buffer(
            policy_rollout_buffer=policy_rollout_buffer,
            verifier_model_type=verifier_model_type,
            verifier_ckpt_dir=verifier_ckpt_dir,
            verifier_config_file=verifier_config_file,
            verifier_tokenizer_file=verifier_tokenizer_file,
            max_seq_len=max_seq_len,
            max_forward_batch_size=max_forward_batch_size,
            dtype=dtype,
            logits_topk=logits_topk
        )

        # collecting logits buffer
        logits_rollout_buffer = collect_logits_buffer(
            verifier_rollout_buffer=verifier_rollout_buffer,
            policy_model_type=policy_model_type,
            policy_ckpt_dir=policy_ckpt_dir,
            policy_config_file=policy_config_file,
            policy_tokenizer_file=policy_tokenizer_file,
            max_seq_len=max_seq_len,
            max_forward_batch_size=max_forward_batch_size,
            epoch=epoch,
            save_dir=save_dir,
            dtype=dtype
        )

        if parallel_infos.global_rank == 0:
            logits_rollout_buffer.save(os.path.join(log_dir, "epoch-%03d" % (epoch + 1)))

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

        if label_file is not None:
            assert task is not None
            evaluate_actor(
                task=task,
                label_file=label_file,
                log_dir=log_dir,
                actor_model_type=policy_model_type,
                actor_config_file=policy_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=policy_tokenizer_file,
                dtype=dtype,
                epoch=epoch,
                actor_save_dir=save_dir,
                max_generate_batch_size=max_generate_batch_size,
                use_chat_template=use_chat_template
            )


if __name__ == '__main__':
    fire.Fire(run)
