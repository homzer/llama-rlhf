import gc
import os

import fire
import torch

from policy_train_ppo import collect_actor_buffer, train_actor
from policy_train_ppo_with_evaluate import evaluate_actor
from src.dataset import JsonDataset
from src.entities import Timer, IterationHandler
from src.modeling import get_parallel_model
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.ppo.buffer import RolloutBuffer, CriticRolloutBuffer
from src.ppo.collector import ActorForwardBufferCollector
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
) -> CriticRolloutBuffer:
    verifier, verifier_tokenizer = get_parallel_model(
        model_type=verifier_model_type,
        config_file=verifier_config_file,
        tokenizer_file=verifier_tokenizer_file,
        max_seq_len=max_seq_len,
        dtype=dtype
    )
    verifier.load(verifier_ckpt_dir)
    verifier_buffer_collector = ActorForwardBufferCollector(
        actor=verifier,
        tokenizer=verifier_tokenizer,
        max_seq_len=max_seq_len
    )
    verifier_rollout_buffer = CriticRolloutBuffer()
    timer = Timer(total=policy_rollout_buffer.size() // max_forward_batch_size, episode=10)
    for data in policy_rollout_buffer.get(max_forward_batch_size):
        timer.step()
        rollout_buffer = verifier_buffer_collector.forward(data.instructions, data.responses)
        verifier_rollout_buffer.extend(CriticRolloutBuffer(
            scores=rollout_buffer["action_logprobs"],
            action_masks=rollout_buffer["action_masks"]
        ))

    verifier.cpu()
    del verifier
    del verifier_buffer_collector
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    return verifier_rollout_buffer


def run(
        train_file: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        verifier_config_file: str = None,
        verifier_tokenizer_file: str = None,
        task: str = None,
        label_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 24,
        max_seq_len: int = 1024,
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
        log_dir: str = None,
        seed: int = None,
        clip_range: float = 0.2,
        save_optim: bool = False,
        accumulation_steps: int = 1,
        max_num_ckpts: int = None,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
):
    parallel_infos = setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        log_mode='w' if begin_epoch == 0 else 'a',
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
            max_seq_len=max_seq_len,
            actor_tokenizer_file=policy_tokenizer_file,
            dtype=dtype,
            actor_ckpt_dir=policy_ckpt_dir,
            epoch=epoch,
            actor_save_dir=save_dir,
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
            dtype=dtype
        )
        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")
        verifier_rollout_buffer.normalize()

        rollout_buffer = RolloutBuffer(
            obs=policy_rollout_buffer["obs"],
            actions=policy_rollout_buffer["actions"],
            action_masks=policy_rollout_buffer["action_masks"],
            action_logprobs=policy_rollout_buffer["action_logprobs"],
            advantages=verifier_rollout_buffer["scores"]
        )

        train_actor(
            rollout_buffer=rollout_buffer,
            actor_model_type=policy_model_type,
            actor_config_file=policy_config_file,
            actor_tokenizer_file=policy_tokenizer_file,
            actor_ckpt_dir=policy_ckpt_dir,
            actor_save_dir=save_dir,
            max_seq_len=max_seq_len,
            actor_lora_rank=lora_rank,
            actor_lora_dtype=lora_dtype,
            dtype=dtype,
            lr=lr,
            epoch=epoch,
            actor_max_batch_size=max_batch_size,
            inner_epochs=inner_epochs,
            clip_range=clip_range,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps,
            max_num_ckpts=max_num_ckpts
        )

        if parallel_infos.global_rank == 0:
            rollout_buffer.save(os.path.join(log_dir, "epoch-%03d" % (epoch + 1)))

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
