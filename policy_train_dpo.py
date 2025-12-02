import gc
import os
import shutil

import fire
import torch

from policy_train_ppo_with_evaluate import evaluate_actor
from src.dataset import PairwiseDataset, ChatTemplateDataset
from src.entities import Timer, IterationHandler
from src.modeling import get_parallel_model
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.initialize import (
    setup_model_parallel,
    set_barrier,
    get_data_parallel_src_rank,
    get_data_parallel_rank,
    get_rank
)
from src.parallel.optimizer import ParallelOptimizer
from src.ppo.buffer import RolloutBuffer
from src.ppo.collector import ActorForwardBufferCollector
from src.ppo.trainer import ParallelDPOTrainerForCausalLM
from src.utils import json_load, print_current_func_args


def collect_reference_buffer(
        dataset: PairwiseDataset,
        reference_ckpt_dir: str,
        reference_model_type: str,
        reference_config_file: str,
        reference_tokenizer_file: str,
        max_forward_batch_size: int,
        max_seq_len: int,
        dtype: str,
        use_chat_template: bool,
        log_dir: str,
        local_epoch: int,
        reuse_buffer: bool = False
) -> RolloutBuffer:
    print("Reference buffer collecting ...")
    ref_buffer_save_dir = os.path.join(log_dir, "epoch-%03d" % local_epoch, str(get_data_parallel_rank()))
    if reuse_buffer and os.path.exists(os.path.join(ref_buffer_save_dir, "buffer.jsonl")):
        ref_rollout_buffer = RolloutBuffer.load(ref_buffer_save_dir)
    else:
        reference, reference_tokenizer = get_parallel_model(
            model_type=reference_model_type,
            config_file=reference_config_file,
            tokenizer_file=reference_tokenizer_file,
            max_seq_len=max_seq_len,
            dtype=dtype
        )
        if use_chat_template:
            dataset = ChatTemplateDataset(dataset, reference_tokenizer)
        dataloader = ParallelDataLoader(dataset, batch_size=max_forward_batch_size)
        reference.load(reference_ckpt_dir)
        ref_rollout_buffer = RolloutBuffer()
        ref_buffer_collector = ActorForwardBufferCollector(
            actor=reference,
            tokenizer=reference_tokenizer,
            max_seq_len=max_seq_len
        )
        timer = Timer(len(dataloader), episode=10)
        for data in dataloader:
            timer.step()
            chosen_buffer = ref_buffer_collector.forward(instructions=data["instruction"], responses=data["chosen"])
            rejected_buffer = ref_buffer_collector.forward(instructions=data["instruction"], responses=data["rejected"])
            ref_rollout_buffer.extend(RolloutBuffer(
                chosen_obs=chosen_buffer["obs"],
                rejected_obs=rejected_buffer["obs"],
                chosen_actions=chosen_buffer["actions"],
                rejected_actions=rejected_buffer["actions"],
                chosen_action_masks=chosen_buffer["action_masks"],
                rejected_action_masks=rejected_buffer["action_masks"],
                ref_chosen_logprobs=chosen_buffer["action_logprobs"],
                ref_rejected_logprobs=rejected_buffer["action_logprobs"],
            ))

        if get_data_parallel_src_rank() == 0:
            ref_rollout_buffer.save(ref_buffer_save_dir)

        reference.cpu()
        del reference
        del ref_buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

    return ref_rollout_buffer


def train_dpo(
        reference_rollout_buffer: RolloutBuffer,
        policy_model_type: str,
        policy_ckpt_dir: str,
        policy_config_file: str,
        policy_tokenizer_file: str,
        save_dir: str,
        max_seq_len: int,
        max_batch_size: int,
        dtype: str,
        lora_rank: int,
        lora_dtype: str,
        beta: float,
        lr: float,
        epoch: int,
        save_optim: bool = False,
        accumulation_steps: int = 1,
        max_num_ckpts: int = None,
):
    policy, policy_tokenizer = get_parallel_model(
        model_type=policy_model_type,
        config_file=policy_config_file,
        tokenizer_file=policy_tokenizer_file,
        max_seq_len=max_seq_len,
        dtype=dtype,
        lora_rank=lora_rank,
        lora_dtype=lora_dtype
    )
    optimizer = ParallelOptimizer(torch.optim.Adam(policy.parameters(), lr=lr))
    trainer = ParallelDPOTrainerForCausalLM(
        policy=policy,
        optimizer=optimizer,
        beta=beta,
        save_optim=save_optim,
        accumulation_steps=accumulation_steps,
    )
    policy.load(policy_ckpt_dir) if (
            epoch == 0
    ) else trainer.load(os.path.join(save_dir, "epoch-%03d" % epoch))
    print("DPO policy training ...")
    timer = Timer(reference_rollout_buffer.size() // max_batch_size, episode=100)
    for data in reference_rollout_buffer.get(max_batch_size, shuffle=True, output_tensor=True):
        timer.step()
        trainer_outputs = trainer.forward(data)
        if trainer.step % 100 == 0:
            print(f'--------- STEP {trainer.step} OF {timer.total} ---------')
            print(f'Loss: {trainer_outputs.loss}')
    trainer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))
    if max_num_ckpts is not None and (epoch + 1 - max_num_ckpts) > 0:
        rm_dir = os.path.join(save_dir, "epoch-%03d" % (epoch + 1 - max_num_ckpts))
        if get_rank() == 0 and os.path.exists(rm_dir):
            shutil.rmtree(rm_dir)

    policy.cpu()
    del policy
    del optimizer
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()


def run(
        task: str,
        log_dir: str,
        train_file: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_tokenizer_file: str,
        policy_config_file: str,
        label_file: str = None,
        max_seq_len: int = 1024,
        max_batch_size: int = 1,
        max_generate_batch_size: int = 1,
        max_forward_batch_size: int = 1,
        lr: float = 1e-6,
        dtype: str = "bfloat16",
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        chunk_size: int = None,
        beta: float = 0.1,
        epochs: int = 1,
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
        seed: int = None,
        reuse_buffer: bool = False,
        save_optim: bool = False,
        accumulation_steps: int = 1,
        max_num_ckpts: int = None,
):
    setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        log_mode="w" if begin_epoch == 0 else "a",
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir

    iterator = IterationHandler(json_load(train_file), epochs, chunk_size, begin_epoch)
    for epoch, datalist in iterator:
        dataset = PairwiseDataset(datalist)
        if len(dataset) == 0:
            continue

        # Reference model logprobs collecting ...
        reference_rollout_buffer = collect_reference_buffer(
            dataset=dataset,
            reference_ckpt_dir=policy_ckpt_dir,
            reference_model_type=policy_model_type,
            reference_config_file=policy_config_file,
            reference_tokenizer_file=policy_tokenizer_file,
            max_forward_batch_size=max_forward_batch_size,
            max_seq_len=max_seq_len,
            dtype=dtype,
            use_chat_template=use_chat_template,
            log_dir=log_dir,
            local_epoch=iterator.local_epoch,
            reuse_buffer=reuse_buffer
        )

        # policy DPO training ...
        train_dpo(
            reference_rollout_buffer=reference_rollout_buffer,
            policy_model_type=policy_model_type,
            policy_ckpt_dir=policy_ckpt_dir,
            policy_config_file=policy_config_file,
            policy_tokenizer_file=policy_tokenizer_file,
            save_dir=save_dir,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            dtype=dtype,
            lora_rank=lora_rank,
            lora_dtype=lora_dtype,
            beta=beta,
            lr=lr,
            epoch=epoch,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps,
            max_num_ckpts=max_num_ckpts
        )

        if label_file is not None:
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
