import gc
import os

import fire
import torch

from policy_train_policy_gradient import train_policy_gradient
from policy_train_ppo import collect_actor_buffer, collect_verifier_buffer
from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer, IterationHandler
from src.modeling import get_parallel_model
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.ppo.buffer import RolloutBuffer, CriticRolloutBuffer
from src.ppo.collector import ActorForwardBufferCollector
from src.utils import json_load, print_current_func_args


def process_pairwise_dataset(dataset: JsonDataset) -> JsonDataset:
    datalist = []
    for data in dataset:
        datalist.append(dict(instruction=data["instruction"], output=data["chosen"], label=1))
        datalist.append(dict(instruction=data["instruction"], output=data["rejected"], label=-1))
    return JsonDataset(datalist)


def collect_actor_forward_buffer_with_label(
        actor_model_type: str,
        actor_config_file: str,
        max_seq_len: int,
        actor_tokenizer_file: str,
        dtype: str,
        actor_ckpt_dir: str,
        epoch: int,
        actor_save_dir: str,
        use_chat_template: bool,
        dataset: JsonDataset,
        max_forward_batch_size: int
) -> RolloutBuffer:
    if len(dataset) == 0:
        return RolloutBuffer()
    actor, actor_tokenizer = get_parallel_model(
        model_type=actor_model_type,
        config_file=actor_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=actor_tokenizer_file,
        lora_rank=-1,
        dtype=dtype
    )
    actor.load(actor_ckpt_dir if epoch == 0 else os.path.join(actor_save_dir, "epoch-%03d" % epoch))
    actor_buffer_collector = ActorForwardBufferCollector(
        actor=actor,
        tokenizer=actor_tokenizer,
        max_seq_len=max_seq_len
    )
    actor_rollout_buffer_with_label = RolloutBuffer()
    print("Actor anchor buffer collecting ...")
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, actor_tokenizer)
    dataloader = ParallelDataLoader(dataset, batch_size=max_forward_batch_size)
    timer = Timer(len(dataloader), episode=10)
    for data in dataloader:
        timer.step()
        actor_rollout_buffer = actor_buffer_collector.forward(data["instruction"], data["output"])
        actor_rollout_buffer_with_label.extend(RolloutBuffer(**actor_rollout_buffer, labels=data['label']))

    actor.cpu()
    del actor
    del actor_buffer_collector
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    return actor_rollout_buffer_with_label


def collect_verifier_forward_buffer(policy_rollout_buffer: RolloutBuffer) -> CriticRolloutBuffer:
    verifier_rollout_buffer = CriticRolloutBuffer(
        scores=policy_rollout_buffer["labels"],
        action_masks=policy_rollout_buffer["action_masks"],
        use_last_token_reward=True,
        last_token_reward_only=False
    )
    return verifier_rollout_buffer


def run(
        train_file: str,
        label_file: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        verifier_config_file: str = None,
        verifier_tokenizer_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_generate_batch_size: int = 256,
        max_batch_size: int = 1,
        max_forward_batch_size: int = 24,
        max_seq_len: int = 1024,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        log_dir: str = None,
        seed: int = None,
        train_strategy: str = "vanilla",
        save_optim: bool = False,
        accumulation_steps: int = 1,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
):
    parallel_infos = setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir

    for epoch, datalist in IterationHandler(json_load(train_file), epochs, chunk_size, begin_epoch):
        dataset = JsonDataset(datalist)
        if len(dataset) == 0:
            continue

        policy_rollout_buffer = collect_actor_forward_buffer_with_label(
            actor_model_type=policy_model_type,
            actor_config_file=policy_config_file,
            max_seq_len=max_seq_len,
            actor_tokenizer_file=policy_tokenizer_file,
            dtype=dtype,
            actor_ckpt_dir=policy_ckpt_dir,
            epoch=epoch,
            actor_save_dir=save_dir,
            use_chat_template=use_chat_template,
            dataset=process_pairwise_dataset(dataset),
            max_forward_batch_size=max_forward_batch_size
        )

        rollout_buffer = RolloutBuffer(
            obs=policy_rollout_buffer["obs"],
            actions=policy_rollout_buffer["actions"],
            rewards=collect_verifier_forward_buffer(policy_rollout_buffer)["scores"],
            action_masks=policy_rollout_buffer["action_masks"],
            action_logprobs=policy_rollout_buffer["action_logprobs"]
        )

        train_policy_gradient(
            rollout_buffer=rollout_buffer,
            policy_ckpt_dir=policy_ckpt_dir,
            policy_model_type=policy_model_type,
            policy_config_file=policy_config_file,
            policy_tokenizer_file=policy_tokenizer_file,
            max_seq_len=max_seq_len,
            lora_rank=lora_rank,
            dtype=dtype,
            lora_dtype=lora_dtype,
            lr=lr,
            epoch=epoch,
            inner_epochs=inner_epochs,
            save_dir=save_dir,
            max_batch_size=max_batch_size,
            train_strategy=train_strategy,
            accumulation_steps=accumulation_steps,
            save_optim=save_optim
        )

        if parallel_infos.local_rank == 0:
            rollout_buffer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))

        policy_rollout_buffer = collect_actor_buffer(
            actor_model_type=policy_model_type,
            actor_config_file=policy_config_file,
            max_seq_len=max_seq_len,
            actor_tokenizer_file=policy_tokenizer_file,
            dtype=dtype,
            actor_ckpt_dir=policy_ckpt_dir,
            epoch=epoch + 1,
            actor_save_dir=save_dir,
            use_chat_template=use_chat_template,
            dataset=JsonDataset(json_load(label_file)[:max_generate_batch_size]),
            max_generate_batch_size=max_generate_batch_size,
            temperature=0.0,
        )

        verifier_rollout_buffer = collect_verifier_buffer(
            verifier_model_type=verifier_model_type,
            verifier_config_file=verifier_config_file,
            max_seq_len=max_seq_len,
            verifier_tokenizer_file=verifier_tokenizer_file,
            dtype=dtype,
            verifier_ckpt_dir=verifier_ckpt_dir,
            actor_rollout_buffer=policy_rollout_buffer,
            max_forward_batch_size=max_forward_batch_size,
            use_last_token_reward=True,
        )
        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")


if __name__ == '__main__':
    fire.Fire(run)
