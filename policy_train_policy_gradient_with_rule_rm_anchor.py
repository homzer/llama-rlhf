import gc
import os
import random

import fire
import torch

from policy_train_policy_gradient import train_policy_gradient
from policy_train_policy_gradient_with_rule_rm import (
    collect_actor_buffer_with_label,
    collect_rule_based_verifier_buffer
)
from policy_train_ppo_with_evaluate import evaluate_actor
from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.ppo.buffer import PolicyRolloutBuffer, RolloutBuffer
from src.ppo.collector import ActorForwardBufferCollector
from src.ppo.parallel_buffer import ParallelRolloutBuffer
from src.utils import json_load, print_current_func_args


def preprocess_anchor_file(train_file: str, num_anchors: int):
    def select_random_data(_datalist: list) -> dict:
        for _ in range(len(_datalist)):
            _data = random.sample(_datalist, 1)[0]
            if len(_data["output"]) > 0:
                return dict(
                    instruction=_data["instruction"],
                    output=random.sample(_data["output"], 1)[0],
                    label=_data["label"]
                )
        raise RuntimeError("Exceeding max attempt for selecting random data.")

    filedata = json_load(train_file)
    datalist = []
    datalist_with_anchor = []
    for data in filedata:
        datalist.append(dict(instruction=data["instruction"], label=data["label"]))
        assert isinstance(data["output"], list)
        for i in range(0, min(len(data["output"]), num_anchors)):
            datalist_with_anchor.append(dict(
                instruction=data["instruction"], output=data["output"][i], label=data["label"]
            ))
        for i in range(0, max(num_anchors - len(data["output"]), 0)):
            datalist_with_anchor.append(select_random_data(filedata))
    assert len(datalist) * num_anchors == len(datalist_with_anchor)
    return datalist, datalist_with_anchor


def collect_actor_buffer_with_label_anchor(
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


def run(
        task: str,
        label_file: str,
        train_file: str,
        log_dir: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 36,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
        num_anchors: int = 1,
        num_neg_anchors: int = 0,
        neg_anchor_file: str = None,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
        train_strategy: str = "vanilla",
        delta: float = 0.01,
        reward_sub_mean: bool = False,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
):
    setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir

    datalist, datalist_with_anchor = preprocess_anchor_file(train_file, num_anchors)
    datalist_with_neg_anchor = []
    if neg_anchor_file is not None:
        _, datalist_with_neg_anchor = preprocess_anchor_file(neg_anchor_file, num_neg_anchors)
    chunk_size = chunk_size or len(datalist)
    local_epochs = len(datalist) // chunk_size
    begin_global_epoch = begin_epoch // local_epochs
    begin_local_epoch = begin_epoch % local_epochs
    for global_epoch in range(begin_global_epoch, epochs):
        for local_epoch in range(begin_local_epoch, local_epochs):
            epoch = local_epoch + global_epoch * local_epochs
            print(f"Epoch - {epoch} of {local_epochs * epochs}")
            dataset = JsonDataset(f=datalist[local_epoch * chunk_size: (local_epoch + 1) * chunk_size])
            dataset_with_anchor = JsonDataset(f=datalist_with_anchor[local_epoch * chunk_size * num_anchors: (local_epoch + 1) * chunk_size * num_anchors])
            dataset_with_neg_anchor = JsonDataset(f=datalist_with_neg_anchor[local_epoch * chunk_size * num_neg_anchors: (local_epoch + 1) * chunk_size * num_neg_anchors])
            if len(dataset) == 0:
                continue

            # Collecting policy buffer
            policy_rollout_buffer = collect_actor_buffer_with_label_anchor(
                actor_model_type=policy_model_type,
                actor_config_file=policy_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=policy_tokenizer_file,
                dtype=dtype,
                actor_ckpt_dir=policy_ckpt_dir,
                epoch=epoch,
                actor_save_dir=save_dir,
                use_chat_template=use_chat_template,
                dataset=dataset_with_neg_anchor,
                max_forward_batch_size=max_forward_batch_size
            )
            policy_rollout_buffer.extend(collect_actor_buffer_with_label_anchor(
                actor_model_type=policy_model_type,
                actor_config_file=policy_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=policy_tokenizer_file,
                dtype=dtype,
                actor_ckpt_dir=policy_ckpt_dir,
                epoch=epoch,
                actor_save_dir=save_dir,
                use_chat_template=use_chat_template,
                dataset=dataset_with_anchor,
                max_forward_batch_size=max_forward_batch_size
            ))

            policy_rollout_buffer.extend(collect_actor_buffer_with_label(
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
            ))

            # Saving policy rollout buffer
            ParallelRolloutBuffer(**policy_rollout_buffer).save(os.path.join(save_dir, "epoch-%03d" % epoch))

            verifier_rollout_buffer = collect_rule_based_verifier_buffer(
                policy_rollout_buffer=policy_rollout_buffer, task=task
            )

            print(f"Average Rewards: {verifier_rollout_buffer.mean(use_last_token_reward=True)}")

            rollout_buffer = PolicyRolloutBuffer(
                obs=policy_rollout_buffer["obs"],
                actions=policy_rollout_buffer["actions"],
                rewards=verifier_rollout_buffer["scores"],
                values=verifier_rollout_buffer["scores"],  # pseudo
                action_logits=policy_rollout_buffer["action_logits"],
                action_masks=policy_rollout_buffer["action_masks"],
                action_logprobs=policy_rollout_buffer["action_logprobs"],
                use_last_token_reward=True,
                reward_sub_mean=reward_sub_mean
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
                delta=delta
            )

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

