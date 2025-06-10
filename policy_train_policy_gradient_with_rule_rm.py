import collections
import gc
import os

import fire
import torch

from policy_train_policy_gradient import train_policy_gradient
from policy_train_ppo_with_evaluate import evaluate_actor
from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.evaluator import EVALUATORS
from src.modeling import get_parallel_model
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.ppo.buffer import PolicyRolloutBuffer, CriticRolloutBuffer, RolloutBuffer
from src.ppo.collector import ActorBufferCollector
from src.utils import json_load, print_current_func_args


def collect_actor_buffer_with_label(
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
        max_generate_batch_size: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
) -> RolloutBuffer:
    dataset.repeat(num_samples_per_prompt).shuffle()

    actor, actor_tokenizer = get_parallel_model(
        model_type=actor_model_type,
        config_file=actor_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=actor_tokenizer_file,
        lora_rank=-1,
        dtype=dtype
    )
    actor.load(actor_ckpt_dir if epoch == 0 else os.path.join(actor_save_dir, "epoch-%03d" % epoch))
    actor_buffer_collector = ActorBufferCollector(
        actor=actor,
        tokenizer=actor_tokenizer,
        max_seq_len=max_seq_len,
        temperature=temperature,
        top_p=top_p
    )
    actor_rollout_buffer_with_label = RolloutBuffer()
    print('Actor buffer collecting ...')
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, actor_tokenizer)
    dataloader = ParallelDataLoader(dataset, batch_size=max_generate_batch_size)
    timer = Timer(len(dataloader))
    for data in dataloader:
        timer.step()
        actor_rollout_buffer = actor_buffer_collector.forward(data['instruction'])
        actor_rollout_buffer_with_label.extend(RolloutBuffer(
            **actor_rollout_buffer, labels=data['label']
        ))
        print(actor_rollout_buffer["instructions"][-1] + "\n" + actor_rollout_buffer["responses"][-1])

    actor.cpu()
    del actor
    del actor_buffer_collector
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    return actor_rollout_buffer_with_label


def reward_fn(policy_rollout_buffer: RolloutBuffer, task: str):
    """
    :return: acc_rewards: 1 for correct answer, -1 for incorrect answer.
    """
    evaluator = EVALUATORS[task.lower()]()
    acc_rewards = []
    think_len_rewards = []
    max_think_len = None
    min_think_len = None
    for data in policy_rollout_buffer.get(1):
        if evaluator.eval(data.responses[0], data.labels[0]) is True:
            acc_rewards.append(1.0)  # answer correct
            think_len = len(data.responses[0])
            max_think_len = think_len if max_think_len is None else max(max_think_len, think_len)
            min_think_len = think_len if min_think_len is None else min(min_think_len, think_len)
        else:
            acc_rewards.append(-1.0)  # answer incorrect

    for i, data in enumerate(policy_rollout_buffer.get(1)):
        if acc_rewards[i] == 1.0:  # answer correct
            if max_think_len is not None and min_think_len is not None and max_think_len > min_think_len:
                think_len = len(data.responses[0])
                if think_len is not None:
                    think_len_rewards.append((think_len - min_think_len) / (max_think_len - min_think_len))
                else:
                    think_len_rewards.append(0)
            else:
                think_len_rewards.append(0)
        else:
            think_len_rewards.append(0)

    assert len(acc_rewards) == len(think_len_rewards)
    Output = collections.namedtuple("Output", ["acc_rewards", "think_len_rewards"])
    return Output(acc_rewards=acc_rewards, think_len_rewards=think_len_rewards)


def collect_rule_based_verifier_buffer(
        policy_rollout_buffer: RolloutBuffer, task: str
) -> CriticRolloutBuffer:
    outputs = reward_fn(policy_rollout_buffer, task=task)
    verifier_rollout_buffer = CriticRolloutBuffer(outputs.acc_rewards, action_masks=policy_rollout_buffer["action_masks"])
    return verifier_rollout_buffer


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
        seed: int = None,
        train_strategy: str = "vanilla",
        delta: float = 0.01,
        reward_sub_mean: bool = False,
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

            # Collecting policy buffer
            policy_rollout_buffer = collect_actor_buffer_with_label(
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

            if parallel_infos.local_rank == 0:
                torch.save({
                    'obs': rollout_buffer.obs,
                    'actions': rollout_buffer.actions,
                    'rewards': rollout_buffer.origin_rewards,
                    'action_masks': rollout_buffer.action_masks
                }, os.path.join(save_dir, "epoch-%03d" % (epoch + 1), f"buffer.bin"))

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
