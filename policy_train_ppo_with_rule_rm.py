import collections
import gc
import os

import fire
import torch

from policy_train_ppo import (
    collect_reference_buffer,
    collect_critic_buffer,
    train_actor,
    train_critic
)
from policy_train_ppo_with_evaluate import evaluate_actor
from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import IterationHandler, Timer
from src.evaluator import EVALUATORS
from src.modeling import get_parallel_model
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.ppo.buffer import PPORolloutBuffer, RolloutBuffer, CriticRolloutBuffer
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


def reward_fn(policy_rollout_buffer: RolloutBuffer, task: str, correct_reward=1.0, error_reward=-1.0):
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
            acc_rewards.append(correct_reward)  # answer correct
            think_len = len(data.responses[0])
            max_think_len = think_len if max_think_len is None else max(max_think_len, think_len)
            min_think_len = think_len if min_think_len is None else min(min_think_len, think_len)
        else:
            acc_rewards.append(error_reward)  # answer incorrect

    for i, data in enumerate(policy_rollout_buffer.get(1)):
        if acc_rewards[i] == correct_reward:  # answer correct
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
        actor_rollout_buffer: RolloutBuffer,
        task: str,
        correct_reward=1.0,
        error_reward=-1.0,
        last_token_reward_only: bool = False
) -> CriticRolloutBuffer:
    outputs = reward_fn(actor_rollout_buffer, task=task, correct_reward=correct_reward, error_reward=error_reward)
    verifier_rollout_buffer = CriticRolloutBuffer(
        scores=outputs.acc_rewards,
        action_masks=actor_rollout_buffer["action_masks"],
        use_last_token_reward=True,
        last_token_reward_only=last_token_reward_only
    )
    return verifier_rollout_buffer


def run(
        task: str,
        train_file: str,
        log_dir: str,
        actor_ckpt_dir: str,
        actor_model_type: str,
        actor_save_dir: str,
        critic_ckpt_dir: str,
        critic_model_type: str,
        critic_save_dir: str,
        label_file: str = None,
        actor_config_file: str = None,
        actor_tokenizer_file: str = None,
        critic_config_file: str = None,
        critic_tokenizer_file: str = None,
        reference_ckpt_dir: str = None,
        actor_lora_rank: int = -1,
        actor_lora_dtype: str = "bfloat16",
        critic_lora_rank: int = -1,
        critic_lora_dtype: str = "bfloat16",
        actor_max_batch_size: int = 1,
        actor_lr: float = 1e-6,
        critic_max_batch_size: int = 1,
        critic_lr: float = 1e-5,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 24,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        kl_coef: float = 0.01,
        vf_coef: float = 1.0,
        clip_range: float = 0.2,
        gamma: float = 0.9,
        gae_lambda: float = 0.95,
        use_chat_template: bool = False,
        save_optim: bool = False,
        accumulation_steps: int = 1,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
        seed: int = None,
):
    parallel_infos = setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        log_mode="w" if begin_epoch == 0 else "a",
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()
    actor_config_file = actor_config_file or actor_ckpt_dir
    actor_tokenizer_file = actor_tokenizer_file or actor_ckpt_dir
    critic_config_file = critic_config_file or critic_ckpt_dir
    critic_tokenizer_file = critic_tokenizer_file or critic_ckpt_dir

    for epoch, datalist in IterationHandler(json_load(train_file), epochs, chunk_size, begin_epoch):
        dataset = JsonDataset(datalist)
        if len(dataset) == 0:
            continue

        # Collecting actor buffer
        actor_rollout_buffer = collect_actor_buffer_with_label(
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

        verifier_rollout_buffer = collect_rule_based_verifier_buffer(
            actor_rollout_buffer=actor_rollout_buffer, task=task
        )

        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")

        reference_rollout_buffer = None
        if reference_ckpt_dir is not None and kl_coef != 0:
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

        rollout_buffer = PPORolloutBuffer(
            obs=actor_rollout_buffer["obs"],
            actions=actor_rollout_buffer["actions"],
            rewards=verifier_rollout_buffer["scores"],
            values=critic_rollout_buffer["scores"],
            action_logits=actor_rollout_buffer["action_logits"],
            action_masks=actor_rollout_buffer["action_masks"],
            action_logprobs=actor_rollout_buffer["action_logprobs"],
            ref_action_logprobs=reference_rollout_buffer.output_tokens_logps if (
                    reference_rollout_buffer is not None
            ) else None,
            use_last_token_reward=True,
            last_token_reward_only=False,
            kl_coef=kl_coef,
            vf_coef=vf_coef,
            gamma=gamma,
            gae_lambda=gae_lambda,
            reward_normalize=False
        )

        train_actor(
            actor_model_type=actor_model_type,
            actor_config_file=actor_config_file,
            max_seq_len=max_seq_len,
            actor_tokenizer_file=actor_tokenizer_file,
            actor_lora_rank=actor_lora_rank,
            dtype=dtype,
            actor_lora_dtype=actor_lora_dtype,
            lr=0.01 * actor_lr if epoch <= 1 else actor_lr,
            epoch=epoch,
            actor_ckpt_dir=actor_ckpt_dir,
            actor_save_dir=actor_save_dir,
            rollout_buffer=rollout_buffer,
            actor_max_batch_size=actor_max_batch_size,
            inner_epochs=inner_epochs,
            clip_range=clip_range,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )

        if parallel_infos.local_rank == 0:
            torch.save({
                'obs': rollout_buffer.obs,
                'actions': rollout_buffer.actions,
                'values': rollout_buffer.origin_values,
                'rewards': rollout_buffer.origin_rewards,
                'action_masks': rollout_buffer.action_masks,
                'advantages': rollout_buffer.advantages,
                'returns': rollout_buffer.returns
            }, os.path.join(actor_save_dir, "epoch-%03d" % (epoch + 1), "buffer.bin"))

        if label_file is not None:
            evaluate_actor(
                task=task,
                label_file=label_file,
                log_dir=log_dir,
                actor_model_type=actor_model_type,
                actor_config_file=actor_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=actor_tokenizer_file,
                dtype=dtype,
                epoch=epoch,
                actor_save_dir=actor_save_dir,
                max_generate_batch_size=max_generate_batch_size,
                use_chat_template=use_chat_template
            )

        train_critic(
            critic_model_type=critic_model_type,
            critic_config_file=critic_config_file,
            max_seq_len=max_seq_len,
            critic_tokenizer_file=critic_tokenizer_file,
            critic_lora_rank=critic_lora_rank,
            dtype=dtype,
            lr=critic_lr,
            critic_lora_dtype=critic_lora_dtype,
            critic_ckpt_dir=critic_ckpt_dir,
            epoch=0 if epoch == 0 else 1,  # TODO For saving memory
            critic_save_dir=critic_save_dir,
            rollout_buffer=rollout_buffer,
            critic_max_batch_size=critic_max_batch_size,
            inner_epochs=inner_epochs
        )


if __name__ == '__main__':
    fire.Fire(run)
