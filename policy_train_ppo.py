import gc
import os

import fire
import torch

from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.modeling import get_parallel_model, get_parallel_verifier
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.parallel.optimizer import ParallelOptimizer
from src.ppo.buffer import (
    CriticRolloutBuffer,
    PolicyRolloutBuffer,
    LogitsRolloutBuffer,
    RolloutBuffer
)
from src.ppo.collector import CriticBufferCollector, LogitsBufferCollector, ActorBufferCollector
from src.ppo.trainer import ParallelActorTrainerForCausalLM, ParallelCriticTrainerForCausalLM
from src.utils import json_load


def random_init_v_head(critic):
    assert hasattr(critic, 'v_head')
    critic.v_head = torch.nn.Linear(  # random re-init value function head
        in_features=critic.v_head.weight.shape[1],
        out_features=critic.v_head.weight.shape[0],
        bias=False,
        device=critic.v_head.weight.device
    ).type(critic.v_head.weight.dtype)


def collect_actor_buffer(
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
        top_p=top_p,
    )
    actor_rollout_buffer = RolloutBuffer()
    print('Actor buffer collecting ...')
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, actor_tokenizer)
    dataloader = ParallelDataLoader(dataset, batch_size=max_generate_batch_size)
    timer = Timer(len(dataloader))
    for data in dataloader:
        timer.step()
        actor_rollout_buffer.extend(actor_buffer_collector.forward(data['instruction']))
        print(actor_rollout_buffer["instructions"][-1] + "\n" + actor_rollout_buffer["responses"][-1])

    actor.cpu()
    del actor
    del actor_buffer_collector
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    return actor_rollout_buffer


def collect_reference_buffer(
        actor_model_type: str,
        actor_config_file: str,
        max_seq_len: int,
        actor_tokenizer_file: str,
        dtype: str,
        reference_ckpt_dir: str,
        actor_rollout_buffer: RolloutBuffer,
        max_forward_batch_size: int,

) -> LogitsRolloutBuffer:
    reference, reference_tokenizer = get_parallel_model(
        model_type=actor_model_type,
        config_file=actor_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=actor_tokenizer_file,
        lora_rank=-1,
        dtype=dtype
    )
    reference.load(reference_ckpt_dir)
    reference_buffer_collector = LogitsBufferCollector(
        model=reference, tokenizer=reference_tokenizer, max_seq_len=max_seq_len
    )
    reference_rollout_buffer = LogitsRolloutBuffer()
    print('Reference buffer collecting ...')
    timer = Timer(total=actor_rollout_buffer.size() // max_forward_batch_size, episode=10)
    for data in actor_rollout_buffer.get(max_forward_batch_size):
        timer.step()
        reference_rollout_buffer.extend(
            reference_buffer_collector.forward(data.instructions, data.responses)
        )

    reference.cpu()
    del reference
    del reference_buffer_collector
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    return reference_rollout_buffer


def collect_critic_buffer(
        critic_model_type: str,
        critic_config_file: str,
        max_seq_len: int,
        critic_tokenizer_file: str,
        dtype: str,
        critic_ckpt_dir: str,
        epoch: int,
        critic_save_dir: str,
        actor_rollout_buffer: RolloutBuffer,
        max_forward_batch_size: int,
) -> CriticRolloutBuffer:
    epoch = 0 if epoch == 0 else 1  # TODO: for saving memory
    critic, critic_tokenizer = get_parallel_verifier(
        model_type=critic_model_type,
        config_file=critic_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=critic_tokenizer_file,
        lora_rank=-1,
        dtype=dtype
    )
    critic.load(critic_ckpt_dir if epoch == 0 else os.path.join(critic_save_dir, "epoch-%03d" % epoch))
    if epoch == 0:  # random initialize value head
        random_init_v_head(critic)
    critic_buffer_collector = CriticBufferCollector(critic, critic_tokenizer, max_seq_len)
    critic_rollout_buffer = CriticRolloutBuffer()
    print('Critic buffer collecting ...')
    timer = Timer(total=actor_rollout_buffer.size() // max_forward_batch_size, episode=10)
    for data in actor_rollout_buffer.get(max_forward_batch_size):
        timer.step()
        critic_rollout_buffer.extend(
            critic_buffer_collector.forward(
                data.instructions, data.actions, data.action_masks
            )
        )

    critic.cpu()
    del critic
    del critic_buffer_collector
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    return critic_rollout_buffer


def collect_verifier_buffer(
        verifier_model_type: str,
        verifier_config_file: str,
        max_seq_len: int,
        verifier_tokenizer_file: str,
        dtype: str,
        verifier_ckpt_dir: str,
        actor_rollout_buffer: RolloutBuffer,
        max_forward_batch_size: int,
) -> CriticRolloutBuffer:
    verifier, verifier_tokenizer = get_parallel_verifier(
        model_type=verifier_model_type,
        config_file=verifier_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=verifier_tokenizer_file,
        lora_rank=-1,
        dtype=dtype
    )
    verifier.load(verifier_ckpt_dir)
    verifier_buffer_collector = CriticBufferCollector(verifier, verifier_tokenizer, max_seq_len)
    verifier_rollout_buffer = CriticRolloutBuffer()
    print('Reward buffer collecting ...')
    timer = Timer(total=actor_rollout_buffer.size() // max_forward_batch_size, episode=10)
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


def train_actor(
        actor_model_type: str,
        actor_config_file: str,
        max_seq_len: int,
        actor_tokenizer_file: str,
        actor_lora_rank: int,
        dtype: str,
        actor_lora_dtype: str,
        lr: float,
        epoch: int,
        actor_ckpt_dir: str,
        actor_save_dir: str,
        rollout_buffer: PolicyRolloutBuffer,
        actor_max_batch_size: int,
        inner_epochs: int,
        clip_range: float = 0.2
):
    actor, actor_tokenizer = get_parallel_model(
        model_type=actor_model_type,
        config_file=actor_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=actor_tokenizer_file,
        lora_rank=actor_lora_rank,
        dtype=dtype,
        lora_dtype=actor_lora_dtype
    )
    actor_optimizer = ParallelOptimizer(torch.optim.Adam(actor.parameters(), lr=0.075 * lr if epoch <= 1 else lr))
    actor_trainer = ParallelActorTrainerForCausalLM(actor, actor_optimizer, clip_range=clip_range)
    actor_trainer.load_model(actor_ckpt_dir) if (
            epoch == 0
    ) else actor_trainer.load(os.path.join(actor_save_dir, "epoch-%03d" % epoch))
    print('Actor training ...')
    timer = Timer(total=(len(rollout_buffer) // actor_max_batch_size) * inner_epochs, episode=100)
    for inner_epoch in range(inner_epochs):
        for data in rollout_buffer.get(actor_max_batch_size):
            timer.step()
            trainer_outputs = actor_trainer.forward(data)
            if actor_trainer.step % 100 == 0:
                print(f'--------- STEP {actor_trainer.step} OF {timer.total} ---------')
                print('Loss: ', trainer_outputs.loss)
                print('Advantages: ', trainer_outputs.advantages)
                print('KL Divergence: ', trainer_outputs.kl)
    actor_trainer.save(os.path.join(actor_save_dir, "epoch-%03d" % (epoch + 1)))

    actor.cpu()
    del actor
    del actor_optimizer
    del actor_trainer
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()


def train_critic(
        critic_model_type: str,
        critic_config_file: str,
        max_seq_len: int,
        critic_tokenizer_file: str,
        critic_lora_rank: int,
        dtype: str,
        lr: float,
        critic_lora_dtype: str,
        critic_ckpt_dir: str,
        epoch: int,
        critic_save_dir: str,
        rollout_buffer: PolicyRolloutBuffer,
        critic_max_batch_size: int,
        inner_epochs: int,
):
    epoch = 0 if epoch == 0 else 1  # TODO For saving memory
    critic, critic_tokenizer = get_parallel_verifier(
        model_type=critic_model_type,
        config_file=critic_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=critic_tokenizer_file,
        lora_rank=critic_lora_rank,
        dtype=dtype,
        lora_dtype=critic_lora_dtype,
    )
    critic_optimizer = ParallelOptimizer(torch.optim.Adam(critic.parameters(), lr=max(1e-5, lr)))
    critic_trainer = ParallelCriticTrainerForCausalLM(critic, critic_optimizer)
    critic_trainer.load_model(critic_ckpt_dir if epoch == 0 else os.path.join(critic_save_dir, "epoch-%03d" % epoch))
    if epoch == 0:
        random_init_v_head(critic)
    print('Critic training ...')
    timer = Timer(total=(len(rollout_buffer) // critic_max_batch_size) * inner_epochs, episode=100)
    for inner_epoch in range(inner_epochs):
        for data in rollout_buffer.get(critic_max_batch_size):
            timer.step()
            trainer_outputs = critic_trainer.forward(data)
            if critic_trainer.step % 100 == 0:
                print(f'--------- STEP {critic_trainer.step} OF {timer.total} ---------')
                print(f'Loss: {trainer_outputs.loss}')
    if epoch == 0:  # TODO For saving memory
        critic_trainer.save(os.path.join(critic_save_dir, "epoch-%03d" % (epoch + 1)))
    else:
        critic_trainer.save(os.path.join(critic_save_dir, "epoch-%03d" % epoch))

    critic.cpu()
    del critic
    del critic_optimizer
    del critic_trainer
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()


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
        gamma: float = 0.9,
        gae_lambda: float = 0.8,
        use_chat_template: bool = False,
        use_last_token_reward: bool = False,
        reward_is_q: bool = False
):
    parallel_infos = setup_model_parallel()
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

            print(f"Average Rewards: {verifier_rollout_buffer.mean(use_last_token_reward=use_last_token_reward)}")

            rollout_buffer = PolicyRolloutBuffer(
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
                use_last_token_reward=use_last_token_reward,
                last_token_reward_only=use_last_token_reward,
                reward_is_q=reward_is_q,  # Reward is Q-Value
                kl_coef=kl_coef,
                gamma=gamma,
                gae_lambda=gae_lambda
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
