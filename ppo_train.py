import gc
import os

import fire
import torch
from torch.utils.data import DataLoader

from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.modeling import get_parallel_model, get_parallel_verifier
from src.ppo.buffer import CriticRolloutBuffer, RolloutBuffer, ActorRolloutBuffer
from src.ppo.collector import CriticBufferCollector, ActorBufferCollector
from src.ppo.trainer import ParallelActorTrainerForCausalLM, ParallelCriticTrainerForCausalLM
from src.utils import setup_model_parallel, set_barrier, masked_mean


def run(
        train_file: str,
        save_dir: str,

        actor_ckpt_dir: str,
        actor_model_type: str,
        critic_ckpt_dir: str,
        critic_model_type: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,

        actor_config_file: str = None,
        actor_tokenizer_file: str = None,
        critic_config_file: str = None,
        critic_tokenizer_file: str = None,
        verifier_config_file: str = None,
        verifier_tokenizer_file: str = None,
        lora_rank: int = 16,
        max_batch_size: int = 1,
        max_eval_batch_size: int = 12,
        max_seq_len: int = 4096,
        epochs: int = 1,
        inner_epochs: int = 2,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        lora_dtype: str = "float32",
        use_chat_template: bool = False
):
    local_rank, world_size = setup_model_parallel()
    actor_save_dir = os.path.join(save_dir, "actor")
    critic_save_dir = os.path.join(save_dir, "critic")
    os.makedirs(actor_save_dir, exist_ok=True)
    os.makedirs(critic_save_dir, exist_ok=True)
    actor_config_file = actor_config_file if actor_config_file else actor_ckpt_dir
    actor_tokenizer_file = actor_tokenizer_file if actor_tokenizer_file else actor_ckpt_dir
    critic_config_file = critic_config_file if critic_config_file else critic_ckpt_dir
    critic_tokenizer_file = critic_tokenizer_file if critic_tokenizer_file else critic_ckpt_dir
    verifier_config_file = verifier_config_file if verifier_config_file else verifier_ckpt_dir
    verifier_tokenizer_file = verifier_tokenizer_file if verifier_tokenizer_file else verifier_ckpt_dir

    dataset = JsonDataset(f=train_file)
    for epoch in range(epochs):
        # Collecting actor buffer
        actor, actor_tokenizer = get_parallel_model(
            model_type=actor_model_type,
            config_file=actor_config_file,
            local_rank=local_rank,
            world_size=world_size,
            max_seq_len=max_seq_len,
            tokenizer_file=actor_tokenizer_file,
            lora_rank=-1,
            dtype=dtype
        )
        actor.load(actor_ckpt_dir if epoch == 0 else os.path.join(actor_save_dir, f"epoch-{epoch}"))
        actor_buffer_collector = ActorBufferCollector(actor, actor_tokenizer, max_seq_len)
        actor_rollout_buffer = ActorRolloutBuffer()
        print('Actor buffer collecting ...')
        if use_chat_template:
            dataset = ChatTemplateDataset(dataset, actor_tokenizer)
        dataloader = DataLoader(dataset, batch_size=max_eval_batch_size)
        timer = Timer(len(dataloader))
        for data in dataloader:
            timer.step()
            actor_rollout_buffer.extend(actor_buffer_collector.forward(data['instruction']))
            print(data['instruction'][-1])
            print(actor_tokenizer.decode(
                actor_rollout_buffer.actions[-1][actor_rollout_buffer.action_masks[-1]].tolist()
            ))

        actor.cpu()
        del actor
        del actor_buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        critic, critic_tokenizer = get_parallel_verifier(
            model_type=critic_model_type,
            config_file=critic_config_file,
            local_rank=local_rank,
            world_size=world_size,
            max_seq_len=max_seq_len,
            tokenizer_file=critic_tokenizer_file,
            lora_rank=-1,
            dtype=dtype
        )
        critic.load(critic_ckpt_dir if epoch == 0 else os.path.join(critic_save_dir, f"epoch-{epoch}"))
        critic_buffer_collector = CriticBufferCollector(critic, critic_tokenizer, max_seq_len)
        critic_rollout_buffer = CriticRolloutBuffer()
        print('Critic buffer collecting ...')
        for data in actor_rollout_buffer.get(max_eval_batch_size):
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

        verifier, verifier_tokenizer = get_parallel_verifier(
            model_type=verifier_model_type,
            config_file=verifier_config_file,
            local_rank=local_rank,
            world_size=world_size,
            max_seq_len=max_seq_len,
            tokenizer_file=verifier_tokenizer_file,
            lora_rank=-1,
            dtype=dtype
        )
        verifier.load(verifier_ckpt_dir)
        verifier_buffer_collector = CriticBufferCollector(verifier, verifier_tokenizer, max_seq_len)
        verifier_rollout_buffer = CriticRolloutBuffer()
        print('Reward buffer collecting ...')
        for data in actor_rollout_buffer.get(max_eval_batch_size):
            verifier_rollout_buffer.extend(
                verifier_buffer_collector.forward(
                    data.instructions, data.actions, data.action_masks
                )
            )
        print("Average Rewards: ", masked_mean(
            verifier_rollout_buffer.scores, verifier_rollout_buffer.scores != 0
        ).mean())

        verifier.cpu()
        del verifier
        del verifier_buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        rollout_buffer = RolloutBuffer(
            obs=actor_rollout_buffer.obs,
            actions=actor_rollout_buffer.actions,
            rewards=verifier_rollout_buffer.scores,
            values=critic_rollout_buffer.scores,
            action_logits=actor_rollout_buffer.action_logits,
            action_masks=actor_rollout_buffer.action_masks
        )

        torch.save({
            'obs': rollout_buffer.obs[: max_eval_batch_size],
            'actions': rollout_buffer.actions[: max_eval_batch_size],
            'values': rollout_buffer.values[: max_eval_batch_size],
            'rewards': rollout_buffer.rewards[: max_eval_batch_size],
            'action_masks': rollout_buffer.action_masks[: max_eval_batch_size],
            'advantages': rollout_buffer.advantages[: max_eval_batch_size],
            'returns': rollout_buffer.returns[: max_eval_batch_size]
        }, os.path.join(save_dir, f"buffer-{epoch}.bin"))

        actor, actor_tokenizer = get_parallel_model(
            model_type=actor_model_type,
            config_file=actor_config_file,
            local_rank=local_rank,
            world_size=world_size,
            max_seq_len=max_seq_len,
            tokenizer_file=actor_tokenizer_file,
            lora_rank=lora_rank,
            dtype=dtype,
            lora_dtype=lora_dtype
        )
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.1 * lr if epoch == 0 else lr)
        actor_trainer = ParallelActorTrainerForCausalLM(actor, actor_optimizer)
        actor_trainer.load_model(actor_ckpt_dir) if (
                epoch == 0
        ) else actor_trainer.load(os.path.join(actor_save_dir, f"epoch-{epoch}"))
        print('Actor training ...')
        for inner_epoch in range(inner_epochs):
            timer = Timer(total=len(rollout_buffer) // max_batch_size, episode=10)
            for data in rollout_buffer.get(max_batch_size):
                timer.step()
                trainer_outputs = actor_trainer.forward(data)
                if actor_trainer.step % 100 == 0:
                    print(f'--------- STEP {actor_trainer.step} OF {timer.total} ---------')
                    print('Loss: ', trainer_outputs.loss)
        actor_trainer.save(os.path.join(actor_save_dir, f"epoch-{epoch + 1}"))

        actor.cpu()
        del actor
        del actor_optimizer
        del actor_trainer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        critic, critic_tokenizer = get_parallel_verifier(
            model_type=critic_model_type,
            config_file=critic_config_file,
            local_rank=local_rank,
            world_size=world_size,
            max_seq_len=max_seq_len,
            tokenizer_file=critic_tokenizer_file,
            lora_rank=lora_rank,
            dtype=dtype,
            lora_dtype=lora_dtype,
        )
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
        critic_trainer = ParallelCriticTrainerForCausalLM(critic, critic_optimizer)
        critic_trainer.load_model(critic_ckpt_dir) if (
                epoch == 0
        ) else critic_trainer.load(os.path.join(critic_save_dir, f"epoch-{epoch}"))
        print('Critic training ...')
        for inner_epoch in range(inner_epochs):
            timer = Timer(total=len(rollout_buffer) // max_batch_size, episode=10)
            for data in rollout_buffer.get(max_batch_size):
                timer.step()
                trainer_outputs = critic_trainer.forward(data)
                if critic_trainer.step % 100 == 0:
                    print(f'--------- STEP {critic_trainer.step} OF {timer.total} ---------')
                    print('Loss: ', trainer_outputs.loss)
        critic_trainer.save(os.path.join(critic_save_dir, f"epoch-{epoch + 1}"))

        critic.cpu()
        del critic
        del critic_optimizer
        del critic_trainer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()


if __name__ == '__main__':
    fire.Fire(run)
