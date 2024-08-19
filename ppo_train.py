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
from src.utils import masked_mean, json_load
from src.parallel.utils import setup_model_parallel, set_barrier


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
        actor_lora_rank: int = -1,
        actor_lora_dtype: str = "bfloat16",
        critic_lora_rank: int = -1,
        critic_lora_dtype: str = "bfloat16",
        actor_max_batch_size: int = 1,
        critic_max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_forward_batch_size: int = 24,
        max_seq_len: int = 1024,
        chunk_size: int = None,
        inner_epochs: int = 3,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False
):
    setup_model_parallel()
    os.makedirs(actor_save_dir, exist_ok=True)
    os.makedirs(critic_save_dir, exist_ok=True)
    actor_config_file = actor_config_file or actor_ckpt_dir
    actor_tokenizer_file = actor_tokenizer_file or actor_ckpt_dir
    critic_config_file = critic_config_file or critic_ckpt_dir
    critic_tokenizer_file = critic_tokenizer_file or critic_ckpt_dir
    verifier_config_file = verifier_config_file or verifier_ckpt_dir
    verifier_tokenizer_file = verifier_tokenizer_file or verifier_ckpt_dir

    datalist = json_load(train_file)
    chunk_size = chunk_size or len(datalist)
    epochs = len(datalist) // chunk_size
    for epoch in range(begin_epoch, epochs):
        print(f"Epoch - {epoch} of {epochs}")
        dataset = JsonDataset(f=datalist[epoch * chunk_size: (epoch + 1) * chunk_size])
        # Collecting actor buffer
        actor, actor_tokenizer = get_parallel_model(
            model_type=actor_model_type,
            config_file=actor_config_file,
            max_seq_len=max_seq_len,
            tokenizer_file=actor_tokenizer_file,
            lora_rank=-1,
            dtype=dtype
        )
        actor.load(actor_ckpt_dir if epoch == 0 else os.path.join(actor_save_dir, f"epoch-{epoch}"))
        actor_buffer_collector = ActorBufferCollector(actor, actor_tokenizer, max_seq_len, temperature=1.0)
        actor_rollout_buffer = ActorRolloutBuffer()
        print('Actor buffer collecting ...')
        if use_chat_template:
            dataset = ChatTemplateDataset(dataset, actor_tokenizer)
        dataloader = DataLoader(dataset, batch_size=max_generate_batch_size)
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
            max_seq_len=max_seq_len,
            tokenizer_file=critic_tokenizer_file,
            lora_rank=-1,
            dtype=dtype
        )
        critic.load(critic_ckpt_dir if epoch == 0 else os.path.join(critic_save_dir, f"epoch-{epoch}"))
        critic_buffer_collector = CriticBufferCollector(critic, critic_tokenizer, max_seq_len)
        critic_rollout_buffer = CriticRolloutBuffer()
        print('Critic buffer collecting ...')
        timer = Timer(total=len(actor_rollout_buffer) // max_forward_batch_size, episode=10)
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
        timer = Timer(total=len(actor_rollout_buffer) // max_forward_batch_size, episode=10)
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

        print("Average Rewards: ", masked_mean(verifier_rollout_buffer.scores, actor_rollout_buffer.action_masks))

        rollout_buffer = RolloutBuffer(
            obs=actor_rollout_buffer.obs,
            actions=actor_rollout_buffer.actions,
            rewards=verifier_rollout_buffer.scores,
            values=critic_rollout_buffer.scores,
            action_logits=actor_rollout_buffer.action_logits,
            action_masks=actor_rollout_buffer.action_masks,
            action_logprobs=actor_rollout_buffer.action_logprobs
        )

        torch.save({
            'obs': rollout_buffer.obs[: max_forward_batch_size],
            'actions': rollout_buffer.actions[: max_forward_batch_size],
            'values': rollout_buffer.values[: max_forward_batch_size],
            'rewards': rollout_buffer.rewards[: max_forward_batch_size],
            'action_masks': rollout_buffer.action_masks[: max_forward_batch_size],
            'advantages': rollout_buffer.advantages[: max_forward_batch_size],
            'returns': rollout_buffer.returns[: max_forward_batch_size]
        }, os.path.join(actor_save_dir, f"buffer-{epoch}.bin"))

        actor, actor_tokenizer = get_parallel_model(
            model_type=actor_model_type,
            config_file=actor_config_file,
            max_seq_len=max_seq_len,
            tokenizer_file=actor_tokenizer_file,
            lora_rank=actor_lora_rank,
            dtype=dtype,
            lora_dtype=actor_lora_dtype
        )
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.075 * lr if epoch <= 1 else lr)
        actor_trainer = ParallelActorTrainerForCausalLM(actor, actor_optimizer)
        actor_trainer.load_model(actor_ckpt_dir) if (
                epoch == 0
        ) else actor_trainer.load(os.path.join(actor_save_dir, f"epoch-{epoch}"))
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
            max_seq_len=max_seq_len,
            tokenizer_file=critic_tokenizer_file,
            lora_rank=critic_lora_rank,
            dtype=dtype,
            lora_dtype=critic_lora_dtype,
        )
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
        critic_trainer = ParallelCriticTrainerForCausalLM(critic, critic_optimizer)
        if epoch == 0:
            critic_trainer.load_model(critic_ckpt_dir)
            assert hasattr(critic, 'v_head')
            critic.v_head = torch.nn.Linear(  # random re-init value function head
                in_features=critic.v_head.weight.shape[1],
                out_features=critic.v_head.weight.shape[0],
                bias=False,
                device=critic.v_head.weight.device
            ).type(critic.v_head.weight.dtype)
        else:
            critic_trainer.load(os.path.join(critic_save_dir, f"epoch-{epoch}"))
        print('Critic training ...')
        timer = Timer(total=(len(rollout_buffer) // critic_max_batch_size) * inner_epochs, episode=100)
        for inner_epoch in range(inner_epochs):
            for data in rollout_buffer.get(critic_max_batch_size):
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
