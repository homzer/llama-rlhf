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
from src.ppo.trainer import ParallelPolicyGradientTrainerForCausalLM
from src.utils import masked_mean, json_load
from src.parallel.utils import setup_model_parallel, set_barrier


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
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
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
    os.makedirs(save_dir, exist_ok=True)
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir
    verifier_config_file = verifier_config_file or verifier_ckpt_dir
    verifier_tokenizer_file = verifier_tokenizer_file or verifier_ckpt_dir

    datalist = json_load(train_file)
    chunk_size = chunk_size or len(datalist)
    epochs = len(datalist) // chunk_size
    for epoch in range(begin_epoch, epochs):
        print(f"Epoch - {epoch} of {epochs}")
        dataset = JsonDataset(f=datalist[epoch * chunk_size: (epoch + 1) * chunk_size])
        # Collecting policy buffer
        policy, policy_tokenizer = get_parallel_model(
            model_type=policy_model_type,
            config_file=policy_config_file,
            max_seq_len=max_seq_len,
            tokenizer_file=policy_tokenizer_file,
            lora_rank=-1,
            dtype=dtype
        )
        policy.load(policy_ckpt_dir if epoch == 0 else os.path.join(save_dir, f"epoch-{epoch}"))
        policy_buffer_collector = ActorBufferCollector(policy, policy_tokenizer, max_seq_len, temperature=1.0)
        policy_rollout_buffer = ActorRolloutBuffer()
        print('Policy buffer collecting ...')
        if use_chat_template:
            dataset = ChatTemplateDataset(dataset, policy_tokenizer)
        dataloader = DataLoader(dataset, batch_size=max_generate_batch_size)
        timer = Timer(len(dataloader))
        for data in dataloader:
            timer.step()
            policy_rollout_buffer.extend(policy_buffer_collector.forward(data['instruction']))
            print(data['instruction'][-1])
            print(policy_tokenizer.decode(
                policy_rollout_buffer.actions[-1][policy_rollout_buffer.action_masks[-1]].tolist()
            ))

        policy.cpu()
        del policy
        del policy_buffer_collector
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
        timer = Timer(total=len(policy_rollout_buffer) // max_forward_batch_size, episode=10)
        for data in policy_rollout_buffer.get(max_forward_batch_size):
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

        print("Average Rewards: ", masked_mean(verifier_rollout_buffer.scores, policy_rollout_buffer.action_masks))

        rollout_buffer = RolloutBuffer(
            obs=policy_rollout_buffer.obs,
            actions=policy_rollout_buffer.actions,
            rewards=verifier_rollout_buffer.scores,
            values=verifier_rollout_buffer.scores,  # pseudo
            action_logits=policy_rollout_buffer.action_logits,
            action_masks=policy_rollout_buffer.action_masks,
            action_logprobs=policy_rollout_buffer.action_logprobs
        )

        torch.save({
            'obs': rollout_buffer.obs[: max_forward_batch_size],
            'actions': rollout_buffer.actions[: max_forward_batch_size],
            'values': rollout_buffer.values[: max_forward_batch_size],
            'rewards': rollout_buffer.rewards[: max_forward_batch_size],
            'action_masks': rollout_buffer.action_masks[: max_forward_batch_size],
            'advantages': rollout_buffer.advantages[: max_forward_batch_size],
            'returns': rollout_buffer.returns[: max_forward_batch_size]
        }, os.path.join(save_dir, f"buffer-{epoch}.bin"))

        policy, policy_tokenizer = get_parallel_model(
            model_type=policy_model_type,
            config_file=policy_config_file,
            max_seq_len=max_seq_len,
            tokenizer_file=policy_tokenizer_file,
            lora_rank=lora_rank,
            dtype=dtype,
            lora_dtype=lora_dtype
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        trainer = ParallelPolicyGradientTrainerForCausalLM(policy, optimizer)
        trainer.load_model(policy_ckpt_dir) if (
                epoch == 0
        ) else trainer.load(os.path.join(save_dir, f"epoch-{epoch}"))
        print('Policy training ...')
        timer = Timer(total=(len(rollout_buffer) // max_batch_size) * inner_epochs, episode=100)
        for inner_epoch in range(inner_epochs):
            for data in rollout_buffer.get(max_batch_size):
                timer.step()
                trainer_outputs = trainer.forward(data)
                if trainer.step % 100 == 0:
                    print(f'--------- STEP {trainer.step} OF {timer.total} ---------')
                    print('Loss: ', trainer_outputs.loss)
                    print('Rewards: ', trainer_outputs.rewards)
        trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))

        policy.cpu()
        del policy
        del optimizer
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()


if __name__ == '__main__':
    fire.Fire(run)
