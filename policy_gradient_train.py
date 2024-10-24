import gc
import os

import fire
import numpy as np
import torch

from ppo_train import collect_actor_buffer, collect_verifier_buffer
from src.dataset import JsonDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.parallel.utils import setup_model_parallel, set_barrier
from src.ppo.buffer import RolloutBuffer
from src.ppo.trainer import ParallelPolicyGradientTrainerForCausalLM
from src.utils import masked_mean, json_load


def re_scoring_eos_rewards(buffer: RolloutBuffer) -> RolloutBuffer:
    # Setting the reward of [EOS] token to average reward of the sequence.
    for i, action_mask in enumerate(buffer.action_masks):
        buffer.rewards[i][np.nonzero(action_mask)[0][-1]] = np.mean(buffer.rewards[i][action_mask])

    return buffer


def train_policy_gradient(
        rollout_buffer: RolloutBuffer,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_config_file: str,
        policy_tokenizer_file: str,
        max_seq_len: int,
        lora_rank: int,
        dtype: str,
        lora_dtype: str,
        lr: float,
        epoch: int,
        inner_epochs: int,
        save_dir: str,
        max_batch_size: int,
):
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
            max_generate_batch_size=max_generate_batch_size
        )

        verifier_rollout_buffer = collect_verifier_buffer(
            verifier_model_type=verifier_model_type,
            verifier_config_file=verifier_config_file,
            max_seq_len=max_seq_len,
            verifier_tokenizer_file=verifier_tokenizer_file,
            dtype=dtype,
            verifier_ckpt_dir=verifier_ckpt_dir,
            actor_rollout_buffer=policy_rollout_buffer,
            max_forward_batch_size=max_forward_batch_size
        )

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
        rollout_buffer = re_scoring_eos_rewards(rollout_buffer)

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
            max_batch_size=max_batch_size
        )

        torch.save({
            'obs': rollout_buffer.obs,
            'actions': rollout_buffer.actions,
            'values': rollout_buffer.values,
            'rewards': rollout_buffer.rewards,
            'action_masks': rollout_buffer.action_masks,
            'advantages': rollout_buffer.advantages,
            'returns': rollout_buffer.returns
        }, os.path.join(save_dir, f"epoch-{epoch + 1}", f"buffer.bin"))


if __name__ == '__main__':
    fire.Fire(run)
