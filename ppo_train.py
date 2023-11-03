import gc

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import JsonDataset
from src.modeling.llama_lora import LoraLlamaVerifier, LoraLlama
from src.modeling.modeling_args import LoraLlamaArgs
from src.ppo.buffer import PolicyRolloutBuffer, EnvRolloutBuffer, RolloutBuffer
from src.ppo.collector import PolicyBufferCollector, EnvBufferCollector
from src.ppo.env import LlamaRewardEnv
from src.ppo.generator import PPOGeneratorForCausalLM
from src.ppo.policy import ParallelActorCriticPolicyForCausalLM
from src.ppo.trainer import ParallelPPOTrainerForCausalLM
from src.tokenizer import LlamaTokenizer
from src.utils import setup_model_parallel, set_barrier, Timer


def convert_solver_ckpt_to_policy_ckpt(solver_ckpt_dir: str, config_file: str, policy_save_dir: str):
    local_rank, world_size = setup_model_parallel()
    args = LoraLlamaArgs(
        max_seq_len=128,
        local_rank=local_rank,
        world_size=world_size,
        r=16
    ).from_json(config_file)
    solver = LoraLlama(args)
    solver.load(solver_ckpt_dir)
    policy = ParallelActorCriticPolicyForCausalLM(solver, None, args.dim)
    policy.save(policy_save_dir)


def run(
        policy_ckpt_dir: str,
        policy_config_file: str,
        verifier_ckpt_dir: str,
        verifier_config_file: str,
        save_dir: str,
        train_file: str,
        lora_rank: int = 16,
        max_batch_size: int = 4,
        max_buffer_size: int = 96,
        max_seq_len: int = 512,
        epochs: int = 1,
        inner_epochs: int = 2,
        lr: float = 1e-5,
        tokenizer_path: str = None,
):
    tokenizer_path = tokenizer_path if tokenizer_path else 'config/tokenizer.model'
    dataset = JsonDataset(filename=train_file)
    dataloader = DataLoader(dataset, batch_size=max_buffer_size)

    local_rank, world_size = setup_model_parallel()
    tokenizer = LlamaTokenizer(tokenizer_path)
    policy_args = LoraLlamaArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=lora_rank
    ).from_json(policy_config_file)
    verifier_args = LoraLlamaArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
        r=lora_rank
    ).from_json(verifier_config_file)

    for epoch in range(epochs):
        solver = LoraLlama(policy_args)
        generator = PPOGeneratorForCausalLM(solver, tokenizer, max_seq_len)
        policy = ParallelActorCriticPolicyForCausalLM(solver, generator, policy_args.dim)
        policy.load(policy_ckpt_dir if epoch == 0 else save_dir)
        policy_collector = PolicyBufferCollector(policy)
        policy_rollout_buffer = PolicyRolloutBuffer()
        print('Policy buffer collecting ...')
        timer = Timer(len(dataloader))
        for data in tqdm(dataloader):
            timer.step()
            buffer = policy_collector.forward(data['instruction'])
            policy_rollout_buffer.extend(buffer)

        # TODO test
        print('test2', tokenizer.decode(policy_rollout_buffer.actions[0][policy_rollout_buffer.action_masks[0]].tolist()))

        solver.cpu()
        policy.cpu()
        del solver
        del policy
        del generator
        del policy_collector
        gc.collect()
        set_barrier()

        verifier = LoraLlamaVerifier(verifier_args)
        verifier.load(verifier_ckpt_dir)
        env = LlamaRewardEnv(verifier, tokenizer, max_seq_len)
        env_collector = EnvBufferCollector(env)
        env_rollout_buffer = EnvRolloutBuffer()
        print('Environment buffer collecting ...')
        timer = Timer(len(policy_rollout_buffer) // max_buffer_size)
        for data in policy_rollout_buffer.get(max_buffer_size):
            timer.step()
            env_rollout_buffer.extend(env_collector.forward(data))

        verifier.cpu()
        del verifier
        del env
        del env_collector
        gc.collect()
        set_barrier()

        rollout_buffer = RolloutBuffer(
            obs=policy_rollout_buffer.obs,
            actions=policy_rollout_buffer.actions,
            rewards=env_rollout_buffer.rewards,
            values=policy_rollout_buffer.values,
            action_logits=policy_rollout_buffer.action_logits,
            action_masks=policy_rollout_buffer.action_masks
        )

        solver = LoraLlama(policy_args)
        generator = PPOGeneratorForCausalLM(solver, tokenizer, max_seq_len)
        policy = ParallelActorCriticPolicyForCausalLM(solver, generator, policy_args.dim)
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        trainer = ParallelPPOTrainerForCausalLM(policy, optimizer)
        trainer.load(policy_ckpt_dir if epoch == 0 else save_dir)
        print('Policy training ...')
        for inner_epoch in range(inner_epochs):
            for data in rollout_buffer.get(max_batch_size):
                outputs = trainer.forward(data)
                if trainer.step % 100 == 0:
                    print(f'---------------------- STEP {trainer.step} -----------------------')
                    print('Loss: ', outputs.loss)
                    print('Policy Loss: ', outputs.policy_loss)
                    print('Value Loss: ', outputs.value_loss)
        trainer.save(save_dir)

        solver.cpu()
        policy.cpu()
        del solver
        del policy
        del generator
        del optimizer
        del trainer
        gc.collect()
        set_barrier()


if __name__ == '__main__':
    fire.Fire(run)
