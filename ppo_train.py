import gc
import os

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
from src.ppo.policy import ActorCriticPolicyForCausalLM
from src.ppo.trainer import PPOTrainerForCausalLM
from src.tokenizer import LlamaTokenizer
from src.utils import setup_model_parallel, set_barrier


def run(
        solver_ckpt_dir: str,
        verifier_ckpt_dir: str,
        train_file: str,
        max_batch_size: int = 16,
        max_buffer_size: int = 32,
        max_seq_len: int = 256,
        epochs: int = 1,
        inner_epochs: int = 4,
        lr: float = 1e-5,
        model_type: str = "gpt2-base",
        tokenizer_path: str = None,
        config_file: str = None
):
    tokenizer_path = tokenizer_path if tokenizer_path else os.path.join('config', model_type)
    config_file = config_file if config_file else os.path.join('config', model_type, 'config.json')
    dataset = JsonDataset(filename=train_file)
    dataloader = DataLoader(dataset, batch_size=max_buffer_size)

    local_rank, world_size = setup_model_parallel()
    tokenizer = LlamaTokenizer(tokenizer_path)
    args = LoraLlamaArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
    ).from_json(config_file)

    for epoch in range(epochs):
        print('Policy buffer collecting ...')
        solver = LoraLlama(args)
        generator = PPOGeneratorForCausalLM(solver, tokenizer, max_seq_len)
        policy = ActorCriticPolicyForCausalLM(solver, generator, args.dim)
        policy.load()
        policy_collector = PolicyBufferCollector(policy)
        policy_rollout_buffer = PolicyRolloutBuffer()
        for data in tqdm(dataloader):
            policy_rollout_buffer.extend(policy_collector.forward(data['instruction']))

        solver.cpu()
        policy.cpu()
        del solver
        del policy
        del generator
        del policy_collector
        gc.collect()
        set_barrier()

        print('Env buffer collecting ...')
        verifier = LoraLlamaVerifier(args)
        verifier.load(verifier_ckpt_dir)
        env = LlamaRewardEnv(verifier, tokenizer, max_seq_len)
        env_collector = EnvBufferCollector(env)
        env_rollout_buffer = EnvRolloutBuffer()
        for data in policy_rollout_buffer.get(max_buffer_size):
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

        print('Policy training ...')
        solver = LoraLlama(args)
        policy = ActorCriticPolicyForCausalLM(solver, generator, args.dim)
        policy.load()
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        trainer = PPOTrainerForCausalLM(policy, optimizer)
        for inner_epoch in range(inner_epochs):
            for data in rollout_buffer.get(max_batch_size):
                outputs = trainer.forward(data)
                print('Loss: ', outputs.loss)
                print('Policy Loss: ', outputs.policy_loss)
                print('Value Loss: ', outputs.value_loss)

        solver.cpu()
        policy.cpu()
        del solver
        del policy
        del optimizer
        del trainer


if __name__ == '__main__':
    fire.Fire(run)
