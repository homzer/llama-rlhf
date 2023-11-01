import os

import fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import JsonDataset
from src.modeling.gpt2 import GPT2
from src.modeling.llama_lora import LoraLlamaVerifier, LoraLlama
from src.modeling.modeling_args import GPT2Args, LoraLlamaArgs
from src.ppo.buffer import PolicyRolloutBuffer, EnvRolloutBuffer, RolloutBuffer
from src.ppo.collector import BufferCollector, PolicyBufferCollector, EnvBufferCollector
from src.ppo.env import LlamaRewardEnv
from src.ppo.generator import PPOGeneratorForCausalLM
from src.ppo.policy import ActorCriticPolicyForCausalLM
from src.ppo.trainer import PPOTrainerForCausalLM
from src.tokenizer import GPT2Tokenizer, LlamaTokenizer
from src.utils import setup_model_parallel


def run(
        solver_ckpt_dir: str,
        verifier_ckpt_dir: str,
        train_file: str,
        max_batch_size: int = 16,
        max_buffer_size: int = 32,
        max_seq_len: int = 256,
        epochs: int = 1,
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

    # solver = LoraLlama(args)
    # solver.load(solver_ckpt_dir)
    # generator = PPOGeneratorForCausalLM(solver, tokenizer, max_seq_len)
    # policy = ActorCriticPolicyForCausalLM(solver, generator, args.n_embd)
    # verifier = LoraLlamaVerifier(args)
    # verifier.load(verifier_ckpt_dir)
    # env = LlamaRewardEnv(verifier, tokenizer, max_seq_len)
    # collector = BufferCollector(env, policy, max_buffer_size, max_seq_len)
    # optimizer = torch.optim.Adam(solver.parameters(), lr=lr)
    # trainer = PPOTrainerForCausalLM(policy, optimizer, max_batch_size)

    for epoch in range(epochs):
        solver = LoraLlama(args)
        solver.load(solver_ckpt_dir)
        generator = PPOGeneratorForCausalLM(solver, tokenizer, max_seq_len)
        policy = ActorCriticPolicyForCausalLM(solver, generator, args.dim)

        verifier = LoraLlamaVerifier(args)
        verifier.load(verifier_ckpt_dir)
        env = LlamaRewardEnv(verifier, tokenizer, max_seq_len)

        print('Policy buffer collecting ...')
        policy_collector = PolicyBufferCollector(policy)
        policy_rollout_buffer = PolicyRolloutBuffer()
        for data in tqdm(dataloader):
            policy_rollout_buffer.extend(policy_collector.forward(data['instruction']))

        print('Env buffer collecting ...')
        env_collector = EnvBufferCollector(env)
        env_rollout_buffer = EnvRolloutBuffer()
        for data in policy_rollout_buffer.get(max_buffer_size):
            env_rollout_buffer.extend(env_collector.forward(data))

        rollout_buffer = RolloutBuffer(
            obs=policy_rollout_buffer.obs,
            actions=policy_rollout_buffer.actions,
            rewards=env_rollout_buffer.rewards,
            values=policy_rollout_buffer.values,
            action_logits=policy_rollout_buffer.action_logits,
            action_masks=policy_rollout_buffer.action_masks
        )

        print('Policy training ...')
        for data in rollout_buffer.get(max_batch_size):

        for data in tqdm(dataloader):
            outputs = trainer.forward(rollout_buffer)
            print('Loss: ', outputs.loss)
            print('Policy Loss: ', outputs.policy_loss)
            print('Value Loss: ', outputs.value_loss)


if __name__ == '__main__':
    fire.Fire(run)
