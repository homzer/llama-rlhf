import gc
import os

import fire
import torch
from torch.utils.data import DataLoader
import numpy as np

from policy_gradient_train import re_scoring_eos_rewards
from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.modeling import get_parallel_model, get_parallel_verifier
from src.parallel.utils import setup_model_parallel, set_barrier
from src.ppo.buffer import CriticRolloutBuffer, RolloutBuffer, ActorRolloutBuffer
from src.ppo.collector import CriticBufferCollector, ActorBufferCollector
from src.ppo.trainer import ParallelPolicyGradientKLDivTrainerForCausalLM
from src.utils import masked_mean, json_load

REVISER_PROMPT = """###[QUESTION]\n{question}\n\n###[ANSWER]\n{rejected}\n\n###[REVISED ANSWER]\n"""


def get_reviser_dataset(origin_dataset: JsonDataset, responses: list) -> JsonDataset:
    """ format datalist for reviser """
    instructions = [data["instruction"] for data in origin_dataset]
    assert len(instructions) == len(responses)
    results = []
    for instruction, response in zip(instructions, responses):
        results.append(dict(
            instruction=REVISER_PROMPT.format_map({
                "question": instruction,
                "rejected": response
            }),
        ))
    return JsonDataset(results)


def update_policy_rollout_buffer(
        policy_buffer: ActorRolloutBuffer, reviser_buffer: ActorRolloutBuffer
) -> ActorRolloutBuffer:

    assert len(policy_buffer) == len(reviser_buffer)

    for i in range(len(policy_buffer)):
        p_begin = np.nonzero(policy_buffer.action_masks[i])[0][0]
        r_begin = np.nonzero(reviser_buffer.action_masks[i])[0][0]
        length = min(
            len(policy_buffer.action_masks[i]) - p_begin - 1,
            np.nonzero(reviser_buffer.action_masks[i])[0][-1] - r_begin + 1
        )
        p_end = p_begin + length
        r_end = r_begin + length

        policy_buffer.obs[i][p_begin: p_end] = reviser_buffer.obs[i][r_begin: r_end]
        policy_buffer.actions[i][p_begin: p_end] = reviser_buffer.actions[i][r_begin: r_end]
        policy_buffer.action_masks[i][p_begin: p_end] = reviser_buffer.action_masks[i][r_begin: r_end]
        policy_buffer.action_logits[i][p_begin: p_end] = reviser_buffer.action_logits[i][r_begin: r_end]
        policy_buffer.action_logprobs[i][p_begin: p_end] = reviser_buffer.action_logprobs[i][r_begin: r_end]
        policy_buffer.responses[i] = reviser_buffer.responses[i]

    return policy_buffer


def run(
        train_file: str,
        save_dir: str,

        policy_ckpt_dir: str,
        policy_model_type: str,
        reviser_ckpt_dir: str,
        reviser_model_type: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,

        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        reviser_config_file: str = None,
        reviser_tokenizer_file: str = None,
        verifier_config_file: str = None,
        verifier_tokenizer_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        policy_generate_batch_size: int = 384,
        reviser_generate_batch_size: int = 64,
        max_forward_batch_size: int = 36,
        policy_max_seq_len: int = 1024,
        reviser_max_seq_len: int = 1536,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False
):
    setup_model_parallel()
    os.makedirs(save_dir, exist_ok=True)
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir
    reviser_config_file = reviser_config_file or reviser_ckpt_dir
    reviser_tokenizer_file = reviser_tokenizer_file or reviser_ckpt_dir
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
            max_seq_len=policy_max_seq_len,
            tokenizer_file=policy_tokenizer_file,
            lora_rank=-1,
            dtype=dtype
        )
        policy.load(policy_ckpt_dir if epoch == 0 else os.path.join(save_dir, f"epoch-{epoch}"))
        policy_buffer_collector = ActorBufferCollector(policy, policy_tokenizer, policy_max_seq_len, temperature=1.2)
        policy_rollout_buffer = ActorRolloutBuffer()
        print('Policy buffer collecting ...')
        dataloader = DataLoader(ChatTemplateDataset(dataset, policy_tokenizer) if (
            use_chat_template
        ) else dataset, batch_size=policy_generate_batch_size)
        timer = Timer(len(dataloader))
        for data in dataloader:
            timer.step()
            policy_rollout_buffer.extend(policy_buffer_collector.forward(data['instruction']))
            print(data['instruction'][-1])
            print(policy_rollout_buffer.responses[-1])

        policy.cpu()
        del policy
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        reviser, reviser_tokenizer = get_parallel_model(
            model_type=reviser_model_type,
            config_file=reviser_config_file,
            max_seq_len=reviser_max_seq_len,
            tokenizer_file=reviser_tokenizer_file,
            lora_rank=-1,
            dtype=dtype
        )
        reviser.load(reviser_ckpt_dir)
        reviser_buffer_collector = ActorBufferCollector(reviser, reviser_tokenizer, reviser_max_seq_len, temperature=1.2)
        reviser_rollout_buffer = ActorRolloutBuffer()
        print("Reviser buffer collecting ...")
        reviser_dataset = get_reviser_dataset(dataset, policy_rollout_buffer.responses)
        reviser_dataloader = DataLoader(ChatTemplateDataset(reviser_dataset, reviser_tokenizer) if (
            use_chat_template
        ) else reviser_dataset, batch_size=reviser_generate_batch_size)
        timer = Timer(len(reviser_dataloader))
        for data in reviser_dataloader:
            timer.step()
            reviser_rollout_buffer.extend(reviser_buffer_collector.forward(data["instruction"]))
            print(data['instruction'][-1])
            print(reviser_rollout_buffer.responses[-1])
        # Replace policy's response with reviser's response
        policy_rollout_buffer = update_policy_rollout_buffer(policy_rollout_buffer, reviser_rollout_buffer)

        reviser.cpu()
        del reviser
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        verifier, verifier_tokenizer = get_parallel_verifier(
            model_type=verifier_model_type,
            config_file=verifier_config_file,
            max_seq_len=policy_max_seq_len,
            tokenizer_file=verifier_tokenizer_file,
            lora_rank=-1,
            dtype=dtype
        )
        verifier.load(verifier_ckpt_dir)
        verifier_buffer_collector = CriticBufferCollector(verifier, verifier_tokenizer, policy_max_seq_len)
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
        rollout_buffer = re_scoring_eos_rewards(rollout_buffer)

        policy, policy_tokenizer = get_parallel_model(
            model_type=policy_model_type,
            config_file=policy_config_file,
            max_seq_len=policy_max_seq_len,
            tokenizer_file=policy_tokenizer_file,
            lora_rank=lora_rank,
            dtype=dtype,
            lora_dtype=lora_dtype
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        trainer = ParallelPolicyGradientKLDivTrainerForCausalLM(policy, optimizer)
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

        torch.save({
            'obs': rollout_buffer.obs,
            'actions': rollout_buffer.actions,
            'values': rollout_buffer.values,
            'rewards': rollout_buffer.rewards,
            'action_masks': rollout_buffer.action_masks,
            'advantages': rollout_buffer.advantages,
            'returns': rollout_buffer.returns
        }, os.path.join(save_dir, f"epoch-{epoch + 1}", f"buffer.bin"))

        policy.cpu()
        del policy
        del optimizer
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()


if __name__ == '__main__':
    fire.Fire(run)
