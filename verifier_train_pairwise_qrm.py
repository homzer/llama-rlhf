import gc
import os
import shutil

import fire
import torch

from src.dataset import PairwiseDataset, ChatTemplateDataset
from src.entities import Timer, IterationHandler
from src.modeling import get_parallel_model
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.initialize import setup_model_parallel, set_barrier, get_rank
from src.ppo.buffer import RolloutBuffer
from src.ppo.collector import ActorForwardBufferCollector
from src.rewards.trainer import ParallelVerifierTrainerForQRM
from src.utils import print_current_func_args, json_load


def collect_policy_pairwise_forward_buffer(
        dataset: PairwiseDataset,
        policy_model_type: str,
        policy_config_file: str,
        max_seq_len: int,
        policy_tokenizer_file: str,
        dtype: str,
        policy_ckpt_dir: str,
        epoch: int,
        policy_save_dir: str,
        use_chat_template: bool,
        max_forward_batch_size: int
) -> RolloutBuffer:
    if len(dataset) == 0:
        return RolloutBuffer()
    policy, policy_tokenizer = get_parallel_model(
        model_type=policy_model_type,
        config_file=policy_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=policy_tokenizer_file,
        lora_rank=-1,
        dtype=dtype
    )
    policy.load(policy_ckpt_dir if epoch == 0 else os.path.join(policy_save_dir, "epoch-%03d" % epoch))
    policy_buffer_collector = ActorForwardBufferCollector(
        actor=policy,
        tokenizer=policy_tokenizer,
        max_seq_len=max_seq_len
    )
    policy_rollout_buffer = RolloutBuffer()
    print("Policy forward buffer collecting ...")
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, policy_tokenizer)
    dataloader = ParallelDataLoader(dataset, batch_size=max_forward_batch_size)
    timer = Timer(len(dataloader), episode=10)
    for data in dataloader:
        timer.step()
        chosen_buffer = policy_buffer_collector.forward(instructions=data["instruction"], responses=data["chosen"])
        rejected_buffer = policy_buffer_collector.forward(instructions=data["instruction"], responses=data["rejected"])
        policy_rollout_buffer.extend(RolloutBuffer(
            chosen_obs=chosen_buffer["obs"],
            rejected_obs=rejected_buffer["obs"],
            chosen_actions=chosen_buffer["actions"],
            rejected_actions=rejected_buffer["actions"],
            chosen_action_masks=chosen_buffer["action_masks"],
            rejected_action_masks=rejected_buffer["action_masks"],
        ))

    policy.cpu()
    del policy
    del policy_buffer_collector
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    return policy_rollout_buffer


def main(
        log_dir: str,
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        model_type: str,
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        max_forward_batch_size: int = 32,
        lr: float = 1e-5,
        epochs: int = 1,
        begin_epoch: int = 0,
        chunk_size: int = None,
        lora_rank: int = -1,
        tokenizer_file: str = None,
        config_file: str = None,
        dtype: str = "bfloat16",
        lora_dtype: str = "float32",
        use_chat_template: bool = False,
        seed: int = None,
        save_optim: bool = False,
        accumulation_steps: int = 1,
        max_num_ckpts: int = None,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
):
    tokenizer_file = tokenizer_file or ckpt_dir
    config_file = config_file or ckpt_dir
    setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        log_mode='w' if begin_epoch == 0 else "a",
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()

    iterator = IterationHandler(json_load(train_file), epochs, chunk_size, begin_epoch)
    for epoch, datalist in iterator:
        dataset = PairwiseDataset(datalist)
        if len(dataset) == 0:
            continue

        policy_rollout_buffer = collect_policy_pairwise_forward_buffer(
            dataset=dataset,
            policy_model_type=model_type,
            policy_config_file=config_file,
            max_seq_len=max_seq_len,
            policy_tokenizer_file=tokenizer_file,
            dtype=dtype,
            policy_ckpt_dir=ckpt_dir,
            epoch=epoch,
            policy_save_dir=save_dir,
            use_chat_template=use_chat_template,
            max_forward_batch_size=max_forward_batch_size
        )

        # verifier training
        verifier, verifier_tokenizer = get_parallel_model(
            model_type=model_type,
            config_file=config_file,
            max_seq_len=max_seq_len,
            tokenizer_file=tokenizer_file,
            lora_rank=lora_rank,
            dtype=dtype,
            lora_dtype=lora_dtype
        )
        optimizer = torch.optim.Adam(verifier.parameters(), lr=lr)
        trainer = ParallelVerifierTrainerForQRM(
            model=verifier,
            optimizer=optimizer,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )
        verifier.load(ckpt_dir) if (
                epoch == 0
        ) else trainer.load(os.path.join(save_dir, "epoch-%03d" % epoch))
        timer = Timer(policy_rollout_buffer.size() // max_batch_size, episode=10)
        for data in policy_rollout_buffer.get(max_batch_size, shuffle=True, output_tensor=True):
            timer.step()
            trainer_outputs = trainer.forward(data)
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {timer.total} ---------------')
                print(f'LOSS: {trainer_outputs.loss.item()} | ACC: {trainer.verifier_accuracy()}')
        trainer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))
        if max_num_ckpts is not None and (epoch + 1 - max_num_ckpts) > 0:
            rm_dir = os.path.join(save_dir, "epoch-%03d" % (epoch + 1 - max_num_ckpts))
            if get_rank() == 0 and os.path.exists(rm_dir):
                shutil.rmtree(rm_dir)

        verifier.cpu()
        del verifier
        del optimizer
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()


if __name__ == '__main__':
    fire.Fire(main)
