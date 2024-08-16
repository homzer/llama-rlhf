import gc
import os

import fire
import torch
from torch.utils.data import DataLoader

from src.dataset import MultiOutputsDataset, JsonDataset
from src.entities import Timer
from src.evaluator import SolverEvaluator
from src.modeling import get_parallel_model
from src.ppo.buffer import LogitsRolloutBuffer, OutputRolloutBuffer
from src.ppo.collector import LogitsBufferCollector, OutputBufferCollector
from src.trainer import ParallelSolverDpoTrainer
from src.utils import json_dump, json_load
from src.parallel import setup_model_parallel, set_barrier


def main(
        train_file: str,

        policy_ckpt_dir: str,
        policy_save_dir: str,
        policy_model_type: str,
        policy_lora_rank: int,
        policy_tokenizer_file: str,
        policy_config_file: str,

        reference_ckpt_dir: str,
        reference_model_type: str,
        reference_tokenizer_file: str,
        reference_config_file: str,
        reference_forward_batch_size: int,

        policy_max_seq_len: int = 1024,
        reference_max_seq_len: int = 1024,
        task: str = None,
        label_file: str = None,
        eval_batch_size: int = 256,
        max_batch_size: int = 1,
        lr: float = 1e-5,
        dtype: str = "float16",
        lora_dtype: str = "float32",
        log_dir: str = None,
        seed: int = None,
        chunk_size: int = 10000,
        begin_epoch: int = 0
):
    if task is not None:
        assert label_file is not None
        assert eval_batch_size is not None
        assert log_dir is not None
        os.makedirs(log_dir, exist_ok=True)
    setup_model_parallel(seed=seed)
    datalist = json_load(train_file)
    epochs = len(datalist) // chunk_size

    for epoch in range(begin_epoch, epochs):
        print(f"Epoch - {epoch} of {epochs}")
        dataset = MultiOutputsDataset(datalist[epoch * chunk_size: (epoch + 1) * chunk_size])
        if len(dataset) == 0:
            return
        dataloader = DataLoader(dataset, batch_size=eval_batch_size)

        # Collect policy's outputs as rejected samples
        policy, policy_tokenizer = get_parallel_model(
            model_type=policy_model_type,
            config_file=policy_config_file,
            max_seq_len=policy_max_seq_len,
            tokenizer_file=policy_tokenizer_file,
            lora_rank=-1,
            dtype=dtype,
            lora_dtype=lora_dtype
        )
        policy.load(policy_ckpt_dir if (
                epoch == 0
        ) else os.path.join(policy_save_dir, f"epoch-{epoch}"), merge_lora=True)
        rejected_buffer_collector = OutputBufferCollector(policy, policy_tokenizer, policy_max_seq_len, temperature=1.0)
        rejected_rollout_buffer = OutputRolloutBuffer()
        chosen_rollout_buffer = OutputRolloutBuffer()
        timer = Timer(len(dataloader))
        for data in dataloader:
            timer.step()
            rejected_rollout_buffer.extend(rejected_buffer_collector.forward(data['instruction']))
            chosen_rollout_buffer.extend(OutputRolloutBuffer(data['instruction'], data['output']))
        assert len(rejected_rollout_buffer) == len(chosen_rollout_buffer)
        policy.cpu()
        del policy
        del rejected_buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        # Reference model logits collecting ...
        reference, reference_tokenizer = get_parallel_model(
            model_type=reference_model_type,
            config_file=reference_config_file,
            max_seq_len=reference_max_seq_len,
            tokenizer_file=reference_tokenizer_file,
            lora_rank=-1,
            dtype=dtype,
            lora_dtype=lora_dtype
        )
        reference.load(reference_ckpt_dir, merge_lora=True)
        reference_buffer_collector = LogitsBufferCollector(reference, reference_tokenizer, reference_max_seq_len)
        reference_rejected_rollout_buffer = LogitsRolloutBuffer()
        reference_chosen_rollout_buffer = LogitsRolloutBuffer()
        timer = Timer(len(chosen_rollout_buffer) // reference_forward_batch_size, episode=10)
        for chosen_data, rejected_data in zip(
                chosen_rollout_buffer.get(reference_forward_batch_size),
                rejected_rollout_buffer.get(reference_forward_batch_size)
        ):
            timer.step()
            reference_chosen_rollout_buffer.extend(
                reference_buffer_collector.forward(
                    chosen_data.instructions, chosen_data.outputs
                )
            )
            reference_rejected_rollout_buffer.extend(
                reference_buffer_collector.forward(
                    rejected_data.instructions, rejected_data.outputs
                )
            )
        assert len(reference_chosen_rollout_buffer) == len(reference_rejected_rollout_buffer)

        reference.cpu()
        del reference
        del reference_buffer_collector
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        policy, policy_tokenizer = get_parallel_model(
            model_type=policy_model_type,
            config_file=policy_config_file,
            max_seq_len=policy_max_seq_len,
            tokenizer_file=policy_tokenizer_file,
            lora_rank=policy_lora_rank,
            dtype=dtype,
            lora_dtype=lora_dtype
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        trainer = ParallelSolverDpoTrainer(
            model=policy,
            tokenizer=policy_tokenizer,
            optimizer=optimizer,
            max_seq_len=policy_max_seq_len
        )
        policy.load(policy_ckpt_dir, merge_lora=True) if (
                epoch == 0
        ) else trainer.load(os.path.join(policy_save_dir, f"epoch-{epoch}"))
        timer = Timer(len(chosen_rollout_buffer) // max_batch_size, episode=100)
        assert len(chosen_rollout_buffer) == len(reference_chosen_rollout_buffer)
        for chosen_data, rejected_data, reference_chosen_data, reference_rejected_data in zip(
            chosen_rollout_buffer.get(max_batch_size),
            rejected_rollout_buffer.get(max_batch_size),
            reference_chosen_rollout_buffer.get(max_batch_size),
            reference_rejected_rollout_buffer.get(max_batch_size)
        ):
            timer.step()
            trainer_outputs = trainer.dpo_forward(
                instructions=chosen_data.instructions,
                chosen=chosen_data.outputs,
                rejected=rejected_data.outputs,
                reference_chosen_logits=reference_chosen_data.logits,
                reference_rejected_logits=reference_rejected_data.logits
            )
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {len(chosen_rollout_buffer) // max_batch_size} ---------------')
                print(f'CE LOSS: ', trainer_outputs.loss_ce.item(), 'DPO LOSS: ', trainer_outputs.loss_dpo.item())
                trainer.predict(trainer_outputs.logits, chosen_data.instructions, chosen_data.outputs)
            if trainer.step % 7200 == 0:
                trainer.save(os.path.join(policy_save_dir, f"epoch-{epoch + 1}"))
        trainer.save(os.path.join(policy_save_dir, f"epoch-{epoch + 1}"))

        if task is not None:
            evaluator = SolverEvaluator(policy, policy_tokenizer, eval_batch_size, policy_max_seq_len)
            eval_outputs = evaluator.forward(task, JsonDataset(label_file))
            print("Evaluate Accuracy: ", eval_outputs.acc, "Missing: ", eval_outputs.missing)
            json_dump(eval_outputs.datalist, os.path.join(
                log_dir, f'results-epoch-{epoch + 1}-{round(eval_outputs.acc, 4)}.json'), indent=4
            )

        policy.cpu()
        del policy
        del optimizer
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()


if __name__ == '__main__':
    fire.Fire(main)
