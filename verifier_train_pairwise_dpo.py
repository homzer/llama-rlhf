import gc
import os

import fire
import torch
from torch.utils.data import DataLoader

from src.rewards.trainer import ParallelVerifierTrainerForDPO
from src.dataset import PairwiseDataset, ChatTemplateDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.ppo.buffer import LogitsRolloutBuffer
from src.ppo.collector import LogitsBufferCollector
from src.parallel.initialize import setup_model_parallel, set_barrier, get_rank


def main(
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        buffer_dir: str = None,
        model_type: str = "qwen",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        forward_batch_size: int = 32,
        lr: float = 1e-5,
        epochs: int = 1,
        begin_epoch: int = 0,
        lora_rank: int = -1,
        tokenizer_file: str = None,
        config_file: str = None,
        dtype: str = "bfloat16",
        lora_dtype: str = "float32",
        use_chat_template: bool = False,
        seed: int = None,
):
    os.makedirs(save_dir, exist_ok=True)
    tokenizer_file = ckpt_dir if tokenizer_file is None else tokenizer_file
    config_file = ckpt_dir if config_file is None else config_file
    buffer_dir = save_dir if buffer_dir is None else buffer_dir
    chosen_buffer_file = os.path.join(buffer_dir, "chosen-buffer.jsonl")
    rejected_buffer_file = os.path.join(buffer_dir, "rejected-buffer.jsonl")
    setup_model_parallel(seed=seed)
    dataset = PairwiseDataset(f=train_file)

    for epoch in range(begin_epoch, epochs):
        if not os.path.exists(chosen_buffer_file) or not os.path.exists(rejected_buffer_file):
            print("Reference buffer collecting ...")
            # Reference model logits collecting
            reference, reference_tokenizer = get_parallel_model(
                model_type=model_type,
                config_file=config_file,
                max_seq_len=max_seq_len,
                tokenizer_file=tokenizer_file,
                lora_rank=lora_rank,
                dtype=dtype,
                lora_dtype=lora_dtype
            )
            reference.load(ckpt_dir)
            reference_buffer_collector = LogitsBufferCollector(
                model=reference,
                tokenizer=reference_tokenizer,
                max_seq_len=max_seq_len
            )
            reference_chosen_rollout_buffer = LogitsRolloutBuffer()
            reference_rejected_rollout_buffer = LogitsRolloutBuffer()
            reference_dataloader = DataLoader(
                ChatTemplateDataset(dataset, reference_tokenizer) if use_chat_template else dataset,
                batch_size=forward_batch_size
            )
            timer = Timer(len(reference_dataloader), episode=10)
            for data in reference_dataloader:
                timer.step()
                reference_chosen_rollout_buffer.extend(
                    reference_buffer_collector.forward(
                        data["instruction"], data["chosen"]
                    )
                )
                reference_rejected_rollout_buffer.extend(
                    reference_buffer_collector.forward(
                        data["instruction"], data["rejected"]
                    )
                )

            reference.cpu()
            del reference
            del reference_buffer_collector
            torch.cuda.empty_cache()
            gc.collect()
            set_barrier()
            if get_rank() == 0:
                reference_chosen_rollout_buffer.save(os.path.join(save_dir, "chosen"))
                reference_rejected_rollout_buffer.save(os.path.join(save_dir, "rejected"))
            set_barrier()
        else:
            reference_chosen_rollout_buffer = LogitsRolloutBuffer().load(chosen_buffer_file)
            reference_rejected_rollout_buffer = LogitsRolloutBuffer().load(rejected_buffer_file)

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
        trainer = ParallelVerifierTrainerForDPO(
            model=verifier,
            tokenizer=verifier_tokenizer,
            optimizer=optimizer,
            max_seq_len=max_seq_len
        )
        verifier.load(ckpt_dir) if (
                epoch == 0
        ) else trainer.load(os.path.join(save_dir, f"epoch-{epoch}"))
        timer = Timer(len(reference_chosen_rollout_buffer) // max_batch_size, episode=10)
        for chosen_data, rejected_data in zip(
                reference_chosen_rollout_buffer.get(max_batch_size),
                reference_rejected_rollout_buffer.get(max_batch_size)
        ):
            assert chosen_data.instructions == rejected_data.instructions
            timer.step()
            trainer_outputs = trainer.forward(
                instructions=chosen_data.instructions,
                chosen=chosen_data.outputs,
                rejected=rejected_data.outputs,
                reference_chosen_log_probs=chosen_data.output_tokens_logps,
                reference_rejected_log_probs=rejected_data.output_tokens_logps
            )
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {timer.total} ---------------')
                print(f'LOSS: ', trainer_outputs.loss.item(), "Acc", trainer.verifier_accuracy())
                trainer.predict(trainer_outputs.logits, chosen_data.instructions, chosen_data.outputs)
            if trainer.step % 10000 == 0:
                trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))
        trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))


if __name__ == '__main__':
    fire.Fire(main)
