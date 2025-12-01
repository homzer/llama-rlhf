import gc
import os
import shutil

import fire
import torch

from policy_train_dpo import collect_reference_buffer
from src.dataset import PairwiseDataset
from src.entities import Timer, IterationHandler
from src.modeling import get_parallel_model
from src.parallel.initialize import setup_model_parallel, set_barrier, get_rank
from src.rewards.trainer import ParallelVerifierTrainerForDPO
from src.utils import print_current_func_args, json_load


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
        reuse_buffer: bool = False,
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

        reference_rollout_buffer = collect_reference_buffer(
            dataset=dataset,
            reference_ckpt_dir=ckpt_dir,
            reference_model_type=model_type,
            reference_config_file=config_file,
            reference_tokenizer_file=tokenizer_file,
            max_forward_batch_size=max_forward_batch_size,
            max_seq_len=max_seq_len,
            dtype=dtype,
            use_chat_template=use_chat_template,
            log_dir=log_dir,
            local_epoch=iterator.local_epoch,
            reuse_buffer=reuse_buffer
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
        trainer = ParallelVerifierTrainerForDPO(
            model=verifier,
            optimizer=optimizer,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )
        verifier.load(ckpt_dir) if (
                epoch == 0
        ) else trainer.load(os.path.join(save_dir, "epoch-%03d" % epoch))
        timer = Timer(len(reference_rollout_buffer) // max_batch_size, episode=10)
        for data in reference_rollout_buffer.get(max_batch_size, shuffle=True, output_tensor=True):
            timer.step()
            trainer_outputs = trainer.forward(data)
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {timer.total} ---------------')
                print(f'LOSS: ', trainer_outputs.loss.item(), "Acc", trainer.verifier_accuracy())
            if trainer.step % 10000 == 0:
                trainer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))
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
