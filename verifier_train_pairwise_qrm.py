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
from src.rewards.trainer import ParallelVerifierTrainerForQRM
from src.trainer import prepare_for_forward
from src.utils import print_current_func_args, json_load


def main(
        log_dir: str,
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        model_type: str,
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lr: float = 1e-5,
        epochs: int = 1,
        begin_epoch: int = 0,
        chunk_size: int = None,
        lora_rank: int = -1,
        tokenizer_file: str = None,
        config_file: str = None,
        dtype: str = "bfloat16",
        lora_dtype: str = "float32",
        coef: float = 1.0,
        ce_coef: float = 1.0,
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

        rollout_buffer = RolloutBuffer()
        if use_chat_template:
            dataset = ChatTemplateDataset(dataset, verifier_tokenizer)
        dataloader = ParallelDataLoader(dataset, batch_size=1024)
        timer = Timer(total=len(dataloader), episode=10)
        for data in dataloader:
            timer.step()
            prepare_chosen_outputs = prepare_for_forward(
                instructions=data["instruction"],
                responses=data["chosen"],
                tokenizer=verifier_tokenizer,
                max_seq_len=max_seq_len
            )
            prepare_chosen_outputs.labels[prepare_chosen_outputs.labels == -100] = 0
            prepare_rejected_outputs = prepare_for_forward(
                instructions=data["instruction"],
                responses=data["rejected"],
                tokenizer=verifier_tokenizer,
                max_seq_len=max_seq_len
            )
            prepare_rejected_outputs.labels[prepare_rejected_outputs.labels == -100] = 0
            rollout_buffer.extend(RolloutBuffer(
                chosen_obs=prepare_chosen_outputs.tokens.cpu().numpy(),
                rejected_obs=prepare_rejected_outputs.tokens.cpu().numpy(),
                chosen_actions=prepare_chosen_outputs.labels.cpu().numpy(),
                rejected_actions=prepare_rejected_outputs.labels.cpu().numpy(),
                chosen_action_masks=prepare_chosen_outputs.masks.cpu().numpy(),
                rejected_action_masks=prepare_rejected_outputs.masks.cpu().numpy(),
            ))

        optimizer = torch.optim.Adam(verifier.parameters(), lr=lr)
        trainer = ParallelVerifierTrainerForQRM(
            model=verifier,
            optimizer=optimizer,
            ce_coef=ce_coef,
            coef=coef,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )
        verifier.load(ckpt_dir) if (
                epoch == 0
        ) else trainer.load(os.path.join(save_dir, "epoch-%03d" % epoch))
        timer = Timer(rollout_buffer.size() // max_batch_size, episode=10)
        for data in rollout_buffer.get(max_batch_size, shuffle=True, output_tensor=True):
            timer.step()
            trainer_outputs = trainer.forward(data)
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {timer.total} ---------------')
                print(f'LOSS: {trainer_outputs.loss} | CE LOSS {trainer_outputs.ce_loss} | ACC: {trainer.verifier_accuracy()}')
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
