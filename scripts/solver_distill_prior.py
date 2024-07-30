import os

import fire
import torch

from src.entities import Timer
from src.modeling import get_parallel_model
from src.trainer import ParallelSolverTripleDistillTrainer
from src.utils import setup_model_parallel


def main(
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        prior_train_file: str,
        model_type: str = "llama-1-7b",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lr: float = 1e-5,
        epochs: int = 1,
        lora_rank: int = 16,
        tokenizer_file: str = None,
        config_file: str = None,
        log_dir: str = None,
        seed: int = None,
        use_float16: bool = True
):
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
    local_rank, world_size = setup_model_parallel(
        seed=seed
    )
    rollout_buffer = torch.load(train_file)
    prior_rollout_buffer = torch.load(prior_train_file)
    assert len(rollout_buffer) == len(prior_rollout_buffer)

    model, tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=lora_rank
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = ParallelSolverTripleDistillTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        max_seq_len=max_seq_len
    )
    trainer.load(ckpt_dir)

    for epoch in range(epochs):
        timer = Timer(len(rollout_buffer) // max_batch_size, episode=100)
        for data, prior_data in zip(
                rollout_buffer.get(max_batch_size),
                prior_rollout_buffer.get(max_batch_size)
        ):
            assert data.instructions[0] == prior_data.instructions[0]
            timer.step()
            outputs = trainer.distill(
                instructions=data.instructions,
                outputs=data.outputs,
                target_logits_a=data.logits,
                target_logits_b=prior_data.logits
            )
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {len(rollout_buffer) // max_batch_size} ---------------')
                print(f'CE LOSS: ', outputs.loss_ce.item(),
                      'KL LOSS: ', outputs.loss_kl.item(),
                      'PRIOR KL LOSS: ', outputs.loss_kl_.item())
                trainer.predict(outputs.logits, data.instructions, data.outputs)
            if trainer.step % 7200 == 0:
                trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))
        trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))


if __name__ == '__main__':
    fire.Fire(main)
