import gc
import os

import fire
import torch.optim

from policy_train_ppo import collect_actor_buffer, collect_verifier_buffer
from policy_train_ppo_best_of_n import select_best_of_n_buffer
from src.dataset import JsonDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.parallel.utils import setup_model_parallel, set_barrier
from src.trainer import ParallelSolverTrainer
from src.utils import json_load


def run(
        train_file: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        verifier_config_file: str = None,
        verifier_tokenizer_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 1,
        max_forward_batch_size: int = 1,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        chunk_size: int = None,
        inner_epochs: int = 1,
        epochs: int = 1,
        lr: float = 1e-6,
        num_samples_per_prompt: int = 4,
        num_samples_keep_per_prompt: int = 1,
        use_last_token_reward: bool = False,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
):
    setup_model_parallel()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir
    verifier_config_file = verifier_config_file or verifier_ckpt_dir
    verifier_tokenizer_file = verifier_tokenizer_file or verifier_ckpt_dir

    datalist = json_load(train_file)
    chunk_size = chunk_size or len(datalist)
    local_epochs = len(datalist) // chunk_size
    begin_global_epoch = begin_epoch // local_epochs
    begin_local_epoch = begin_epoch % local_epochs
    for global_epoch in range(begin_global_epoch, epochs):
        for local_epoch in range(begin_local_epoch, local_epochs):
            epoch = local_epoch + global_epoch * local_epochs
            print(f"Epoch - {epoch} of {local_epochs * epochs}")
            dataset = JsonDataset(f=datalist[local_epoch * chunk_size: (local_epoch + 1) * chunk_size])
            if len(dataset) == 0:
                continue

            # Collecting policy buffer
            policy_rollout_buffer = collect_actor_buffer(
                actor_model_type=policy_model_type,
                actor_config_file=policy_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=policy_tokenizer_file,
                dtype=dtype,
                actor_ckpt_dir=policy_ckpt_dir,
                epoch=epoch,
                actor_save_dir=save_dir,
                use_chat_template=use_chat_template,
                dataset=dataset,
                max_generate_batch_size=max_generate_batch_size,
                temperature=temperature,
                top_p=top_p,
                num_samples_per_prompt=num_samples_per_prompt,
            )

            # Collecting verifier buffer
            verifier_rollout_buffer = collect_verifier_buffer(
                verifier_model_type=verifier_model_type,
                verifier_config_file=verifier_config_file,
                max_seq_len=max_seq_len,
                verifier_tokenizer_file=verifier_tokenizer_file,
                dtype=dtype,
                verifier_ckpt_dir=verifier_ckpt_dir,
                actor_rollout_buffer=policy_rollout_buffer,
                max_forward_batch_size=max_forward_batch_size
            )

            policy_rollout_buffer, _ = select_best_of_n_buffer(
                actor_rollout_buffer=policy_rollout_buffer,
                verifier_rollout_buffer=verifier_rollout_buffer,
                num_samples_per_prompt=num_samples_per_prompt,
                num_samples_keep_per_prompt=num_samples_keep_per_prompt,
                use_last_token_reward=use_last_token_reward
            )

            print("Training policy ......")
            model, tokenizer = get_parallel_model(
                model_type=policy_model_type,
                config_file=policy_config_file,
                max_seq_len=max_seq_len,
                tokenizer_file=policy_tokenizer_file,
                lora_dtype=lora_dtype,
                lora_rank=lora_rank,
                dtype=dtype
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            trainer = ParallelSolverTrainer(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                max_seq_len=max_seq_len
            )
            trainer.load_model(policy_ckpt_dir) if (
                    epoch == 0
            ) else trainer.load(os.path.join(save_dir, f"epoch-{epoch}"))
            timer = Timer(total=(len(policy_rollout_buffer) // max_batch_size) * inner_epochs, episode=100)
            for inner_epoch in range(inner_epochs):
                for data in policy_rollout_buffer.get(max_batch_size):
                    timer.step()
                    trainer_outputs = trainer.forward(
                        instructions=data.instructions.tolist(),
                        outputs=data.responses.tolist()
                    )
                    if trainer.step % 100 == 0:
                        print(f'--------- STEP {trainer.step} OF {timer.total} ---------')
                        print('Loss: ', trainer_outputs.loss)
            trainer.save(os.path.join(save_dir, f"epoch-{epoch + 1}"))

            model.cpu()
            del model
            del optimizer
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
            set_barrier()


if __name__ == '__main__':
    fire.Fire(run)
