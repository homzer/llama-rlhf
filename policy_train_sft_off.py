import fire

from policy_train_ppo import collect_actor_buffer, collect_verifier_buffer
from policy_train_rft import train_rft_policy
from src.dataset import JsonDataset
from src.entities import IterationHandler
from src.modeling import TOKENIZERS
from src.parallel.initialize import setup_model_parallel
from src.ppo.buffer import RolloutBuffer
from src.utils import json_load, print_current_func_args


def process_pairwise_dataset_for_sft(dataset: JsonDataset, model_type, tokenizer_file) -> RolloutBuffer:
    instructions = []
    responses = []
    tokenizer = TOKENIZERS[model_type](tokenizer_file)
    for data in dataset:
        instructions.append(tokenizer.apply_chat_template(data["instruction"]))
        responses.append(data["chosen"])
    return RolloutBuffer(instructions=instructions, responses=responses)


def run(
        train_file: str,
        label_file: str,
        log_dir: str,
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
        max_generate_batch_size: int = 256,
        max_batch_size: int = 1,
        max_forward_batch_size: int = 12,
        max_seq_len: int = 1024,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
        accumulation_steps: int = 1,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
):
    setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        log_mode="w" if begin_epoch == 0 else "a",
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir

    for epoch, datalist in IterationHandler(json_load(train_file), epochs, chunk_size, begin_epoch):
        dataset = JsonDataset(datalist)
        if len(dataset) == 0:
            continue

        train_rft_policy(
            policy_rollout_buffer=process_pairwise_dataset_for_sft(dataset, policy_model_type, policy_tokenizer_file),
            policy_model_type=policy_model_type,
            policy_config_file=policy_config_file,
            max_seq_len=max_seq_len,
            policy_tokenizer_file=policy_tokenizer_file,
            lora_dtype=lora_dtype,
            lora_rank=lora_rank,
            dtype=dtype,
            lr=lr,
            policy_ckpt_dir=policy_ckpt_dir,
            epoch=epoch,
            save_dir=save_dir,
            max_batch_size=max_batch_size,
            inner_epochs=inner_epochs,
            accumulation_steps=accumulation_steps
        )

        policy_rollout_buffer = collect_actor_buffer(
            actor_model_type=policy_model_type,
            actor_config_file=policy_config_file,
            max_seq_len=max_seq_len,
            actor_tokenizer_file=policy_tokenizer_file,
            dtype=dtype,
            actor_ckpt_dir=policy_ckpt_dir,
            epoch=epoch + 1,
            actor_save_dir=save_dir,
            use_chat_template=use_chat_template,
            dataset=JsonDataset(json_load(label_file)[:max_generate_batch_size]),
            max_generate_batch_size=max_generate_batch_size,
            temperature=0.0,
        )

        verifier_rollout_buffer = collect_verifier_buffer(
            verifier_model_type=verifier_model_type,
            verifier_config_file=verifier_config_file,
            max_seq_len=max_seq_len,
            verifier_tokenizer_file=verifier_tokenizer_file,
            dtype=dtype,
            verifier_ckpt_dir=verifier_ckpt_dir,
            actor_rollout_buffer=policy_rollout_buffer,
            max_forward_batch_size=max_forward_batch_size,
            use_last_token_reward=True,
        )
        print(f"Average Rewards: {verifier_rollout_buffer.mean()}")


if __name__ == '__main__':
    fire.Fire(run)
