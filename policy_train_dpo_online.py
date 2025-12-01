import os

import fire

from policy_train_dpo import train_dpo, collect_reference_buffer
from policy_train_policy_gradient_with_rule_rm_anchor import preprocess_anchor_file
from policy_train_ppo import collect_actor_buffer
from policy_train_ppo_with_evaluate import evaluate_actor
from src.dataset import PairwiseDataset, JsonDataset
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.initialize import setup_model_parallel
from src.ppo.buffer import RolloutBuffer
from src.utils import print_current_func_args


def create_pairwise_dataset(dataset: JsonDataset, reference_rollout_buffer: RolloutBuffer) -> PairwiseDataset:
    datalist = []
    dataloader = ParallelDataLoader(dataset, batch_size=1)  # TODO: Do not rely on this dataloader to get chosen sample
    assert len(dataloader) == reference_rollout_buffer.size()
    for data, buffer_data in zip(dataloader, reference_rollout_buffer.get(1)):
        assert data["instruction"][0] in buffer_data.instructions[0]
        datalist.append(dict(
            instruction=data["instruction"][0],
            chosen=data["output"][0],
            rejected=buffer_data.responses[0]
        ))
    return PairwiseDataset(datalist)


def run(
        task: str,
        train_file: str,
        label_file: str,
        log_dir: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_tokenizer_file: str,
        policy_config_file: str,
        max_seq_len: int = 1024,
        max_batch_size: int = 1,
        forward_batch_size: int = 1,
        generate_batch_size: int = 1,
        lr: float = 1e-6,
        dtype: str = "bfloat16",
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        chunk_size: int = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        beta: float = 0.1,
        epochs: int = 1,
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1
):
    setup_model_parallel(
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size,
        seed=seed,
        log_dir=log_dir
    )
    print_current_func_args()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir

    _, datalist = preprocess_anchor_file(train_file, num_anchors=2)
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

            reference_ckpt_dir = policy_ckpt_dir if epoch <= 1 else os.path.join(save_dir, "epoch-%03d" % (epoch - 1))
            # Reference model buffer collecting ...
            reference_rollout_buffer = collect_actor_buffer(
                actor_model_type=policy_model_type,
                actor_config_file=policy_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=policy_tokenizer_file,
                dtype=dtype,
                actor_ckpt_dir=reference_ckpt_dir,
                epoch=epoch,
                actor_save_dir=save_dir,
                use_chat_template=use_chat_template,
                dataset=dataset,
                max_generate_batch_size=generate_batch_size,
                temperature=temperature,
                top_p=top_p,
            )

            pairwise_dataset = create_pairwise_dataset(dataset, reference_rollout_buffer)

            # Reference model logprobs collecting ...
            ref_chosen_rollout_buffer, ref_rejected_rollout_buffer = collect_reference_buffer(
                dataset=pairwise_dataset,
                reference_ckpt_dir=reference_ckpt_dir,
                reference_model_type=policy_model_type,
                reference_config_file=policy_config_file,
                reference_tokenizer_file=policy_tokenizer_file,
                max_forward_batch_size=forward_batch_size,
                max_seq_len=max_seq_len,
                dtype=dtype,
                use_chat_template=use_chat_template,
                save_dir=save_dir,
                local_epoch=local_epoch
            )

            # policy DPO training ...
            train_dpo(
                ref_chosen_rollout_buffer=ref_chosen_rollout_buffer,
                ref_rejected_rollout_buffer=ref_rejected_rollout_buffer,
                policy_model_type=policy_model_type,
                policy_ckpt_dir=policy_ckpt_dir,
                policy_config_file=policy_config_file,
                policy_tokenizer_file=policy_tokenizer_file,
                save_dir=save_dir,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                dtype=dtype,
                lora_rank=lora_rank,
                lora_dtype=lora_dtype,
                beta=beta,
                lr=lr,
                epoch=epoch
            )

            evaluate_actor(
                task=task,
                label_file=label_file,
                log_dir=log_dir,
                actor_model_type=policy_model_type,
                actor_config_file=policy_config_file,
                max_seq_len=max_seq_len,
                actor_tokenizer_file=policy_tokenizer_file,
                dtype=dtype,
                epoch=epoch,
                actor_save_dir=save_dir,
                max_generate_batch_size=generate_batch_size,
                use_chat_template=use_chat_template
            )


if __name__ == '__main__':
    fire.Fire(run)
