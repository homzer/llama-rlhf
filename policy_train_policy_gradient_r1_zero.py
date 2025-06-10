import collections
import os.path
import re
from itertools import zip_longest, islice

import fire
import numpy as np

from policy_train_policy_gradient import train_policy_gradient
from policy_train_policy_gradient_with_rule_rm import collect_actor_buffer_with_label
from policy_train_ppo_with_evaluate import evaluate_actor
from src.dataset import JsonDataset
from src.evaluator import EVALUATORS
from src.parallel.initialize import setup_model_parallel
from src.ppo.buffer import PolicyRolloutBuffer, RolloutBuffer, CriticRolloutBuffer
from src.ppo.parallel_buffer import ParallelRolloutBuffer
from src.utils import json_load, print_current_func_args

SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first tinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process is must enclosed within <think> </think> tags. The answer is must enclosed within <answer> </answer> tags, and Enclose final answer with $\\boxed{{}}$. For example:
User: Some question ...
Assistant: 
<think> Provide your reasoning here </think>
<answer> The answer is $\\boxed{{...}}$ </answer>. 
User: {prompt} 
Assistant: """


def preprocess_datalist(datalist: list) -> list:
    assert "task" in datalist[0]
    assert "label" in datalist[0]
    assert "instruction" in datalist[0]
    for i in range(len(datalist)):
        datalist[i]["instruction"] = SYSTEM_PROMPT.format_map({"prompt": datalist[i]["instruction"]})
    return datalist


def preprocess_eval_dataset(label_file: str) -> JsonDataset:
    datalist = json_load(label_file)
    assert "instruction" in datalist[0]
    for i in range(len(datalist)):
        datalist[i]["instruction"] = SYSTEM_PROMPT.format_map({"prompt": datalist[i]["instruction"]})
    return JsonDataset(datalist)


def format_correct(s: str) -> bool:
    """
    Unit Tests:
    assert format_correct("I am a<think>True is</think> So. <answer>180</answer> KK") is True
    assert format_correct("<think><think>True is</think> So. <answer>180</answer> KK") is True
    assert format_correct("<think>True is</think> So. <answer>180</answer> KK") is True
    assert format_correct("<think>True is</think><answer>180</answer> KK") is True
    assert format_correct("<think>True is</think><answer>180</answer>") is True
    assert format_correct("<think>True is</think>\nW<answer>180</answer>") is True
    assert format_correct("I am\n a<think>True\n is</think> So. <answer>180\n</answer> KK\n") is True
    assert format_correct("I am\n a<think>True\n is<think> So. <answer>180\n</answer> KK\n") is False
    assert format_correct("I am\n a<think>True\n is</think> So. <answer>180\n<answer> KK\n") is False
    assert format_correct("I am\n a<think></think> So. <answer>180\n</answer> KK\n") is False
    assert format_correct("I am\n a<think>True\n is</think> So. <answer></answer> KK\n") is False
    """
    return re.search(r".*<think>.+</think>.*<answer>.+</answer>.*", s, flags=re.DOTALL) is not None


def prethink_length(s: str) -> int:
    match = re.search(r"^(.*)<think>", s, flags=re.DOTALL)
    if match is not None:
        return len(match.group(1))
    return 0


def think_length(s: str) -> int | None:
    match = re.search(r"<think>(.*)</think>", s, flags=re.DOTALL)
    if match is not None:
        return len(match.group(1))
    return None


def repetition(s: str) -> float:
    """Compute repetition score. 0 means no repetition, 1 means the highest degree of repetition."""
    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0

        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue

            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1

            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))

    return cnt * 2 / (n * (n + 1))


def reward_fn(policy_rollout_buffer: RolloutBuffer, task: str):
    """
    :return:
    acc_rewards: 1 for correct answer, -1 for incorrect answer.
    format_rewards: 1 for format correct response, -1 for format incorrect response.
    think_len_rewards: 1 for the longest correct think response, 0 for the shortest.
    prethink_len_rewards: 1 for response without prethink, -1 for the response with longest prethink.
    """
    evaluator = EVALUATORS[task.lower()]()
    format_rewards = []
    acc_rewards = []
    think_len_rewards = []
    prethink_len_rewards = []
    repetition_rewards = []
    max_prethink_len = 1
    max_think_len = None
    min_think_len = None
    for i, data in enumerate(policy_rollout_buffer.get(1)):
        if evaluator.eval(data.responses[0], data.labels[0]) is True:
            acc_rewards.append(1.0)  # answer correct
            think_len = think_length(data.responses[0])
            if think_len is not None:
                max_think_len = think_len if max_think_len is None else max(max_think_len, think_len)
                min_think_len = think_len if min_think_len is None else min(min_think_len, think_len)
        else:
            acc_rewards.append(-1.0)  # answer incorrect

        if format_correct(data.responses[0]):
            format_rewards.append(1.0)  # format correct
            max_prethink_len = max(max_prethink_len, prethink_length(data.responses[0]))
        else:
            format_rewards.append(-1.0)  # format incorrect

        repetition_rewards.append(repetition(data.responses[0]))

    for i, data in enumerate(policy_rollout_buffer.get(1)):
        if acc_rewards[i] == 1.0:  # answer correct
            # think length rewards
            if max_think_len is not None and min_think_len is not None and max_think_len > min_think_len:
                think_len = think_length(data.responses[0])
                if think_len is not None:
                    think_len_rewards.append((think_len - min_think_len) / (max_think_len - min_think_len))
                else:
                    think_len_rewards.append(0)
            else:
                think_len_rewards.append(0)
        else:
            think_len_rewards.append(0)

        if format_rewards[i] == 1.0:
            # prethink rewards
            prethink_len_rewards.append(1 - 2 * prethink_length(data.responses[0]) / max_prethink_len)
        else:
            prethink_len_rewards.append(-1)

    assert len(acc_rewards) == len(format_rewards)
    assert len(acc_rewards) == len(think_len_rewards)
    assert len(acc_rewards) == len(prethink_len_rewards)
    Output = collections.namedtuple("Output", [
        "acc_rewards", "format_rewards", "think_len_rewards", "prethink_len_rewards", "repetition_rewards"])
    return Output(
        acc_rewards=acc_rewards,
        format_rewards=format_rewards,
        think_len_rewards=think_len_rewards,
        prethink_len_rewards=prethink_len_rewards,
        repetition_rewards=repetition_rewards
    )


def collect_rule_based_verifier_buffer(
        policy_rollout_buffer: RolloutBuffer, task: str
) -> CriticRolloutBuffer:
    outputs = reward_fn(policy_rollout_buffer, task=task)
    print(f"Accuracy Rewards: {np.mean(outputs.acc_rewards)}")
    print(f"Format Rewards: {np.mean(outputs.format_rewards)}")
    print(f"Think Length Rewards: {np.mean(outputs.think_len_rewards)}")
    print(f"Prethink Length Rewards: {np.mean(outputs.prethink_len_rewards)}")
    print(f"Repetition Scores (lower the better): {np.mean(outputs.repetition_rewards)}")
    scores = []
    for acc_reward, format_reward, think_len_reward, prethink_len_reward, repetition_reward in zip(
            outputs.acc_rewards, outputs.format_rewards, outputs.think_len_rewards, outputs.prethink_len_rewards, outputs.repetition_rewards
    ):
        scores.append(acc_reward + 0.3 * format_reward + 1.5 * think_len_reward + 0.4 * prethink_len_reward - 1.5 * repetition_reward)
    verifier_rollout_buffer = CriticRolloutBuffer(scores, action_masks=policy_rollout_buffer["action_masks"])
    return verifier_rollout_buffer


def run(
        train_file: str,
        log_dir: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 4,
        max_generate_batch_size: int = 256,
        max_seq_len: int = 4096,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 64,
        epochs: int = 1,
        chunk_size: int = 128,
        inner_epochs: int = 1,
        lr: float = 1e-6,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        seed: int = None,
        train_strategy: str = "vanilla",
        delta: float = 0.01,
        use_chat_template: bool = False,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
):
    setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir

    datalist = preprocess_datalist(json_load(train_file))
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
            policy_rollout_buffer = collect_actor_buffer_with_label(
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
                num_samples_per_prompt=num_samples_per_prompt
            )

            verifier_rollout_buffer = collect_rule_based_verifier_buffer(
                policy_rollout_buffer=policy_rollout_buffer, task='math'
            )

            print(f"Average Rewards: {verifier_rollout_buffer.mean(use_last_token_reward=True)}")

            ParallelRolloutBuffer(**policy_rollout_buffer).save(
                os.path.join(save_dir, "epoch-%03d" % epoch, "policy-buffer"))
            ParallelRolloutBuffer(**verifier_rollout_buffer).save(
                os.path.join(save_dir, "epoch-%03d" % epoch, "verifier-buffer"))

            rollout_buffer = PolicyRolloutBuffer(
                obs=policy_rollout_buffer["obs"],
                actions=policy_rollout_buffer["actions"],
                rewards=verifier_rollout_buffer["scores"],
                values=verifier_rollout_buffer["scores"],
                action_logits=policy_rollout_buffer["action_logits"],
                action_masks=policy_rollout_buffer["action_masks"],
                action_logprobs=policy_rollout_buffer["action_logprobs"],
                use_last_token_reward=True,
                last_token_reward_only=False,
                reward_normalize=False,
                reward_sub_mean=False
            )

            train_policy_gradient(
                rollout_buffer=rollout_buffer,
                policy_ckpt_dir=policy_ckpt_dir,
                policy_model_type=policy_model_type,
                policy_config_file=policy_config_file,
                policy_tokenizer_file=policy_tokenizer_file,
                max_seq_len=max_seq_len,
                lora_rank=lora_rank,
                dtype=dtype,
                lora_dtype=lora_dtype,
                lr=lr,
                epoch=epoch,
                inner_epochs=inner_epochs,
                save_dir=save_dir,
                max_batch_size=max_batch_size,
                train_strategy=train_strategy,
                delta=delta
            )

            for task in ['prm800k', 'aime2024', 'aime2025', 'amc23', 'gsm8k', 'gpqa-diamond']:
                label_file = f"../../data/{task}_test_with_zero_shot.jsonl"
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
                    max_generate_batch_size=max_generate_batch_size,
                    use_chat_template=use_chat_template,
                    dataset=preprocess_eval_dataset(label_file)
                )


if __name__ == '__main__':
    fire.Fire(run)
