import collections
import gc
import os

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import savgol_filter
from tqdm import trange

from policy_train_ppo_with_rule_rm import collect_actor_buffer_with_label, collect_rule_based_verifier_buffer
from policy_train_ppo_with_evaluate import evaluate_actor
from src.dataset import JsonDataset
from src.entities import Timer
from src.modeling import get_parallel_model
from src.models.modeling import ParallelModelForCausalLM
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.parallel.optimizer import ParallelOptimizer
from src.ppo.buffer import PPORolloutBuffer, PPORolloutBufferSample
from src.trainer import ParallelTrainer
from src.utils import json_load, print_current_func_args

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


class ParallelGRPOTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            model: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            clip_range: float = 0.2,
            kl_coef: float = 0.04
    ):
        super().__init__(model, optimizer)
        self.model = model
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.step = 0

    def forward(self, rollout_data: PPORolloutBufferSample):
        self.model.train()
        self.step += 1

        obs = rollout_data.observations.to(self.model.device())
        actions = rollout_data.actions.to(self.model.device())
        action_masks = rollout_data.action_masks.to(self.model.device())
        rewards = rollout_data.rewards.to(self.model.device())
        old_action_logprobs = rollout_data.old_action_logprobs.to(self.model.device())

        outputs = self.model.forward(obs)
        action_logprobs = torch.gather(
            torch.log_softmax(outputs.logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        # Normalize rewards
        rewards = torch.masked_select(rewards.view(-1), action_masks.view(-1))
        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(action_logprobs - old_action_logprobs)
        ratio = torch.masked_select(ratio.view(-1), action_masks.view(-1))
        # clipped surrogate loss
        policy_loss = rewards * ratio
        if self.clip_range > 0:
            clipped_actor_loss = rewards * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = torch.min(policy_loss, clipped_actor_loss)
        policy_loss = - torch.mean(policy_loss)

        kl_loss = 0.0
        if rollout_data.ref_action_logprobs is not None:
            ref_action_logprobs = rollout_data.ref_action_logprobs.to(self.model.device())
            probs_ratios = torch.exp(ref_action_logprobs - action_logprobs)
            kl_loss = self.kl_coef * torch.masked_select(
                (probs_ratios - (ref_action_logprobs - action_logprobs) - 1).view(-1),
                action_masks.view(-1)
            ).mean()

        loss = policy_loss + kl_loss
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = []
        for name, param in self.model.named_parameters():
            if "q_proj.weight" in name or "k_proj.weight" in name or "v_proj.weight" in name:
                grad_norm.append(torch.linalg.norm(param.grad).item())
        self.optimizer.step()

        Outputs = collections.namedtuple('Outputs', [
            'loss', "policy_loss", 'rewards', "kl_loss", "ratio", "grad_norm"])
        return Outputs(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            rewards=torch.mean(rewards).item(),
            kl_loss=kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            ratio=ratio.detach().cpu().tolist(),
            grad_norm=np.mean(grad_norm).item()
        )


def train_policy_gradient(
        rollout_buffer: PPORolloutBuffer,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_config_file: str,
        policy_tokenizer_file: str,
        max_seq_len: int,
        lora_rank: int,
        dtype: str,
        lora_dtype: str,
        lr: float,
        epoch: int,
        inner_epochs: int,
        save_dir: str,
        max_batch_size: int
) -> (list, list):
    policy, policy_tokenizer = get_parallel_model(
        model_type=policy_model_type,
        config_file=policy_config_file,
        max_seq_len=max_seq_len,
        tokenizer_file=policy_tokenizer_file,
        lora_rank=lora_rank,
        dtype=dtype,
        lora_dtype=lora_dtype
    )
    optimizer = ParallelOptimizer(torch.optim.Adam(policy.parameters(), lr=lr))
    trainer = ParallelGRPOTrainerForCausalLM(policy, optimizer)

    ratios = []
    grad_norms = []
    trainer.load_model(policy_ckpt_dir) if (
            epoch == 0
    ) else trainer.load(os.path.join(save_dir, "epoch-%03d" % epoch))
    print('Policy training ...')
    timer = Timer(total=(len(rollout_buffer) // max_batch_size) * inner_epochs, episode=100)
    for inner_epoch in range(inner_epochs):
        for data in rollout_buffer.get(max_batch_size):
            timer.step()
            trainer_outputs = trainer.forward(data)
            ratios.append(trainer_outputs.ratio)
            grad_norms.append(trainer_outputs.grad_norm)
            if trainer.step % 100 == 0:
                print(f'--------- STEP {trainer.step} OF {timer.total} ---------')
                print(f'Loss: {trainer_outputs.loss}')
                print(f'Rewards: {trainer_outputs.rewards}')
    trainer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))

    policy.cpu()
    del policy
    del optimizer
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    set_barrier()

    return ratios, grad_norms


def run(
        task: str,
        label_file: str,
        train_file: str,
        log_dir: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        max_batch_size: int = 1,
        max_generate_batch_size: int = 48,
        max_seq_len: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_samples_per_prompt: int = 1,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 1e-5,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
        reward_sub_mean: bool = False,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
):
    parallel_infos = setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()
    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir

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
                actor_rollout_buffer=policy_rollout_buffer, task=task
            )

            print(f"Average Rewards: {verifier_rollout_buffer.mean(use_last_token_reward=True)}")

            rollout_buffer = PPORolloutBuffer(
                obs=policy_rollout_buffer["obs"],
                actions=policy_rollout_buffer["actions"],
                rewards=verifier_rollout_buffer["scores"],
                values=verifier_rollout_buffer["scores"],  # pseudo
                action_logits=policy_rollout_buffer["action_logits"],
                action_masks=policy_rollout_buffer["action_masks"],
                action_logprobs=policy_rollout_buffer["action_logprobs"],
                use_last_token_reward=True,
                reward_sub_mean=reward_sub_mean
            )

            ratios, grad_norms = train_policy_gradient(
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
                max_batch_size=max_batch_size
            )

            if parallel_infos.global_rank == 0:
                torch.save({
                    'ratios': ratios,
                    'grad_norms': grad_norms,
                    'actions': rollout_buffer.actions,
                    'rewards': rollout_buffer.origin_rewards,
                    'action_masks': rollout_buffer.action_masks,
                    'action_logprobs': rollout_buffer.action_logprobs
                }, os.path.join(save_dir, "epoch-%03d" % (epoch + 1), f"buffer.bin"))

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
                use_chat_template=use_chat_template
            )


def style_subplot(ax, label=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 移动左边框和底边框，使它们不相交于原点
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    # 加粗
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(width=2)

    # 隐藏y轴和x轴的原点处的刻度线
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid(alpha=0.3)

    # 添加子图标签，如果提供了label参数的话
    if label is not None:
        ax.text(-0.2, -0.12, f'({label})', transform=ax.transAxes, size=14, weight='bold')


def style_subplot2(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 移动左边框和底边框，使它们不相交于原点
    ax.spines['right'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    # 加粗
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(width=2)

    # 隐藏y轴和x轴的原点处的刻度线
    ax.yaxis.set_ticks_position('right')
    ax.xaxis.set_ticks_position('bottom')


def draw_plot(ax, y, ymin=None, ymax=None, x_ticks=None, x_ticklabels=None, xlabel=None, ylabel=None):
    smoothed_y = savgol_filter(y, window_length=11, polyorder=2)
    # 绘制图形
    ax.plot(y, color=COLORS[0], alpha=0.2, linewidth=3)  # 原始折线图，透明度为0.5
    ax.plot(smoothed_y, color=COLORS[0], label='Loss')  # 平滑折线图
    ax.set_xlabel(xlabel or 'Rollout Epochs', fontweight='medium')
    ax.set_ylabel(ylabel or 'PPO Policy Loss', fontweight='bold')
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.legend()


def draw_plot_twinx(ax, x1, y1, x2, y2, ymin: float = None, ymax: float = None,
                    title: str = None, y2_ticks: list = None, x_label: str = None, x_ticks=None, x_ticklabels=None):
    ax2 = ax.twinx()
    # 使用 Savitzky-Golay 滤波器平滑数据
    smoothed_y1 = savgol_filter(y1, window_length=11, polyorder=2)
    # 绘制图形
    ax.plot(x1, y1, color=COLORS[0], alpha=0.2, linewidth=3)  # 原始折线图，透明度为0.5
    ax.plot(smoothed_y1, color=COLORS[0], label='Ratio')  # 平滑折线图
    ax2.plot(x2, y2, color=COLORS[1], label="Accuracy")
    # 添加图例和标签
    ax.set_xlabel(x_label or 'Rollout Epochs', fontweight='medium')
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_ticklabels is not None:
        ax.set_xticklabels(x_ticklabels)
    ax.set_ylabel('PPO Ratio', fontweight='bold')
    ax2.set_ylabel('Test Accuracy', fontweight='bold')
    style_subplot2(ax2)
    if title is not None:
        ax.set_title(title, fontweight='medium', fontsize='medium')
    ax.set_ylim(ymin, ymax)

    ax.spines['left'].set_color(COLORS[0])
    ax.tick_params(axis='y', colors=COLORS[0])  # 设置 y 轴刻度的颜色
    ax.yaxis.label.set_color(COLORS[0])  # 设置 y 轴标签的颜色
    ax2.spines['right'].set_color(COLORS[1])
    ax2.tick_params(axis='y', colors=COLORS[1])
    ax2.yaxis.label.set_color(COLORS[1])
    if y2_ticks is not None:
        ax2.set_yticks(y2_ticks)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax2.legend(lines1 + lines2, labels1 + labels2, ncol=1,
               handletextpad=0.5,  # 减少图例标记和文本之间的间距
               columnspacing=0.5,  # 如果有多列，则减少列间距
               borderpad=0.3,  # 减少图例外边框和内容之间的间距
               loc="upper center"
               )


def draw_pdf():
    import re
    import numpy as np

    root_dir = "results/qwen-2.5-math-7b/policy/policy-gradient/prm800k-aime-gsm8k-rule-rm-samples-1-lr-5e-6-chunk-size-3072-ratio-visualize/full/"
    log_file = "log/qwen-2.5-math-7b/policy/policy-gradient/prm800k-aime-gsm8k-rule-rm-samples-1-lr-5e-6-chunk-size-3072-ratio-visualize/full/output.log"
    rewards = []
    losses = []
    accuracies = []
    loss_step = 0
    loss_steps = []
    with open(log_file, 'r', encoding='utf-8') as reader:
        for text in reader:
            if "Average Rewards:" in text:
                match = re.search(r"Average Rewards:\s*(-?\d+\.\d+)", text)
                if match is not None:
                    rewards.append(float(match.group(1)))
            if "Loss:" in text:
                match = re.search(r"Loss:\s*(-?\d+\.\d+)", text)
                if match is not None:
                    losses.append(float(match.group(1)))
                    loss_step += 1
            if "PRM800K Evaluate Accuracy:" in text:
                match = re.search(r"PRM800K Evaluate Accuracy:\s*(-?\d+\.\d+)", text)
                if match is not None:
                    accuracies.append(float(match.group(1)))
                    loss_steps.append(loss_step)

    chunk_size = 100
    ratio_steps = []
    acc_steps = []
    ratios = []
    action_probs = []
    action_steps = []
    action_step = 0
    step = 0
    epochs = []
    for epoch in trange(1, 13):
        epochs.append(epoch)
        buffer = torch.load(os.path.join(root_dir, "epoch-%03d" % epoch, "buffer.bin"))
        i = 0
        while i * chunk_size < len(buffer["ratios"]):
            ratio_data = []
            for j in range(i * chunk_size, (i + 1) * chunk_size):
                if j >= len(buffer["ratios"]):
                    break
                ratio_data.extend(buffer["ratios"][j])
            ratios.append(np.mean(ratio_data))
            ratio_steps.append(step)
            step += 1
            i += 1
        acc_steps.append(step)

        i = 0
        while i * chunk_size < len(buffer["action_logprobs"]):
            action_probs_data = []
            for j in range(i * chunk_size, (i + 1) * chunk_size):
                if j >= len(buffer["action_logprobs"]):
                    break
                action_logprobs = buffer["action_logprobs"][j][buffer["action_masks"][j]]
                action_probs_data.extend(action_logprobs.exp())
            action_probs.append(np.mean(action_probs_data))
            action_step += 1
            i += 1
        action_steps.append(action_step)

    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    axs = np.reshape(axs, (-1,))
    draw_plot_twinx(axs[0], ratio_steps, ratios, acc_steps, accuracies, x_ticks=acc_steps, x_ticklabels=epochs)
    draw_plot(axs[1], action_probs, x_ticks=action_steps, x_ticklabels=epochs, xlabel="Action Probs")
    draw_plot(axs[2], losses, ymin=-2.0, ymax=2.0, x_ticks=loss_steps, x_ticklabels=epochs)

    for i, ax in enumerate(axs):
        style_subplot(ax, chr(97 + i))

    plt.tight_layout()
    plt.savefig("plot_results.pdf")
    # plt.show()


if __name__ == '__main__':
    fire.Fire(run)
