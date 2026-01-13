"""
Gradient Norm Visualize V2
"""
import collections
import gc
import os

import fire
import torch

from policy_train_ppo import collect_actor_buffer
from policy_train_ppo_with_dpo_rm import collect_verifier_buffer
from policy_train_lco_with_qrm_v2 import collect_verifier_buffer as collect_verifier_buffer_lco, collect_logits_buffer
from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import IterationHandler, Timer
from src.evaluator import DataParallelPolicyEvaluator
from src.modeling import get_parallel_model
from src.models.modeling import ParallelModelForCausalLM
from src.parallel.data_parallel.utils import gather_tensor_from_data_parallel_region
from src.parallel.initialize import setup_model_parallel, set_barrier
from src.parallel.optimizer import ParallelOptimizer
from src.ppo.buffer import RolloutBuffer
from src.trainer import ParallelTrainer
from src.utils import create_lco_log_target, print_current_func_args, json_load, json_dump

Outputs = collections.namedtuple('Outputs', ['gradient', 'action_probs'])


def evaluate_policy(task, policy, policy_tokenizer, label_file, use_chat_template, max_generate_batch_size,
                    max_seq_len) -> float:
    print("Actor Evaluating ...")
    label_dataset = JsonDataset(label_file)
    if use_chat_template:
        label_dataset = ChatTemplateDataset(label_dataset, policy_tokenizer)
    evaluator = DataParallelPolicyEvaluator(
        model=policy,
        tokenizer=policy_tokenizer,
        batch_size=max_generate_batch_size,
        max_seq_len=max_seq_len
    )
    evaluator_outputs = evaluator.forward(task=task, dataset=label_dataset)
    print(f"{task.upper()} Evaluate Accuracy: {evaluator_outputs.acc}")

    del evaluator

    return evaluator_outputs.acc


@torch.no_grad()
def compute_average_grad_norm(
        policy: ParallelModelForCausalLM,
        return_tensor: bool = False,
        gather_from_data_parallel_region: bool = True
):
    grad_norm = []
    for name, param in policy.named_parameters():
        if "q_proj.weight" in name or "k_proj.weight" in name or "v_proj.weight" in name:
            grad_norm.append(torch.linalg.norm(param.grad))
    grad_norm = torch.stack(grad_norm)
    if gather_from_data_parallel_region:
        grad_norm = gather_tensor_from_data_parallel_region(grad_norm).mean()
    else:
        grad_norm = grad_norm.mean()
    return grad_norm if return_tensor else grad_norm.item()


@torch.no_grad()
def compute_average_action_probs(logits: torch.Tensor, actions: torch.Tensor) -> float:
    action_probs = torch.gather(
        torch.softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
    ).squeeze(-1).mean()
    return gather_tensor_from_data_parallel_region(action_probs).mean().item()


class ParallelPPOTrainerForCausalLM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            clip_range: float = 0.2,
            save_optim: bool = False,
            accumulation_steps: int = 1
    ):
        super().__init__(
            model=policy,
            optimizer=optimizer,
            save_optim=save_optim,
            accumulation_steps=accumulation_steps
        )
        self.policy = policy
        self.clip_range = clip_range

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        advantages = rollout_data.advantages.to(self.policy.device())
        old_action_logprobs = rollout_data.action_logprobs.to(self.policy.device())

        actions = actions.view(-1)[action_masks.view(-1)]
        advantages = advantages.view(-1)[action_masks.view(-1)]
        old_action_logprobs = old_action_logprobs.view(-1)[action_masks.view(-1)]

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]
        action_logprobs = torch.gather(
            torch.log_softmax(logits, dim=-1), dim=-1, index=actions.unsqueeze(-1)
        ).squeeze(-1)

        ratio = torch.exp(action_logprobs - old_action_logprobs)
        # clipped surrogate loss
        loss = advantages * ratio
        if self.clip_range > 0:
            clipped_loss = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            loss = torch.min(loss, clipped_loss)
        loss = - torch.mean(loss)

        self.step += 1
        loss = loss / self.accumulation_steps
        loss.backward()
        gradient = compute_average_grad_norm(self.policy)
        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        action_probs = compute_average_action_probs(logits, actions)

        return Outputs(gradient=gradient, action_probs=action_probs)


class ParallelLCOTrainerForQRM(ParallelTrainer):
    def __init__(
            self,
            policy: ParallelModelForCausalLM,
            optimizer: torch.optim.Optimizer,
            beta: float = 0.1,
            save_optim: bool = False,
            accumulation_steps: int = 1,
    ):
        super().__init__(policy, optimizer, save_optim=save_optim, accumulation_steps=accumulation_steps)
        self.policy = policy
        self.beta = beta
        self.criterion = torch.nn.KLDivLoss(reduction="none", log_target=True)

    def forward(self, rollout_data):
        self.policy.train()

        obs = rollout_data.obs.to(self.policy.device())
        actions = rollout_data.actions.to(self.policy.device())
        action_masks = rollout_data.action_masks.to(self.policy.device())
        advantages = rollout_data.advantages.to(self.policy.device())
        advantage_indices = rollout_data.advantage_indices.to(self.policy.device())
        old_logits = rollout_data.logits.to(self.policy.device())

        actions = actions.view(-1)[action_masks.view(-1)]
        advantages = advantages.view(-1, advantages.shape[-1])[action_masks.view(-1)]
        advantage_indices = advantage_indices.view(-1, advantage_indices.shape[-1])[action_masks.view(-1)]
        old_logits = old_logits.view(-1, old_logits.shape[-1])[action_masks.view(-1)]

        logits = self.policy.forward(obs).logits
        logits = logits.view(-1, logits.shape[-1])[action_masks.view(-1)]

        advantages = (advantages / torch.std(advantages, dim=-1, keepdim=True)).nan_to_num(0.0)
        advantages_ = torch.full_like(old_logits, fill_value=-100)
        advantages_[torch.arange(advantages_.shape[0])[:, None], advantage_indices] = advantages

        log_targets = create_lco_log_target(old_logits, advantages_, beta=self.beta)
        loss = self.criterion.forward(
            torch.log_softmax(logits, dim=-1), target=log_targets.to(logits)
        ).sum(-1).mean().nan_to_num(0.0)
        self.step += 1
        loss = loss / self.accumulation_steps
        loss.backward()
        gradient = compute_average_grad_norm(self.policy)
        if self.step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        action_probs = compute_average_action_probs(logits, actions)

        return Outputs(gradient=gradient, action_probs=action_probs)


# For DPO-RM
def run(
        task: str,
        strategy: str,  # `lco` or `ppo`
        train_file: str,
        label_file: str,
        log_dir: str,
        save_dir: str,
        policy_ckpt_dir: str,
        policy_model_type: str,
        verifier_ckpt_dir: str,
        verifier_model_type: str,
        reference_ckpt_dir: str,
        policy_config_file: str = None,
        policy_tokenizer_file: str = None,
        verifier_config_file: str = None,
        verifier_tokenizer_file: str = None,
        eval_steps: int = 100,
        max_batch_size: int = 2,
        max_generate_batch_size: int = 256,
        max_forward_batch_size: int = 12,
        max_seq_len: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.95,
        num_samples_per_prompt: int = 1,
        epochs: int = 1,
        chunk_size: int = None,
        inner_epochs: int = 1,
        lr: float = 5e-6,
        logits_topk: int = 5,
        beta: float = 0.1,
        dtype: str = "bfloat16",
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
        save_optim: bool = False,
        accumulation_steps: int = 1,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1,
):
    parallel_infos = setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        log_mode='w' if begin_epoch == 0 else 'a',
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()

    policy_config_file = policy_config_file or policy_ckpt_dir
    policy_tokenizer_file = policy_tokenizer_file or policy_ckpt_dir
    verifier_config_file = verifier_config_file or verifier_ckpt_dir
    verifier_tokenizer_file = verifier_tokenizer_file or verifier_ckpt_dir

    results = []
    for epoch, datalist in IterationHandler(json_load(train_file), epochs, chunk_size, begin_epoch):
        if epoch >= 3:
            break

        dataset = JsonDataset(datalist)
        if len(dataset) == 0:
            continue

        # collecting policy buffer
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
            num_samples_per_prompt=num_samples_per_prompt
        )

        if strategy == 'ppo':
            # collecting verifier buffer
            verifier_rollout_buffer = collect_verifier_buffer(
                policy_rollout_buffer=policy_rollout_buffer,
                verifier_model_type=verifier_model_type,
                verifier_ckpt_dir=verifier_ckpt_dir,
                verifier_config_file=verifier_config_file,
                verifier_tokenizer_file=verifier_tokenizer_file,
                reference_ckpt_dir=reference_ckpt_dir,
                max_seq_len=max_seq_len,
                max_forward_batch_size=max_forward_batch_size,
                dtype=dtype
            )
            print(f"Average Rewards: {verifier_rollout_buffer.mean()}")
            verifier_rollout_buffer.normalize()
            policy_rollout_buffer["advantages"] = verifier_rollout_buffer["scores"]
            rollout_buffer = RolloutBuffer(
                obs=policy_rollout_buffer["obs"],
                actions=policy_rollout_buffer["actions"],
                action_masks=policy_rollout_buffer["action_masks"],
                action_logprobs=policy_rollout_buffer["action_logprobs"],
                advantages=verifier_rollout_buffer["scores"]
            )
        elif strategy == 'lco':
            # collecting verifier buffer
            verifier_rollout_buffer = collect_verifier_buffer_lco(
                policy_rollout_buffer=policy_rollout_buffer,
                verifier_model_type=verifier_model_type,
                verifier_ckpt_dir=verifier_ckpt_dir,
                verifier_config_file=verifier_config_file,
                verifier_tokenizer_file=verifier_tokenizer_file,
                max_seq_len=max_seq_len,
                max_forward_batch_size=max_forward_batch_size,
                dtype=dtype,
                logits_topk=logits_topk
            )

            # collecting logits buffer
            rollout_buffer = collect_logits_buffer(
                verifier_rollout_buffer=verifier_rollout_buffer,
                policy_model_type=policy_model_type,
                policy_ckpt_dir=policy_ckpt_dir,
                policy_config_file=policy_config_file,
                policy_tokenizer_file=policy_tokenizer_file,
                max_seq_len=max_seq_len,
                max_forward_batch_size=max_forward_batch_size,
                epoch=epoch,
                save_dir=save_dir,
                dtype=dtype
            )
        else:
            raise ValueError(strategy)

        policy, policy_tokenizer = get_parallel_model(
            model_type=policy_model_type,
            config_file=policy_config_file,
            max_seq_len=max_seq_len,
            tokenizer_file=policy_tokenizer_file,
            dtype=dtype,
        )
        optimizer = ParallelOptimizer(torch.optim.Adam(policy.parameters(), lr=lr))
        if strategy == 'ppo':
            trainer = ParallelPPOTrainerForCausalLM(
                policy=policy,
                optimizer=optimizer,
                save_optim=save_optim,
                accumulation_steps=accumulation_steps
            )
        else:
            trainer = ParallelLCOTrainerForQRM(
                policy=policy,
                optimizer=optimizer,
                beta=beta,
                save_optim=save_optim,
                accumulation_steps=accumulation_steps
            )
        trainer.load_model(policy_ckpt_dir) if (
                epoch == 0
        ) else trainer.load(os.path.join(save_dir, "epoch-%03d" % epoch))
        print("Policy training ...")
        timer = Timer(total=(rollout_buffer.size() // max_batch_size) * inner_epochs, episode=100)
        for inner_epoch in range(inner_epochs):
            for data in rollout_buffer.get(max_batch_size, shuffle=True, output_tensor=True):
                timer.step()
                trainer_outputs = trainer.forward(data)
                results.append(dict(
                    gradient=trainer_outputs.gradient,
                    action_probs=trainer_outputs.action_probs
                ))
                if trainer.step % 10 == 0:
                    print(f'--------- STEP {trainer.step} OF {timer.total} ---------')
                    print(f'Gradient: {trainer_outputs.gradient}')
                    print(f'Action Probs: {trainer_outputs.action_probs}')
                if trainer.step % eval_steps == 0:
                    accuracy = evaluate_policy(
                        task=task,
                        policy=policy,
                        policy_tokenizer=policy_tokenizer,
                        label_file=label_file,
                        use_chat_template=use_chat_template,
                        max_seq_len=max_seq_len,
                        max_generate_batch_size=max_generate_batch_size
                    )
                    results[-1]["accuracy"] = accuracy
                else:
                    results[-1]["accuracy"] = -1

        if parallel_infos.global_rank == 0:
            os.makedirs(log_dir, exist_ok=True)
            json_dump(results, os.path.join(log_dir, "grad.jsonl"))

        trainer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))

        policy.cpu()
        del policy
        del optimizer
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()


if __name__ == '__main__':
    fire.Fire(run)
