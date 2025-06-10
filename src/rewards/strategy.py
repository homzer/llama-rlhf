"""
Different strategies for reward modeling
"""
from typing import List

import torch

from src.criterion import DPOLoss, SimPOLoss, ImplicitPRMLoss


class PointwiseVerifierStrategy:
    def trainer_forward(self, **kwargs):
        raise NotImplementedError

    def generator_forward(self, **kwargs):
        raise NotImplementedError


class PairwiseVerifierStrategy:
    def trainer_forward(self, **kwargs):
        raise NotImplementedError

    def generator_forward(self, **kwargs):
        raise NotImplementedError


class PointwiseVerifierStrategyForLastToken(PointwiseVerifierStrategy):
    """ Binary Classification. Default to be Cross-Entropy Loss """
    def trainer_forward(
            self,
            scores: torch.Tensor,
            masks: torch.Tensor,
            labels: torch.Tensor,
            **kwargs
    ):
        bsz, seq_len = scores.shape
        valid_bsz = bsz
        loss = torch.tensor(0.).to(scores)
        labels = labels.to(scores)
        for i in range(bsz):
            nonzero_indices = masks[i].nonzero()
            if len(nonzero_indices) == 0:
                valid_bsz -= 1
                continue
            end_idx = nonzero_indices[-1].item()
            score = scores[i][end_idx]
            loss += torch.nn.functional.binary_cross_entropy_with_logits(score, labels[i])
        if valid_bsz > 0:
            loss = loss / valid_bsz
        return loss

    def generator_forward(self, scores: torch.Tensor, masks: torch.Tensor) -> List[float]:
        scores = scores.detach().cpu()
        bsz = scores.shape[0]
        reduce_scores = []
        for i in range(bsz):
            check_end = masks[i].nonzero()
            if len(check_end) == 0:
                print("Warming: instruction len out of range. Setting reward score to 0.")
                reduce_scores.append(0)
                continue
            reduce_scores.append(torch.sigmoid(scores[i][check_end[-1].item()]).item())
        return reduce_scores


class PointwiseVerifierStrategyForFocalLoss(PointwiseVerifierStrategyForLastToken):
    def __init__(self, gamma: float = 2.0):
        self.gamma = gamma

    def trainer_forward(
            self,
            scores: torch.Tensor,
            masks: torch.Tensor,
            labels: torch.Tensor,
            **kwargs
    ):
        bsz, seq_len = scores.shape
        valid_bsz = bsz
        loss = torch.tensor(0.).to(scores)
        labels = labels.to(scores)
        for i in range(bsz):
            nonzero_indices = masks[i].nonzero()
            if len(nonzero_indices) == 0:
                valid_bsz -= 1
                continue
            end_idx = nonzero_indices[-1].item()
            score = scores[i][end_idx]
            p = torch.sigmoid(score)
            penalty = (1 - labels[i]) * p ** self.gamma + labels[i] * (1 - p) ** self.gamma
            loss += penalty * torch.nn.functional.binary_cross_entropy_with_logits(score, labels[i])
        if valid_bsz > 0:
            loss = loss / valid_bsz
        return loss


class PointwiseVerifierStrategyForStepPRM(PairwiseVerifierStrategy):
    def trainer_forward(
            self,
            scores: torch.Tensor,
            masks: torch.Tensor,
            labels: torch.Tensor,
            indices: torch.Tensor
    ):
        bsz, seq_len = scores.shape
        valid_bsz = bsz
        loss = torch.tensor(0.).to(scores)
        labels = labels.to(scores)
        for i in range(bsz):
            nonzero_indices = masks[i].nonzero()
            if len(nonzero_indices) == 0:
                valid_bsz -= 1
                continue
            instruct_len = nonzero_indices[0].item()
            index = instruct_len + indices[i][indices[i] > 0]
            index = index[index < seq_len]  # truncate
            assert len(index.shape) == 1
            if len(index) == 0:
                valid_bsz -= 1
                continue
            label = labels[i][: len(index)]
            score = torch.tanh(scores[i])[index]
            loss += ((score - label) ** 2).mean()
        if valid_bsz > 0:
            loss = loss / valid_bsz
        return loss

    def generator_forward(self, scores: torch.Tensor, masks: torch.Tensor, indices: torch.Tensor) -> List[List[float]]:
        scores = scores.detach().cpu()
        bsz, seq_len = scores.shape
        results = []
        for i in range(bsz):
            nonzero_indices = masks[i].nonzero()
            if len(nonzero_indices) == 0:
                print("Warming: instruction len out of range.")
                results.append([])
                continue
            instruct_len = nonzero_indices[0].item()
            index = instruct_len + indices[i][indices[i] > 0]
            index = index[index < seq_len]  # truncate
            if len(index) == 0:
                print("Warming: rating index len out of range.")
                results.append([])
                continue
            score = torch.tanh(scores[i])[index].tolist()
            results.append(score)
        return results


class PointwiseVerifierStrategyForImplicitPRM(PointwiseVerifierStrategy):
    def __init__(self, beta=0.1):
        self.criterion = ImplicitPRMLoss(beta=beta)

    def trainer_forward(
            self,
            logits: torch.Tensor,
            tokens: torch.Tensor,
            masks: torch.Tensor,
            labels: torch.Tensor,
            ref_log_probs: torch.Tensor
    ):
        return self.criterion.forward(
            logits=logits,
            tokens=tokens,
            masks=masks,
            labels=labels,
            ref_log_probs=ref_log_probs
        )

    def generator_forward(
            self,
            logits: torch.Tensor,
            tokens: torch.Tensor,
            ref_log_probs: torch.Tensor,
            masks: torch.Tensor = None
    ) -> List[float]:
        log_probs, ref_log_probs = self.criterion.prepare_for_loss(
            logits=logits,
            labels=tokens,
            masks=masks,
            ref_log_probs=ref_log_probs
        )
        return torch.sigmoid(log_probs - ref_log_probs).tolist()


class PairwiseVerifierStrategyForLastToken(PairwiseVerifierStrategy):
    def trainer_forward(
            self,
            chosen_scores: torch.Tensor,
            rejected_scores: torch.Tensor,
            chosen_masks: torch.Tensor,
            rejected_masks: torch.Tensor,
            **kwargs
    ):
        bsz, seq_len = chosen_scores.shape
        valid_bsz = bsz
        loss = torch.tensor(0.).to(chosen_scores)
        for i in range(bsz):
            chosen_check_start = chosen_masks[i].nonzero()
            rejected_check_start = rejected_masks[i].nonzero()
            if len(chosen_check_start) == 0 or len(rejected_check_start) == 0:
                valid_bsz -= 1
                continue
            chosen_end_idx = chosen_check_start[-1].item()
            rejected_end_idx = rejected_check_start[-1].item()
            chosen_score = chosen_scores[i][chosen_end_idx]
            rejected_score = rejected_scores[i][rejected_end_idx]
            loss += - torch.nn.functional.logsigmoid(chosen_score - rejected_score)
        if valid_bsz > 0:
            loss = loss / valid_bsz
        return loss

    def generator_forward(self, scores: torch.Tensor, masks: torch.Tensor) -> List[float]:
        scores = scores.detach().cpu()
        bsz = scores.shape[0]
        reduce_scores = []
        for i in range(bsz):
            check_end = masks[i].nonzero()
            if len(check_end) == 0:
                print("Warming: instruction len out of range. Setting reward score to 0.")
                reduce_scores.append(0)
                continue
            reduce_scores.append(scores[i][check_end[-1].item()].item())
        return reduce_scores


class PairwiseVerifierStrategyForPGTG(PairwiseVerifierStrategy):
    """ Preference-Grounded Token-level Guidance """
    def trainer_forward(
            self,
            chosen_scores: torch.Tensor,
            rejected_scores: torch.Tensor,
            chosen_masks: torch.Tensor,
            rejected_masks: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        bsz, seq_len = chosen_scores.shape
        valid_bsz = bsz
        loss = 0
        chosen_scores = torch.sigmoid(chosen_scores)
        rejected_scores = torch.sigmoid(rejected_scores)
        for i in range(bsz):
            chosen_check_start = chosen_masks[i].nonzero()
            if len(chosen_check_start) == 0:
                valid_bsz -= 1
                continue
            start_idx = chosen_check_start[0].item()
            rejected_check_start = rejected_masks[i].nonzero()
            assert start_idx == rejected_check_start[0].item()
            chosen_end_idx = chosen_check_start[-1].item() + 1
            rejected_end_idx = rejected_check_start[-1].item() + 1
            c = (chosen_end_idx + rejected_end_idx - 2 * start_idx) / 2
            chosen_score = c * chosen_scores[i][start_idx: chosen_end_idx].mean()
            rejected_score = c * rejected_scores[i][start_idx: rejected_end_idx].mean()
            loss += - torch.nn.functional.logsigmoid(chosen_score - rejected_score)
        if valid_bsz > 0:
            loss = loss / valid_bsz
        return loss

    def generator_forward(self, scores: torch.Tensor, masks: torch.Tensor) -> List[float]:
        scores = scores.detach().cpu()
        bsz = scores.shape[0]
        reduce_scores = []
        scores = torch.sigmoid(scores)
        for i in range(bsz):
            reduce_scores.append(torch.masked_select(scores[i], masks[i]).mean().item())
        return reduce_scores


class PairwiseVerifierStrategyForMeanScore(PairwiseVerifierStrategy):
    def __init__(self, beta: float = 0.2, gamma: float = 2.0):
        self.beta = beta
        self.gamma = gamma

    def trainer_forward(
            self,
            chosen_scores: torch.Tensor,  # shape [batch_size, seq_length]
            rejected_scores: torch.Tensor,
            chosen_masks: torch.Tensor,  # shape [batch_size, seq_length]
            rejected_masks: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = chosen_scores.shape
        valid_bsz = bsz
        loss = 0
        for i in range(bsz):
            chosen_check_start = chosen_masks[i].nonzero()
            if len(chosen_check_start) == 0:
                valid_bsz -= 1
                continue
            start_idx = chosen_check_start[0].item()
            rejected_check_start = rejected_masks[i].nonzero()
            assert start_idx == rejected_check_start[0].item()
            chosen_end_idx = chosen_check_start[-1].item() + 1
            rejected_end_idx = rejected_check_start[-1].item() + 1

            chosen_score = chosen_scores[i][start_idx: chosen_end_idx].mean()
            rejected_score = rejected_scores[i][start_idx: rejected_end_idx].mean()
            loss += - torch.nn.functional.logsigmoid(self.beta * (chosen_score - rejected_score) - self.gamma)
        if valid_bsz > 0:
            loss = loss / valid_bsz
        return loss

    def generator_forward(self, scores: torch.Tensor, masks: torch.Tensor) -> List[float]:
        scores = scores.detach().cpu()
        bsz = scores.shape[0]
        reduce_scores = []
        for i in range(bsz):
            reduce_scores.append(torch.masked_select(scores[i], masks[i]).mean().item())
        return reduce_scores


class PairwiseVerifierStrategyForFocalMeanScore(PairwiseVerifierStrategyForMeanScore):
    """ 每个token位置输出标量分数，取mean pooling，加Focal Loss."""
    def trainer_forward(
            self,
            chosen_scores: torch.Tensor,
            rejected_scores: torch.Tensor,
            chosen_masks: torch.Tensor,
            rejected_masks: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = chosen_scores.shape
        valid_bsz = bsz
        loss = torch.tensor(0.).to(chosen_scores.device)
        for i in range(bsz):
            chosen_check_start = chosen_masks[i].nonzero()
            if len(chosen_check_start) == 0:
                valid_bsz -= 1
                continue
            start_idx = chosen_check_start[0].item()
            rejected_check_start = rejected_masks[i].nonzero()
            assert start_idx == rejected_check_start[0].item()
            chosen_end_idx = chosen_check_start[-1].item() + 1
            rejected_end_idx = rejected_check_start[-1].item() + 1

            chosen_score = chosen_scores[i][start_idx: chosen_end_idx].mean()
            rejected_score = rejected_scores[i][start_idx: rejected_end_idx].mean()
            p_ij = torch.sigmoid(chosen_score - rejected_score)
            loss += -((1.0 - 2.0 * torch.nn.functional.relu(p_ij - 0.5)) ** 2.0) * torch.nn.functional.logsigmoid(
                chosen_score - rejected_score)
        if valid_bsz > 0:
            loss = loss / valid_bsz
        return loss


class PairwiseVerifierStrategyForFocalLoss(PairwiseVerifierStrategyForLastToken):
    """ https://arxiv.org/pdf/2403.17297 """
    def trainer_forward(
            self,
            chosen_scores: torch.Tensor,
            rejected_scores: torch.Tensor,
            chosen_masks: torch.Tensor,
            rejected_masks: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        bsz, seq_len = chosen_scores.shape
        valid_bsz = bsz
        loss = torch.tensor(0.).to(chosen_scores.device)
        for i in range(bsz):
            chosen_check_start = chosen_masks[i].nonzero()
            rejected_check_start = rejected_masks[i].nonzero()
            if len(chosen_check_start) == 0 or len(rejected_check_start) == 0:
                valid_bsz -= 1
                continue
            chosen_end_idx = chosen_check_start[-1].item()
            rejected_end_idx = rejected_check_start[-1].item()
            chosen_score = chosen_scores[i][chosen_end_idx]
            rejected_score = rejected_scores[i][rejected_end_idx]
            p_ij = torch.sigmoid(chosen_score - rejected_score)
            l_rank = -((1.0 - 2.0 * torch.nn.functional.relu(p_ij - 0.5)) ** 2.0) * torch.nn.functional.logsigmoid(
                chosen_score - rejected_score)
            l_penalty_c = -(torch.nn.functional.logsigmoid(chosen_score + 5.0) + torch.nn.functional.logsigmoid(
                5.0 - chosen_score))
            l_penalty_r = -(torch.nn.functional.logsigmoid(rejected_score + 5.0) + torch.nn.functional.logsigmoid(
                5.0 - rejected_score))
            l_penalty = (l_penalty_c + l_penalty_r) / 2.0
            loss += (l_rank + 0.02 * l_penalty)
        if valid_bsz > 0:
            loss = loss / valid_bsz
        return loss


class PairwiseVerifierStrategyForSimPO(PairwiseVerifierStrategy):
    def __init__(self):
        self.criterion = SimPOLoss()

    def trainer_forward(
            self,
            chosen_logits: torch.Tensor,
            rejected_logits: torch.Tensor,
            chosen_labels: torch.Tensor,
            rejected_labels: torch.Tensor,
            chosen_masks: torch.Tensor,
            rejected_masks: torch.Tensor,
    ):
        return self.criterion.forward(
            chosen_logits=chosen_logits,
            rejected_logits=rejected_logits,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
        )

    def generator_forward(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            masks: torch.Tensor = None
    ) -> List[float]:
        log_probs = self.criterion.prepare_for_loss(
            logits=logits,
            labels=labels,
            masks=masks
        )
        return log_probs.tolist()


class PairwiseVerifierStrategyForDPO(PairwiseVerifierStrategy):
    def __init__(self):
        self.criterion = DPOLoss()

    def trainer_forward(
            self,
            chosen_logits: torch.Tensor,
            rejected_logits: torch.Tensor,
            chosen_labels: torch.Tensor,
            rejected_labels: torch.Tensor,
            chosen_masks: torch.Tensor,
            rejected_masks: torch.Tensor,
            ref_chosen_log_probs: torch.Tensor,
            ref_rejected_log_probs: torch.Tensor,
    ):
        return self.criterion.forward(
            chosen_logits=chosen_logits,
            rejected_logits=rejected_logits,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            chosen_masks=chosen_masks,
            rejected_masks=rejected_masks,
            ref_chosen_log_probs=ref_chosen_log_probs,
            ref_rejected_log_probs=ref_rejected_log_probs
        )

    def generator_forward(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            masks: torch.Tensor = None,
            ref_log_probs: torch.Tensor = None,
    ) -> List[float]:
        log_probs, ref_log_probs = self.criterion.prepare_for_loss(
            logits=logits,
            labels=labels,
            masks=masks,
            ref_log_probs=ref_log_probs
        )
        scores = (self.criterion.beta * (log_probs - ref_log_probs)).tolist()
        return scores
