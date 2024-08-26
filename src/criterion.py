from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import powmax, masked_mean


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class KLDivLoss(Loss):
    def __init__(self, eps=7e-5, return_scalar: bool = True):
        super().__init__()
        self.eps = eps
        self.return_scalar = return_scalar

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            masks: torch.Tensor = None,
            temperature: float = 1.0,
            targets_after_softmax: bool = False
    ):
        """
        Compute KL-Divergence loss.
        :param temperature: Temperature, default to be 1.
        :param logits: the logits of the estimated distribution, before `softmax`
        :param targets: the target logits.
        :param masks: Optional. For masked selection.
        :param targets_after_softmax: whether the targets are summed up to be 1.
        Shape is identical to the shape of `logits` up to last dim.
        :return: scalar loss.
        """
        bzs = logits.shape[0]
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1, targets.size(-1)).to(logits)
        if targets_after_softmax:
            loss = (targets.float() * (
                torch.log(torch.clamp(targets.float(), min=1e-12)) - torch.log_softmax(logits.float(), dim=-1)
            )).type_as(logits)
        else:
            loss = (torch.softmax(targets.float(), dim=-1) * (
                torch.log_softmax(targets.float(), dim=-1) - torch.log_softmax(logits.float(), dim=-1)
            )).type_as(logits)
        loss = torch.sum(loss, dim=-1)
        if self.return_scalar:
            if masks is not None:
                masks = masks.view(-1).to(logits.device)
                loss = torch.masked_select(loss, masks)
            return loss.mean()
        else:
            if masks is not None:
                masks = masks.view(-1).to(logits.device)
                loss = loss * masks
            return loss.view(bzs, -1)  # [b, s]


class ReverseKLDivLoss(KLDivLoss):
    def __init__(self, eps=7e-5, return_scalar: bool = True):
        super().__init__(eps=eps, return_scalar=return_scalar)

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            masks: torch.Tensor = None,
            temperature: float = 1.0
    ):
        bzs = logits.shape[0]
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1, targets.size(-1)).to(logits)
        estimates = torch.softmax(logits.float(), dim=-1).type_as(logits)
        targets = torch.softmax(targets.float() / temperature, dim=-1).type_as(targets)
        estimates = powmax(estimates + self.eps)
        targets = powmax(targets + self.eps)

        loss = estimates * (torch.log(estimates) - torch.log(targets))
        loss = torch.sum(loss, dim=-1)
        if self.return_scalar:
            if masks is not None:
                masks = masks.view(-1).to(logits.device)
                loss = torch.masked_select(loss, masks)
            return loss.mean()
        else:
            if masks is not None:
                masks = masks.view(-1).to(logits.device)
                loss = loss * masks
            return loss.view(bzs, -1)  # [b, s]


class JSDivLoss(KLDivLoss):
    def __init__(self, eps=7e-5, return_scalar: bool = True):
        super().__init__(eps=eps, return_scalar=return_scalar)

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            masks: torch.Tensor = None,
            temperature: float = 1.0
    ):
        bzs = logits.shape[0]
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1, targets.size(-1)).to(logits)
        estimates = torch.softmax(logits.float(), dim=-1).type_as(logits)
        targets = torch.softmax(targets.float() / temperature, dim=-1).type_as(targets)
        estimates = powmax(estimates + self.eps)
        targets = powmax(targets + self.eps)
        mediates = 0.5 * (targets + estimates)

        loss = 0.5 * targets * torch.log(targets / mediates) + 0.5 * estimates * torch.log(estimates / mediates)
        loss = torch.sum(loss, dim=-1)
        if self.return_scalar:
            if masks is not None:
                masks = masks.view(-1).to(logits.device)
                loss = torch.masked_select(loss, masks)
            return loss.mean()
        else:
            if masks is not None:
                masks = masks.view(-1).to(logits.device)
                loss = loss * masks
            return loss.view(bzs, -1)  # [b, s]


class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            masks: torch.Tensor = None,
    ):
        loss = (logits - targets) ** 2
        loss = masked_mean(loss, masks)
        return loss.mean()


class PairwiseScoreLoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            chosen_rewards: torch.Tensor,
            rejected_rewards: torch.Tensor,
            chosen_masks: torch.Tensor = None,
            rejected_masks: torch.Tensor = None
    ):
        bzs = chosen_rewards.shape[0]
        chosen_rewards = chosen_rewards.view(bzs, -1)
        rejected_rewards = rejected_rewards.view(bzs, -1)
        if chosen_masks is not None:
            chosen_masks = chosen_masks.view(bzs, -1)
        if rejected_masks is None:
            rejected_masks = rejected_masks.view(bzs, -1)

        c_rewards = masked_mean(chosen_rewards, chosen_masks, dim=-1)  # [b]
        r_rewards = masked_mean(rejected_rewards, rejected_masks, dim=-1)  # [b]

        loss = - torch.log(torch.sigmoid(c_rewards - r_rewards)).mean()
        return loss


class LastTokenScoreLoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            scores: torch.Tensor,  # [b, s]
            masks: torch.Tensor,  # [b, s]
            labels: Union[torch.Tensor, List[int]]  # [b]
    ):
        bzs = scores.shape[0]
        loss = 0
        for i in range(bzs):
            if len(masks[i].nonzero()) == 0:
                continue
            last_token_id = masks[i].nonzero()[-1].item()
            score = torch.sigmoid(scores[i][last_token_id])
            loss += (score - labels[i]) ** 2
        return loss / bzs


class SimPoLoss(Loss):
    def __init__(self, beta: float = 2.0, gamma: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def prepare_for_loss(self, logits, labels, masks):
        log_probs = torch.log_softmax(
            logits.float() if logits.dtype == torch.float16 else logits, dim=-1
        ).type_as(logits)
        labels = labels.to(logits.device).long()
        labels[labels == -100] = 0
        # [b, s]
        log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        if masks is None:
            masks = torch.ones_like(log_probs)
        masks = masks.to(logits.device)
        log_probs = (log_probs * masks).sum(-1) / (masks.sum(-1) + self.eps)  # [b]
        return log_probs

    def forward(
            self,
            chosen_logits: torch.Tensor,
            rejected_logits: torch.Tensor,
            chosen_labels: torch.Tensor,
            rejected_labels: torch.Tensor,
            chosen_masks: torch.Tensor = None,
            rejected_masks: torch.Tensor = None,
    ):
        chosen_log_probs = self.prepare_for_loss(
            logits=chosen_logits,
            labels=chosen_labels,
            masks=chosen_masks,
        )

        rejected_log_probs = self.prepare_for_loss(
            logits=rejected_logits,
            labels=rejected_labels,
            masks=rejected_masks,
        )

        loss = - F.logsigmoid(self.beta * (chosen_log_probs - rejected_log_probs) - self.gamma)
        return loss.mean()


class DpoLoss(Loss):
    def __init__(self, beta=0.1, logits_norm: bool = False, label_smoothing: float = 0.0, eps=1e-5):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.eps = eps
        self.logits_norm = logits_norm

    def prepare_for_loss(self, logits, labels, masks, ref_log_probs=None, ref_logits=None):
        logits = self._norm(logits) if self.logits_norm else logits
        log_probs = torch.log_softmax(
            logits.float() if logits.dtype == torch.float16 else logits, dim=-1
        ).type_as(logits)
        labels = labels.to(logits.device).long()
        labels[labels == -100] = 0
        # [b, s]
        log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        if masks is None:
            masks = torch.ones_like(log_probs)
        masks = masks.to(logits.device)
        log_probs = (log_probs * masks).sum(-1)  # / (masks.sum(-1) + self.eps)

        if ref_log_probs is None and ref_logits is not None:
            ref_logits = ref_logits.to(logits)
            ref_logits = self._norm(ref_logits) if self.logits_norm else ref_logits
            ref_log_probs = torch.log_softmax(
                ref_logits.float() if ref_logits.dtype == torch.float16 else ref_logits, dim=-1
            ).type_as(ref_logits)
            ref_log_probs = torch.gather(ref_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            # NaN might appear because the logits chosen by the label might be negative infinity.
            ref_log_probs = torch.clamp(ref_log_probs, min=-1e5, max=1e5)
        if ref_log_probs is not None:
            ref_log_probs = ref_log_probs.to(logits)
            ref_log_probs = (ref_log_probs * masks).sum(-1)  # / (masks.sum(-1) + self.eps)
        else:
            ref_log_probs = 0.0

        return log_probs, ref_log_probs

    def _norm(self, x: torch.Tensor, dim: int = -1):
        return x / (x.std(dim=dim, keepdim=True) + self.eps)

    def forward(
            self,
            chosen_logits: torch.Tensor,
            rejected_logits: torch.Tensor,
            chosen_labels: torch.Tensor,
            rejected_labels: torch.Tensor,
            chosen_masks: torch.Tensor = None,
            rejected_masks: torch.Tensor = None,
            ref_chosen_log_probs: torch.Tensor = None,
            ref_rejected_log_probs: torch.Tensor = None,
            ref_chosen_logits: torch.Tensor = None,
            ref_rejected_logits: torch.Tensor = None,
    ):
        """
        Compute Dpo loss.
        :param chosen_logits: [batch_size, seq_len, vocab_size] from policy model.
        :param rejected_logits: [batch_size, seq_len, vocab_size] from policy model.
        :param chosen_labels: [batch_size, seq_len], chosen token ids.
        :param rejected_labels: [batch_size, seq_len], rejected token ids.
        :param chosen_masks: [batch_size, seq_len] with values of `True` or `False`.
        :param rejected_masks: [batch_size, seq_len] with values of `True` or `False`.
        :param ref_chosen_logits: [batch_size, seq_len, vocab_size] from reference model.
        :param ref_rejected_logits: [batch_size, seq_len, vocab_size] from reference model.
        :param ref_chosen_log_probs: [batch_size, seq_len] from reference model. If not provided, computed from ref_chosen_logits.
        :param ref_rejected_log_probs: [batch_size, seq_len] from reference model. If not provided, computed from ref_rejected_logits.
        :return: Scalar loss tensor.
        """
        assert not ((ref_chosen_logits is None) ^ (ref_rejected_logits is None))

        chosen_log_probs, ref_chosen_log_probs = self.prepare_for_loss(
            logits=chosen_logits,
            labels=chosen_labels,
            masks=chosen_masks,
            ref_log_probs=ref_chosen_log_probs,
            ref_logits=ref_chosen_logits
        )

        rejected_log_probs, ref_rejected_log_probs = self.prepare_for_loss(
            logits=rejected_logits,
            labels=rejected_labels,
            masks=rejected_masks,
            ref_log_probs=ref_rejected_log_probs,
            ref_logits=ref_rejected_logits
        )

        log_probs = (chosen_log_probs - rejected_log_probs) - (ref_chosen_log_probs - ref_rejected_log_probs)
        # (chosen_log_probs - ref_chosen_log_probs) - (rejected_log_probs - ref_rejected_log_probs)
        loss = (
                - F.logsigmoid(self.beta * log_probs) * (1 - self.label_smoothing)
                - F.logsigmoid(- self.beta * log_probs) * self.label_smoothing
        )
        return loss.mean()


def norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-5):
    return x / (x.std(dim=dim, keepdim=True) + eps)


if __name__ == '__main__':
    torch.manual_seed(0)
    criterion = DpoLoss(logits_norm=False)
    _chosen_logits = torch.Tensor([
        [[1, 0, 100, -100], [1, 100, 0, -100], [0, 1, 100, -100]],
        [[-100, 1, 0, 100], [0, 1, 100, -100], [100, 1, 0, -100]]
    ])
    _rejected_logits = torch.Tensor([
        [[1, 0, -100, 100], [1, 100, 0, -100], [0, 1, 100, -100]],
        [[-0, 1, 100, -100], [0, 1, 100, -100], [100, 1, 0, -100]]
    ])
    # _chosen_logits, _rejected_logits = norm(_chosen_logits), norm(_rejected_logits)
    _chosen_labels = torch.Tensor([[2, 1, 2],
                                   [3, 2, -100]])
    _rejected_labels = torch.Tensor([[3, 1, -100],
                                     [2, -100, -100]])
    _chosen_masks = _chosen_labels != -100
    _rejected_masks = _rejected_labels != -100
    _reference_chosen_logits = - torch.Tensor([
        [[1, 0, -50, 1000], [1, -50, 0, 1000], [0, 1, -50, 500]],
        [[500, 5, 0, -500], [0, 1, -500, 100], [50, 10, 0, -100]]
    ])
    _reference_rejected_logits = torch.Tensor([
        [[10, 0, 50, -500], [1, -50, 0, 50], [0, 1, -500, 50]],
        [[-0, 1, -50, 50], [0, 10, 50, -50], [50, 1, 0, -50]]
    ])
    # _reference_chosen_logits, _reference_rejected_logits = norm(_reference_chosen_logits), norm(_reference_rejected_logits)
    print(criterion.forward(
        _chosen_logits, _rejected_logits, _chosen_labels, _rejected_labels, _chosen_masks, _rejected_masks,
        _reference_chosen_logits, _reference_rejected_logits
    ))
    # _chosen_masks = torch.tensor([[False, False, False, False, True]])
    # _rejected_masks = torch.tensor([[True, True, True, True, False]])
    # print(_chosen_rewards * _chosen_masks)
    # _loss = criterion.forward(_chosen_rewards, _rejected_rewards, _chosen_masks, _rejected_masks)
    # print(_loss)
