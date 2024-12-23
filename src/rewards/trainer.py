import collections
from typing import List

import torch

from src.rewards.strategy import PairwiseVerifierStrategyForLastToken, \
    PairwiseVerifierStrategyForMeanScore, PairwiseVerifierStrategyForFocalMeanScore, \
    PairwiseVerifierStrategyForFocalLoss, PairwiseVerifierStrategyForDPO, PairwiseVerifierStrategyForSimPO
from src.models.modeling import ParallelVerifier, ParallelModelForCausalLM
from src.tokenizers import Tokenizer
from src.trainer import ParallelVerifierTrainer, ParallelSolverTrainer


class ParallelVerifierTrainerForLastToken(ParallelVerifierTrainer):
    def __init__(
            self,
            model: ParallelVerifier,
            tokenizer: Tokenizer,
            optimizer: torch.optim.Optimizer,
            accumulation_steps: int = 1
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            accumulation_steps=accumulation_steps
        )
        self.strategy = PairwiseVerifierStrategyForLastToken()
        self.predictions = []

    def forward(self, instructions: List[str], chosen: List[str], rejected: List[str]):
        self.model.train()
        bsz = len(instructions)
        instructions = [*instructions, *instructions]
        outputs = [*chosen, *rejected]
        examples = self.prepare_for_training(instructions, outputs)
        scores = self.model.forward(examples.tokens).scores

        loss = self.strategy.trainer_forward(
            chosen_scores=scores[:bsz],
            rejected_scores=scores[bsz:],
            chosen_masks=examples.masks[:bsz],
            rejected_masks=examples.masks[bsz:]
        )

        if loss != 0:
            self._back_propagation(loss)

        chosen_scores = self.strategy.generator_forward(
            scores=scores[:bsz],
            masks=examples.masks[:bsz]
        )
        rejected_scores = self.strategy.generator_forward(
            scores=scores[bsz:],
            masks=examples.masks[bsz:]
        )
        self.predictions.extend([cs > rs for cs, rs in zip(chosen_scores, rejected_scores)])

        Output = collections.namedtuple('Output', ['loss'])
        return Output(loss=loss.item() if isinstance(loss, torch.Tensor) else loss)

    def verifier_accuracy(self) -> float:
        accuracy = sum(self.predictions) / len(self.predictions)
        self.predictions = []
        return accuracy


class ParallelVerifierTrainerForMeanScore(ParallelVerifierTrainerForLastToken):
    def __init__(
            self,
            model: ParallelVerifier,
            tokenizer: Tokenizer,
            optimizer: torch.optim.Optimizer,
            accumulation_steps: int = 1,
            beta: float = 1.0,
            margin: float = 0.0,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            accumulation_steps=accumulation_steps
        )
        self.strategy = PairwiseVerifierStrategyForMeanScore(beta=beta, margin=margin)


class ParallelVerifierTrainerForFocalMeanScore(ParallelVerifierTrainerForLastToken):
    def __init__(
            self,
            model: ParallelVerifier,
            tokenizer: Tokenizer,
            optimizer: torch.optim.Optimizer,
            accumulation_steps: int = 1
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            accumulation_steps=accumulation_steps
        )
        self.strategy = PairwiseVerifierStrategyForFocalMeanScore()


class ParallelVerifierTrainerForFocalLoss(ParallelVerifierTrainerForLastToken):
    def __init__(
            self,
            model: ParallelVerifier,
            tokenizer: Tokenizer,
            optimizer: torch.optim.Optimizer,
            accumulation_steps: int = 1
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            accumulation_steps=accumulation_steps
        )
        self.strategy = PairwiseVerifierStrategyForFocalLoss()


class ParallelVerifierTrainerForSimPO(ParallelSolverTrainer):
    def __init__(
            self,
            model: ParallelModelForCausalLM,
            tokenizer: Tokenizer,
            optimizer: torch.optim.Optimizer,
            max_seq_len: int,
            accumulation_steps: int = 1
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            max_seq_len=max_seq_len,
            accumulation_steps=accumulation_steps
        )
        self.strategy = PairwiseVerifierStrategyForSimPO()
        self.predictions = []

    def forward(self, instructions: List[str], chosen: List[str], rejected: List[str]):
        bsz = len(instructions)
        instructions = [*instructions, *instructions]
        outputs = [*chosen, *rejected]
        examples = self.prepare_for_training(instructions, outputs)

        logits = self.model.forward(examples.tokens).logits
        loss = self.strategy.trainer_forward(
            chosen_logits=logits[:bsz],
            rejected_logits=logits[bsz:],
            chosen_labels=examples.labels[:bsz],
            rejected_labels=examples.labels[bsz:],
            chosen_masks=examples.masks[:bsz],
            rejected_masks=examples.masks[bsz:],
        )
        self._back_propagation(loss)

        chosen_scores = self.strategy.generator_forward(
            logits=logits[:bsz],
            labels=examples.labels[:bsz],
            masks=examples.masks[:bsz]
        )
        rejected_scores = self.strategy.generator_forward(
            logits=logits[bsz:],
            labels=examples.labels[bsz:],
            masks=examples.masks[bsz:]
        )
        self.predictions.extend([cs > rs for cs, rs in zip(chosen_scores, rejected_scores)])

        Output = collections.namedtuple('Output', ['logits', 'loss'])
        return Output(logits=logits[:bsz], loss=loss)

    def verifier_accuracy(self) -> float:
        accuracy = sum(self.predictions) / len(self.predictions)
        self.predictions = []
        return accuracy


class ParallelVerifierTrainerForDPO(ParallelSolverTrainer):
    def __init__(
            self,
            model: ParallelModelForCausalLM,
            tokenizer: Tokenizer,
            optimizer: torch.optim.Optimizer,
            max_seq_len: int,
            accumulation_steps: int = 1
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            max_seq_len=max_seq_len,
            accumulation_steps=accumulation_steps
        )
        self.strategy = PairwiseVerifierStrategyForDPO()
        self.predictions = []

    def forward(
            self,
            instructions: List[str],
            chosen: List[str],
            rejected: List[str],
            reference_chosen_log_probs: torch.Tensor,
            reference_rejected_log_probs: torch.Tensor
    ):
        bsz = len(instructions)
        instructions = [*instructions, *instructions]
        outputs = [*chosen, *rejected]
        examples = self.prepare_for_training(instructions, outputs)

        logits = self.model.forward(examples.tokens).logits
        loss = self.strategy.trainer_forward(
            chosen_logits=logits[:bsz],
            rejected_logits=logits[bsz:],
            chosen_labels=examples.labels[:bsz],
            rejected_labels=examples.labels[bsz:],
            chosen_masks=examples.masks[:bsz],
            rejected_masks=examples.masks[bsz:],
            ref_chosen_log_probs=reference_chosen_log_probs,
            ref_rejected_log_probs=reference_rejected_log_probs,
        )
        self._back_propagation(loss)

        chosen_scores = self.strategy.generator_forward(
            logits=logits[:bsz],
            labels=examples.labels[:bsz],
            masks=examples.masks[:bsz]
        )
        rejected_scores = self.strategy.generator_forward(
            logits=logits[bsz:],
            labels=examples.labels[bsz:],
            masks=examples.masks[bsz:]
        )
        self.predictions.extend([cs > rs for cs, rs in zip(chosen_scores, rejected_scores)])

        Output = collections.namedtuple('Output', ['logits', 'loss'])
        return Output(logits=logits[:bsz], loss=loss)

    def verifier_accuracy(self) -> float:
        accuracy = sum(self.predictions) / len(self.predictions)
        self.predictions = []
        return accuracy
