import collections
from typing import List, Union

import torch

from src.rewards.strategy import (
    PairwiseVerifierStrategyForLastToken,
    PairwiseVerifierStrategyForMeanScore,
    PairwiseVerifierStrategyForFocalMeanScore,
    PairwiseVerifierStrategyForFocalLoss,
    PairwiseVerifierStrategyForDPO,
    PairwiseVerifierStrategyForSimPO,
    PointwiseVerifierStrategyForFocalLoss,
    PointwiseVerifierStrategyForLastToken,
    PointwiseVerifierStrategyForImplicitPRM,
    PointwiseVerifierStrategyForStepPRM
)
from src.generator import GeneratorForVerifier
from src.models.modeling import ParallelVerifier, Verifier, ModelForCausalLM, ParallelModelForCausalLM
from src.ppo.generator import LogitsGeneratorForCausalLM
from src.tokenizers.tokenizer import Tokenizer


class PointwiseVerifierGeneratorForLastToken(GeneratorForVerifier):
    def __init__(
            self,
            model: Union[Verifier, ParallelVerifier],
            tokenizer: Tokenizer,
            max_seq_len: int,
    ):
        super().__init__(model=model, tokenizer=tokenizer, max_seq_len=max_seq_len)
        self.strategy = PointwiseVerifierStrategyForLastToken()

    def forward(self, instructions: Union[List[str], List[List[int]]], outputs: Union[List[str], List[List[int]]]):
        self.model.eval()
        examples = self.prepare_for_generation(instructions, outputs)
        with torch.no_grad():
            scores = self.model.forward(examples.tokens).scores
        Outputs = collections.namedtuple("Outputs", ["scores"])
        return Outputs(scores=self.strategy.generator_forward(scores, examples.masks))


class PointwiseVerifierGeneratorForFocalLoss(PointwiseVerifierGeneratorForLastToken):
    def __init__(
            self,
            model: Union[Verifier, ParallelVerifier],
            tokenizer: Tokenizer,
            max_seq_len: int,
    ):
        super().__init__(model=model, tokenizer=tokenizer, max_seq_len=max_seq_len)
        self.strategy = PointwiseVerifierStrategyForFocalLoss()


class PointwiseVerifierGeneratorForImplicitPRM:
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int
    ):
        super().__init__()
        self.generator = LogitsGeneratorForCausalLM(model, tokenizer, max_seq_len)
        self.strategy = PointwiseVerifierStrategyForImplicitPRM()

    def forward(
            self,
            instructions: List[str],
            outputs: List[str],
            reference_log_probs: torch.Tensor
    ):
        examples = self.generator.prepare_for_forward(instructions, outputs)
        logits = self.generator.model_forward(examples.tokens).logits
        scores = self.strategy.generator_forward(
            logits=logits,
            tokens=examples.labels,
            masks=examples.masks,
            ref_log_probs=reference_log_probs
        )
        Outputs = collections.namedtuple("Outputs", ["scores"])
        return Outputs(scores=scores)


class PointwiseVerifierGeneratorForStepPRM(GeneratorForVerifier):
    def __init__(
            self,
            model: Union[Verifier, ParallelVerifier],
            tokenizer: Tokenizer,
            max_seq_len: int,
    ):
        super().__init__(model=model, tokenizer=tokenizer, max_seq_len=max_seq_len)
        self.strategy = PointwiseVerifierStrategyForStepPRM()

    def forward(self, instructions, outputs, indices):
        self.model.eval()
        examples = self.prepare_for_generation(instructions, outputs)
        with torch.no_grad():
            scores = self.model.forward(examples.tokens).scores
        Outputs = collections.namedtuple("Outputs", ["scores"])
        return Outputs(scores=self.strategy.generator_forward(scores, examples.masks, indices))


class VerifierGeneratorForLastToken(GeneratorForVerifier):
    def __init__(
            self,
            model: Union[Verifier, ParallelVerifier],
            tokenizer: Tokenizer,
            max_seq_len: int,
    ):
        super().__init__(model=model, tokenizer=tokenizer, max_seq_len=max_seq_len)
        self.strategy = PairwiseVerifierStrategyForLastToken()

    def forward(self, instructions: Union[List[str], List[List[int]]], outputs: Union[List[str], List[List[int]]]):
        self.model.eval()
        examples = self.prepare_for_generation(instructions, outputs)
        with torch.no_grad():
            scores = self.model.forward(examples.tokens).scores
        Outputs = collections.namedtuple("Outputs", ["scores"])
        return Outputs(scores=self.strategy.generator_forward(scores, examples.masks))


class VerifierGeneratorForMeanScores(VerifierGeneratorForLastToken):
    def __init__(
            self,
            model: Union[Verifier, ParallelVerifier],
            tokenizer: Tokenizer,
            max_seq_len: int,
    ):
        super().__init__(model=model, tokenizer=tokenizer, max_seq_len=max_seq_len)
        self.strategy = PairwiseVerifierStrategyForMeanScore()


class VerifierGeneratorForFocalMeanScores(VerifierGeneratorForLastToken):
    def __init__(
            self,
            model: Union[Verifier, ParallelVerifier],
            tokenizer: Tokenizer,
            max_seq_len: int,
    ):
        super().__init__(model=model, tokenizer=tokenizer, max_seq_len=max_seq_len)
        self.strategy = PairwiseVerifierStrategyForFocalMeanScore()


class VerifierGeneratorForFocalLoss(VerifierGeneratorForLastToken):
    def __init__(
            self,
            model: Union[Verifier, ParallelVerifier],
            tokenizer: Tokenizer,
            max_seq_len: int,
    ):
        super().__init__(model=model, tokenizer=tokenizer, max_seq_len=max_seq_len)
        self.strategy = PairwiseVerifierStrategyForFocalLoss()


class VerifierGeneratorForSimPO:
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int
    ):
        self.generator = LogitsGeneratorForCausalLM(model, tokenizer, max_seq_len)
        self.strategy = PairwiseVerifierStrategyForSimPO()

    def forward(
            self,
            instructions: List[str],
            outputs: List[str],
    ):
        examples = self.generator.prepare_for_forward(instructions, outputs)
        logits = self.generator.model_forward(examples.tokens).logits
        scores = self.strategy.generator_forward(
            logits=logits,
            labels=examples.labels,
            masks=examples.masks
        )
        Outputs = collections.namedtuple("Outputs", ["scores"])
        return Outputs(scores=scores)


class VerifierGeneratorForDPO:
    def __init__(
            self,
            model: Union[ModelForCausalLM, ParallelModelForCausalLM],
            tokenizer: Tokenizer,
            max_seq_len: int
    ):
        super().__init__()
        self.generator = LogitsGeneratorForCausalLM(model, tokenizer, max_seq_len)
        self.strategy = PairwiseVerifierStrategyForDPO()

    def forward(
            self,
            instructions: List[str],
            outputs: List[str],
            reference_log_probs: torch.Tensor
    ):
        examples = self.generator.prepare_for_forward(instructions, outputs)
        logits = self.generator.model_forward(examples.tokens).logits
        scores = self.strategy.generator_forward(
            logits=logits,
            labels=examples.labels,
            masks=examples.masks,
            ref_log_probs=reference_log_probs
        )
        Outputs = collections.namedtuple("Outputs", ["scores"])
        return Outputs(scores=scores)
