import torch

from src.parallel.data_parallel.optimizer import DataParallelOptimizer
from src.parallel.sequence_parallel.optimizer import SequenceParallelOptimizer


class ParallelOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer: torch.optim.Optimizer):
        """
        Wraps an Optimizer and performs an all-reduce operation on the gradients of the parameters before the step.
        Supports sequence parallel and data parallel training.
        Args:
            optimizer (Optimizer): The PyTorch optimizer to wrap.
        """
        self.optimizer = SequenceParallelOptimizer(
            DataParallelOptimizer(optimizer)
        )
        super().__init__(self.optimizer.param_groups, self.optimizer.defaults)

    def step(self, closure=None):
        self.optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        """
        Clears the gradients of the optimizer.
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """
        Returns the state dictionary of the optimizer.
        """
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """
        Loads the state dictionary into the optimizer.
        """
        self.optimizer.load_state_dict(state_dict)
