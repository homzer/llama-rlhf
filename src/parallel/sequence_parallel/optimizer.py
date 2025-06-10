import torch

from src.parallel.initialize import get_sequence_parallel_group, get_sequence_parallel_world_size


class SequenceParallelOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer: torch.optim.Optimizer):
        """
        Wraps an Optimizer and performs an all-reduce operation on the gradients of the parameters before the step.

        Args:
            optimizer (Optimizer): The PyTorch optimizer to wrap.
        """
        self.optimizer = optimizer
        super().__init__(self.optimizer.param_groups, self.optimizer.defaults)
        self.sequence_parallel_world_size = get_sequence_parallel_world_size()

    def all_reduce_grads(self):
        """
        Performs an all-reduce operation on the gradients of all parameters in the optimizer.
        """
        group = get_sequence_parallel_group()
        # Bypass the function if we are using only 1 GPU.
        if torch.distributed.get_world_size(group=group) == 1:
            return

        # All-reduce.
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    # Perform all-reduce on the gradient
                    torch.distributed.all_reduce(param.grad, group=group)
                    # Average the gradient
                    param.grad /= self.sequence_parallel_world_size

    def step(self, closure=None):
        """
        Performs an all-reduce operation on the gradients before executing the optimizer step.
        """
        # Perform all-reduce on the gradients
        self.all_reduce_grads()
        # Call the original optimizer step method
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
