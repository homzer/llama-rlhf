from fairscale.nn.model_parallel.initialize import get_data_parallel_world_size, get_data_parallel_rank
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class ParallelDataLoader(DataLoader):
    """ Usage: dataloader = ParallelDataLoader(dataset, batch_size=batch_size) """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_data_parallel_world_size(),
            rank=get_data_parallel_rank(),
            shuffle=shuffle
        )
        super().__init__(dataset, batch_size=batch_size, sampler=sampler)
