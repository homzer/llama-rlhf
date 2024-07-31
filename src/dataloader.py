from fairscale.nn.model_parallel.initialize import get_data_parallel_rank, get_data_parallel_world_size
from torch.utils.data import DistributedSampler, DataLoader, Dataset


class ParallelDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_data_parallel_world_size(),
            rank=get_data_parallel_rank(),
            shuffle=shuffle
        )
        super().__init__(dataset, batch_size=batch_size, sampler=sampler)
