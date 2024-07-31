import os

from fairscale.nn.model_parallel.initialize import get_data_parallel_rank, get_data_parallel_world_size
from torch.utils.data import DistributedSampler, DataLoader, Dataset

from src.utils import set_barrier, get_data_parallel_src_rank


class ParallelDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_data_parallel_world_size(),
            rank=get_data_parallel_rank(),
            shuffle=shuffle
        )
        super().__init__(dataset, batch_size=batch_size, sampler=sampler)


class ParallelDataWriter:
    def __init__(self, file: str, mode: str = 'w'):
        self.world_size = int(os.environ.get("WORLD_SIZE"))
        self.data_parallel_src_rank = get_data_parallel_src_rank()
        self.data_parallel_rank = get_data_parallel_rank()
        self.writer = open(file, mode=mode, encoding="utf-8")

    def __del__(self):
        self.writer.close()

    def write(self, s: str):
        for i in range(self.world_size):  # Sequentially write
            if i == self.data_parallel_rank:
                if self.data_parallel_rank == self.data_parallel_src_rank:
                    self.writer.write(s)
                    self.writer.flush()
            set_barrier()
