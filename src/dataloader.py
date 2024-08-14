import os
import re

from fairscale.nn.model_parallel.initialize import get_data_parallel_rank, get_data_parallel_world_size, \
    get_model_parallel_src_rank, get_model_parallel_world_size
from torch.utils.data import DistributedSampler, DataLoader, Dataset

from src.utils import set_model_parallel_barrier


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
        self.global_rank = int(os.environ.get("RANK"))
        self.model_parallel_src_rank = get_model_parallel_src_rank()
        self.model_parallel_world_size = get_model_parallel_world_size()
        self.worker_id = self.model_parallel_src_rank // self.model_parallel_world_size
        self.file = self.format_file(file)
        self.writer = open(self.file, mode=mode, encoding="utf-8")

    def format_file(self, file: str) -> str:
        match = re.search(r".+(\..+)$", file)
        if match:
            return re.sub(rf"{match.group(1)}$", f".worker.{self.worker_id}{match.group(1)}", file)
        return f"{file}.worker.{self.worker_id}"

    def __del__(self):
        self.writer.close()

    def write(self, s: str):
        if self.global_rank == self.model_parallel_src_rank:
            self.writer.write(s)
            self.writer.flush()
        set_model_parallel_barrier()
