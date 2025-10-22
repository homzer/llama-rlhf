import os.path

import numpy as np

from src.parallel.data_parallel.utils import (
    gather_object_from_data_parallel_region,
    scatter_object_to_data_parallel_region
)
from src.parallel.initialize import (
    get_data_parallel_world_size,
    get_data_parallel_src_rank,
    get_data_parallel_rank,
    set_barrier
)
from src.ppo.buffer import RolloutBuffer


class ParallelRolloutBuffer(RolloutBuffer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def gather_from_data_parallel_region(self):
        assert self.size() > 0
        for key in self.keys():
            self[key] = np.array(gather_object_from_data_parallel_region(self[key].tolist()))

    def scatter_to_data_parallel_region(self):
        assert self.size() > 0
        # ensure divisibility
        world_size = get_data_parallel_world_size()
        if self.size() % world_size != 0:
            self.rearrange(np.concatenate(
                [np.arange(0, self.size()), np.arange(0, world_size - self.size() % world_size)]))

        for key in self.keys():
            self[key] = np.array(scatter_object_to_data_parallel_region(self[key].tolist()))

    def save(self, save_dir: str, overwrite: bool = True):
        if get_data_parallel_src_rank() == 0:
            save_dir = os.path.join(save_dir, "%03d" % get_data_parallel_rank())
            super().save(save_dir=save_dir, overwrite=overwrite)
        set_barrier()

    @classmethod
    def load(cls, buffer_dir: str, start: int = 0, stop: int = None, **kwargs) -> "ParallelRolloutBuffer":
        buffer_dir = os.path.join(buffer_dir, "%03d" % get_data_parallel_rank())
        return ParallelRolloutBuffer(**super().load(buffer_dir=buffer_dir, start=start, stop=stop))
