import os
import re

from fairscale.nn.model_parallel.initialize import get_model_parallel_src_rank, get_model_parallel_world_size

from src.parallel.utils import set_model_parallel_barrier


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

    def write(self, s: str, flush: bool = False):
        if self.global_rank == self.model_parallel_src_rank:
            self.writer.write(s)
            if flush:
                self.writer.flush()
        set_model_parallel_barrier()
