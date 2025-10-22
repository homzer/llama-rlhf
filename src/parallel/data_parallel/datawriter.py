import os
import re

from src.parallel.data_parallel.utils import gather_object_from_data_parallel_region
from src.parallel.initialize import get_data_parallel_src_rank, get_data_parallel_rank


class ParallelDataWriter:
    def __init__(self, file: str, mode: str = 'w', final_gather: bool = True):
        self.global_rank = int(os.environ.get("RANK"))
        self.data_parallel_src_rank = get_data_parallel_src_rank()
        self.worker_id = get_data_parallel_rank()
        self.file = file
        self.worker_file = self.format_file(file)
        self.writer = None
        if self.data_parallel_src_rank == 0:
            self.writer = open(self.worker_file, mode=mode, encoding="utf-8")
        self.final_gather = final_gather

    def format_file(self, file: str) -> str:
        match = re.search(r".+(\..+)$", file)
        if match:
            return re.sub(rf"{match.group(1)}$", f".worker.{self.worker_id}{match.group(1)}", file)
        return f"{file}.worker.{self.worker_id}"

    def __del__(self):
        self.flush()

        if self.final_gather:
            # Gather data from data parallel region
            with open(self.worker_file, mode='r', encoding="utf-8") as reader:
                s = [line.strip() for line in reader]
            ss = gather_object_from_data_parallel_region(s)
            if self.global_rank == 0:
                with open(self.file, "w", encoding="utf-8") as writer:
                    for s in ss:
                        writer.write(s + '\n')

        if self.writer:
            self.writer.close()

    def flush(self):
        if self.writer:
            self.writer.flush()

    def write(self, s: str, flush: bool = False):
        if self.writer:
            self.writer.write(s)
            if flush:
                self.writer.flush()
