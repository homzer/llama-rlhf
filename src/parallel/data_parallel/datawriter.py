import os
import re
from collections.abc import Callable

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

    def filter_unprocessed_data(self, datalist: list, key_extractor: Callable) -> list:
        """
        Filters out already processed data and returns the list of unprocessed data.

        This method implements checkpoint resumption for generation tasks. It reads processed
        records from a worker file, compares them with the current data list, and filters
        out data items that have not been processed yet.

        Args:
            datalist: List of data to be processed
            key_extractor: Function to extract a unique identifier from a data item

        Returns:
            List of unprocessed data
        """
        processed_keys = []
        if os.path.exists(self.worker_file):
            with open(self.worker_file, mode='r', encoding="utf-8") as reader:
                processed_keys = [key_extractor(line.strip()) for line in reader]
        processed_keys = gather_object_from_data_parallel_region(processed_keys)
        print(f"Number of processed data {len(processed_keys)}")
        unprocessed_data = []
        for data in datalist:
            if key_extractor(data) not in processed_keys:
                unprocessed_data.append(data)
        print(f"Number of unprocessed data {len(unprocessed_data)}")
        return unprocessed_data


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
