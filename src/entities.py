import math
import time
from typing import List

import torch


class SlimLogits:
    def __init__(self, logits: torch.Tensor = None, n=5):
        self.n = n
        self.vocab_size = None
        self.max_seq_len = None
        self.values = None
        self.indices = None
        if logits is not None:
            assert len(logits.shape) == 3
            self.max_seq_len = logits.shape[1]
            self.vocab_size = logits.shape[2]
            self._set(logits)

    def _set(self, logits: torch.Tensor):
        (batch_size, seq_len, vocab_size) = logits.shape
        assert self.vocab_size == vocab_size
        if logits.device.type == "cpu":
            logits = logits.float()  # topk is not implemented for half on cpu
        values, indices = torch.topk(logits, k=self.n)
        self.values = values.detach().cpu()  # [b, s, n]
        self.indices = indices.detach().cpu()  # [b, s, n]

    def extend(self, slim_logits: "SlimLogits"):
        """ Batch extend. """
        if self.vocab_size is None:
            self.vocab_size = slim_logits.vocab_size
        else:
            assert self.vocab_size == slim_logits.vocab_size
        if self.max_seq_len is None:
            self.max_seq_len = slim_logits.max_seq_len
        else:
            assert self.max_seq_len == slim_logits.max_seq_len

        self.values = slim_logits.values if (
                self.values is None
        ) else torch.stack([*self.values, *slim_logits.values])
        # torch.cat([self.values, slim_logits.values], dim=0)
        self.indices = slim_logits.indices if (
                self.indices is None
        ) else torch.stack([*self.indices, *slim_logits.indices])
        # torch.cat([self.indices, slim_logits.indices], dim=0)

    def __len__(self):
        if self.values is not None:
            return len(self.values)
        return 0

    def __getitem__(self, i) -> "SlimLogits":
        slim_logits = SlimLogits()
        slim_logits.max_seq_len = self.max_seq_len
        slim_logits.n = self.n
        slim_logits.vocab_size = self.vocab_size
        slim_logits.values = self.values[None, i]
        slim_logits.indices = self.indices[None, i]
        return slim_logits

    def to_dict(self) -> dict:
        return {
            "max_seq_len": self.max_seq_len,
            "n": self.n,
            "vocab_size": self.vocab_size,
            "values": self.values.tolist(),
            "indices": self.indices.tolist()
        }

    def from_dict(self, data: List[dict]) -> "SlimLogits":
        if isinstance(data, dict):
            data = [data]
        self.max_seq_len = data[0]['max_seq_len']
        self.n = data[0]["n"]
        self.vocab_size = data[0]["vocab_size"]
        self.values = torch.stack([torch.tensor(d["values"]).squeeze(0) for d in data])
        self.indices = torch.stack([torch.tensor(d["indices"]).squeeze(0) for d in data])
        return self

    def fetch(self, i: int) -> torch.Tensor:
        assert 0 <= i < len(self), "Index out of range error"
        value = self.values[i]  # [s, n]
        index = self.indices[i]  # [s, n]
        logits = torch.full(
            (self.max_seq_len, self.vocab_size),
            fill_value=-1e4
        )
        for j in range(self.max_seq_len):
            logits[j, index[j]] = value[j].to(logits)
        return logits  # [s, v]


class Timer:
    def __init__(self, total: int, episode: int = 1):
        self.total = total
        self.ticktock = 0
        self.last = None
        self.avg_time = 0
        self.episode = episode

    @staticmethod
    def format_clock(period):
        hour, minute, second = period // 3600, (period % 3600) // 60, period % 60
        return int(hour), int(minute), int(second)

    def step(self):
        if self.last is not None:
            period = time.time() - self.last
            self.avg_time = (self.avg_time * (self.ticktock - 1) + period) / self.ticktock
            h1, m1, s1 = self.format_clock(self.avg_time * (self.ticktock + 1))
            h2, m2, s2 = self.format_clock(self.avg_time * (self.total - self.ticktock))
            if self.ticktock % self.episode == 0:
                print(
                    f"STEP {self.ticktock}/{self.total} | USED: %02d:%02d:%02d | AVG %.2f s/it | "
                    f"ETA: %02d:%02d:%02d" % (h1, m1, s1, self.avg_time, h2, m2, s2)
                )
        self.last = time.time()
        self.ticktock += 1
        if self.ticktock == self.total:
            self.reset()

    def reset(self):
        self.ticktock = 0
        self.last = None
        self.avg_time = 0


class AverageMeter:
    def __init__(self):
        self.average = 0
        self.step = 0

    def forward(self, x: int):
        """ Accumulate average computation """
        self.step += 1
        self.average = self.average + 1 / self.step * (x - self.average)
        return self.average

    def reset(self):
        self.average = 0
        self.step = 0


class VarianceMeter:
    def __init__(self):
        self.average_meter = AverageMeter()
        self.sum_square = 0
        self.variance = 0
        self.step = 0

    def std(self):
        return math.sqrt(self.variance)

    def forward(self, x: int):
        """ Accumulate average computation """
        self.average_meter.forward(x)
        self.sum_square += x ** 2
        self.step += 1
        self.variance = self.sum_square / self.step - self.average_meter.average ** 2

        return self.variance

    def reset(self):
        self.average_meter.reset()
        self.sum_square = 0
        self.variance = 0
        self.step = 0
