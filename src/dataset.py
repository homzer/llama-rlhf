import random
from typing import List

import torch
from torch.utils.data import Dataset as TorchDataset

from src.utils import json_load


class JsonDataset(TorchDataset):
    """ Load dataset from json file. """
    def __init__(self, filename):
        self.datalist = json_load(filename)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, i):
        return self.datalist[i]

    def shuffle(self) -> TorchDataset:
        indices = torch.randperm(len(self))
        dataset = torch.utils.data.Subset(self, indices)
        return dataset


class SolutionDataset(JsonDataset):
    def __init__(self, filename):
        super().__init__(filename)
        assert "output" in self.datalist[0].keys()
        assert type(self.datalist[0]['output']) is list

    def __getitem__(self, i):
        data = self.datalist[i].copy()
        data['output'] = random.sample(data['output'], 1)[0]
        return data


class RewardDataset(JsonDataset):
    def __init__(self, filename, randomize: bool = True):
        super().__init__(filename)
        assert "chosen" in self.datalist[0].keys()
        assert type(self.datalist[0]['chosen']) is list
        assert "rejected" in self.datalist[0].keys()
        assert type(self.datalist[0]['rejected']) is list
        self.randomize = randomize

    def __getitem__(self, i):
        data = self.datalist[i].copy()
        # at least one ground truth answer
        assert len(data['chosen']) != 0
        data['chosen'] = random.sample(data['chosen'], 1)[0]

        if len(data['rejected']) == 0:
            rejection = random.sample(self.datalist, 1)[0]
            while len(rejection['rejected']) == 0:
                rejection = random.sample(self.datalist, 1)[0]
            data['rejected'].extend(rejection['rejected'])

        # 10% the time using random sampled rejection or chosen response
        if self.randomize and random.randint(1, 10) == 1:
            if random.randint(1, 2) == 1:
                rejection = random.sample(self.datalist, 1)[0]
                while len(rejection['rejected']) == 0:
                    rejection = random.sample(self.datalist, 1)[0]
                data['rejected'] = random.sample(rejection['rejected'], 1)[0]
            else:
                j = random.randint(0, len(self.datalist) - 1)
                while j == i:
                    j = random.randint(0, len(self.datalist) - 1)
                data['rejected'] = random.sample(self.datalist[j]['chosen'], 1)[0]
        else:  # 90% the time using origin rejection
            data['rejected'] = random.sample(data['rejected'], 1)[0]

        return data


class LogitsData(str):
    def __init__(self, data: List[dict]):
        super().__init__()
        self.data = data
        self.limit = len(self.data)
        self.pointer = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer < self.limit:
            val = self.data[self.pointer]
            self.pointer += 1
            return val
        else:
            self.pointer = 0
            raise StopIteration


class DistillingDataset(JsonDataset):
    """ Dataset for collecting logits data. """
    def __init__(self, filename):
        super().__init__(filename)
        assert "logits" in self.datalist[0].keys()
        for data in self.datalist:
            data["logits"] = [LogitsData(item) for item in data["logits"]]
