import random
from typing import List

import torch
import numpy as np
from torch.utils.data import Dataset

from src.tokenizers import Tokenizer
from src.utils import json_load, deduplicate_texts


class JsonDataset(Dataset):
    """ Load dataset from json file. """
    def __init__(self, f):
        if type(f) is str:
            self.datalist = json_load(f)
        else:
            self.datalist = f
        self.check_for_none()

    def check_for_none(self):
        for data in self.datalist:
            for key in data.keys():
                if data[key] is None:
                    data[key] = ""  # omit

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, i):
        return self.datalist[i].copy()

    def shuffle(self) -> Dataset:
        indices = torch.randperm(len(self))
        dataset = torch.utils.data.Subset(self, indices)
        return dataset


class ChatTemplateDataset(Dataset):
    def __init__(self, dataset: JsonDataset, tokenizer: Tokenizer):
        self.dataset = dataset
        if type(tokenizer).__name__ == "Llama3Tokenizer":
            tokenizer.eos_id = getattr(tokenizer, "eot_id")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        """ Applying chat template. """
        data = self.dataset.__getitem__(i)
        assert "instruction" in data
        if isinstance(data["instruction"], str):
            data["origin_instruction"] = data["instruction"]
            data["instruction"] = [{"role": "user", "content": data["instruction"]}]
        assert isinstance(data["instruction"], list)
        assert isinstance(data["instruction"][0], dict)
        data["instruction"] = self.tokenizer.apply_chat_template(data["instruction"])
        return data


class MultiOutputsDataset(JsonDataset):
    def __init__(self, f, randomize: bool = False, exhaustive: bool = True):
        super().__init__(f)
        self.exhaustive = exhaustive
        self.randomize = randomize
        self.num_responses_per_data = None
        if self.datalist:
            assert "output" in self.datalist[0].keys()
            assert type(self.datalist[0]['output']) is list
            if self.exhaustive:
                self.num_responses_per_data = len(self.datalist[0]["output"])
                assert all(
                    len(data["output"]) == self.num_responses_per_data for data in self.datalist
                ), "all the number of responses in each data is expected to be equal."

    def __len__(self):
        return (len(self.datalist) * self.num_responses_per_data) if self.exhaustive else super().__len__()

    def __getitem__(self, i):
        if self.exhaustive:
            data = self.datalist[i % len(self.datalist)].copy()
            data["output"] = data["output"][i // len(self.datalist)]
        else:
            data = self.datalist[i].copy()
            data['output'] = data['output'][0]
        return data


class MultiOutputsConsistentDataset(MultiOutputsDataset):
    def __init__(self, f):
        super().__init__(f)
        assert "output" in self.datalist[0]
        assert "indices" in self.datalist[0]
        assert type(self.datalist[0]['output']) is list
        assert type(self.datalist[0]['indices']) is list

    def __getitem__(self, i):
        data = self.datalist[i].copy()
        assert len(data['output']) == len(data['indices'])
        assert len(data['output']) >= 2
        id1, id2 = random.sample([j for j in range(len(data['output']))], 2)
        data['output'] = random.sample(data['output'], 1)[0] if self.randomize else data['output'][0]
        data['indices'] = random.sample(data['indices'], 1)[0] if self.randomize else data['indices'][0]
        return dict(
            instruction=data['instruction'],
            output_a=data['output'][id1],
            output_b=data['output'][id2],
            indices_a=data['indices'][id1],
            indices_b=data['indices'][id2]
        )


class EvoMultiOutputsDataset(MultiOutputsDataset):
    def __init__(self, f):
        super().__init__(f)
        self.map = {}
        for i, data in enumerate(self.datalist):
            self.map[data['instruction']] = i
        assert len(list(self.map.keys())) == len(self.datalist)

    def extend(self, dataset, deduplicate: bool = False) -> int:
        cnt = self.num_outputs()
        for data in dataset.datalist:
            assert data['instruction'] in self.map.keys()
            i = self.map[data['instruction']]
            self.datalist[i]['output'].extend(data['output'])
            if 'output_extend' not in self.datalist[i].keys():
                self.datalist[i]['output_extend'] = []
            self.datalist[i]['output_extend'].extend(data['output'])
            if deduplicate:
                self.datalist[i]['output'] = deduplicate_texts(self.datalist[i]['output'])
        return self.num_outputs() - cnt

    def __getitem__(self, i):
        data = self.datalist[i].copy()
        if 'output_extend' in data.keys():
            data.pop('output_extend')
        outputs = []
        b = len(data['output'])
        for a in range(b):  # Bigger chances for later outputs
            if random.randint(a + 1, b) == b:
                outputs.append(data['output'][a])
        if len(outputs) == 0:  # TODO
            outputs = ['']
        data['output'] = random.sample(outputs, 1)[0]
        return data

    def num_outputs(self) -> int:
        """ Return the total number of outputs. """
        cnt = 0
        for data in self.datalist:
            cnt += len(data['output'])
        return cnt


class PairwiseDataset(JsonDataset):
    def __init__(self, f):
        super().__init__(f)
        assert "chosen" in self.datalist[0].keys()
        if isinstance(self.datalist[0]['chosen'], list):
            print("Warming, chosen is a list, only first element will be used.")
            for data in self.datalist:
                data["chosen"] = data["chosen"][0]
        if isinstance(self.datalist[0]['rejected'], list):
            print("Warming, rejected is a list, only first element will be used.")
            for data in self.datalist:
                data["rejected"] = data["rejected"][0]

    def __getitem__(self, i):
        data = self.datalist[i].copy()
        return data


class ReviseDataset(JsonDataset):
    def __init__(self, f):
        super().__init__(f)
        for i in range(len(self.datalist)):
            if "teacher_output" not in self.datalist[i].keys():
                self.datalist[i]["teacher_output"] = []
            if "student_output" not in self.datalist[i].keys():
                self.datalist[i]["student_output"] = []
        self.map = {}
        for i, data in enumerate(self.datalist):
            self.map[data['instruction']] = i
        assert len(list(self.map.keys())) == len(self.datalist)

    def extend(self, dataset):
        for data in dataset.datalist:
            assert data['instruction'] in self.map.keys()
            i = self.map[data['instruction']]
            self.datalist[i]['student_output'].extend(data['student_output'])
            if 'student_output_extend' not in self.datalist[i].keys():
                self.datalist[i]['student_output_extend'] = []
            self.datalist[i]['student_output_extend'].extend(data['student_output'])

            self.datalist[i]['teacher_output'].extend(data['teacher_output'])
            if 'teacher_output_extend' not in self.datalist[i].keys():
                self.datalist[i]['teacher_output_extend'] = []
            self.datalist[i]['teacher_output_extend'].extend(data['teacher_output'])

    def __getitem__(self, i):
        data = self.datalist[i].copy()
        if 'student_output_extend' in data.keys():
            data.pop('student_output_extend')
        if 'teacher_output_extend' in data.keys():
            data.pop('teacher_output_extend')
        # TODO:
        assert len(data['student_output']) == len(data['teacher_output']) > 0
        i = random.randint(0, len(data['teacher_output']) - 1)
        data['student_output'] = data['student_output'][i]
        data['teacher_output'] = data['teacher_output'][i]
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
    def __init__(self, f):
        super().__init__(f)
        assert "logits" in self.datalist[0].keys()
        for data in self.datalist:
            data["logits"] = [LogitsData(item) for item in data["logits"]]


class MnistDataset(Dataset):
    def __init__(self, f, train: bool = True):
        self.x_data, self.y_data = None, None
        dataset = np.load(f)
        if train:
            self.x_data, self.y_data = dataset['x_train'], dataset['y_train']
        else:
            self.x_data, self.y_data = dataset['x_test'], dataset['y_test']
        self.x_data = np.reshape(self.x_data, [self.x_data.shape[0], -1]) / 255.

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, i):
        return dict(x=self.x_data[i], y=self.y_data[i])
