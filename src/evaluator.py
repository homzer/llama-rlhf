import collections
import re
from typing import List

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PairwiseDataset, JsonDataset
from src.entities import Timer
from src.generator import GeneratorForCausalLM, GeneratorForVerifier
from src.modeling.llama_abstract import AbstractLoraLlamaVerifier
from src.modeling.modeling import ModelForCausalLM
from src.tokenizer import Tokenizer, LlamaTokenizer


class SolverEvaluator:
    def __init__(
            self,
            model: ModelForCausalLM,
            tokenizer: Tokenizer,
            batch_size: int,
            max_seq_len: int
    ):
        self.generator = GeneratorForCausalLM(model, tokenizer, max_seq_len)
        self.evaluators = {
            "GSM8K": GSM8KEvaluator,
        }
        self.batch_size = batch_size

    def forward(self, task: str, dataset: JsonDataset, t: float = 0.0, p: float = 1.0):
        print(f"Evaluating {task}.........")
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        evaluator = self.evaluators[task]()

        datalist = []
        timer = Timer(len(dataloader))
        for data in tqdm(dataloader):
            timer.step()
            outs = self.generator.forward(data['instruction'], t=t, p=p)
            for i, out in enumerate(outs):
                datalist.append(dict(
                    instruction=out['instruction'],
                    output=out['output'],
                    label=data['label'][i]
                ))

        for data in datalist:
            data['predict'] = evaluator.forward(data['output'], data['label'])

        Output = collections.namedtuple('Output', ['acc', 'datalist', 'missing', 'correct'])
        return Output(acc=evaluator.meter.avg, datalist=datalist, missing=evaluator.miss, correct=evaluator.correct)


class VerifierEvaluator:
    def __init__(
            self,
            model: AbstractLoraLlamaVerifier,
            tokenizer: LlamaTokenizer,
            batch_size: int,
            max_seq_len: int
    ):
        self.generator = GeneratorForVerifier(model, tokenizer, max_seq_len)
        self.meter = AvgMeter()
        self.batch_size = batch_size

    def forward(self, dataset: PairwiseDataset):
        print("Evaluating ...")
        dataset.randomize = False
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        self.meter.reset()
        datalist = []
        for data in tqdm(dataloader):
            chosen_outs = self.generator.forward(data['instruction'], data['chosen'])
            rejected_outs = self.generator.forward(data['instruction'], data['rejected'])
            for i in range(len(data['instruction'])):
                c_reward = chosen_outs.rewards[i]
                r_reward = rejected_outs.rewards[i]
                datalist.append(dict(
                    instruction=data['instruction'][i],
                    chosen=data['chosen'][i],
                    rejected=data['rejected'][i],
                    chosen_reward=c_reward,
                    rejected_reward=r_reward,
                    chosen_tokens_rewards=chosen_outs.tokens_rewards[i],
                    rejected_tokens_rewards=rejected_outs.tokens_rewards[i]
                ))
                self.meter.forward(1 if c_reward > r_reward else 0)
        Output = collections.namedtuple('Output', ['acc', 'datalist'])
        return Output(acc=self.meter.avg, datalist=datalist)


# ================================================================================ #


class GSM8KEvaluator:
    def __init__(self):
        super().__init__()
        self.meter = AvgMeter()
        self.float = r"(-?\d+)(,?\d+)?(\.\d+)?"
        self.patterns = [
            r'(?:Therefore|therefore)(.*)\n?',
            r'(?:So|so)(.*)\n?',
        ]
        self.patterns_rg = r',|\.0+$'
        self.numeric_words = {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12"
        }
        self.miss = 0
        self.correct = 0

    def reset(self):
        self.meter.reset()
        self.miss = 0
        self.correct = 0

    def words_to_numbers(self, text: str):
        """ replace `One`, `two` with `1`, `2` etc in a text. """
        pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for (
            word
        ) in self.numeric_words.keys()) + r')\b', re.IGNORECASE)
        # Replace numeric words with their corresponding numbers
        converted_text = pattern.sub(
            lambda match: self.numeric_words[match.group().lower()],
            text
        )
        return converted_text

    def format_label(self, label: str):
        return re.sub(self.patterns_rg, "", label)

    def extract_numbers(self, text: str):
        results = []
        text = self.words_to_numbers(text)
        number_matches = re.findall(self.float, text)
        for nm in number_matches:
            results.append(self.format_label("".join(nm)))
        return results

    def forward(self, output: str, label: str = None) -> List:
        final_results = []
        # splitting
        output = output.strip().split("\n")
        for out in output:
            results = []
            for pattern in self.patterns:
                # looking forward "Therefore" or "So" sentence
                matches = re.findall(pattern, out)
                for match in matches:
                    results.extend(self.extract_numbers(match))
            if len(results) == 0:
                results.extend(self.extract_numbers(out))
            if len(results) != 0:
                final_results = results

        # evaluation
        if len(final_results) == 0:
            self.miss += 1
        elif label is not None:
            if self.format_label(label) in final_results[-1:]:
                self.meter.forward(1)
                self.correct += 1
            else:
                self.meter.forward(0)

        return final_results


class AvgMeter:
    def __init__(self):
        self.avg = 0
        self.step = 0

    def forward(self, x: int):
        """ Accumulate average computation """
        self.step += 1
        self.avg = self.avg + 1 / self.step * (x - self.avg)
        return self.avg

    def reset(self):
        self.avg = 0
        self.step = 0
