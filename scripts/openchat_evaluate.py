import os

import fire

from src.dataset import JsonDataset
from src.evaluator import SolverEvaluator
from src.models import OpenChat
from src.models.modeling_args import OpenChatArgs
from src.tokenizers import OpenChatTokenizer
from src.utils import setup_model_parallel, json_dump


class OpenChatDataset(JsonDataset):
    GPT4_USER = "GPT4 Correct User: "
    GPT4_ASSISTANT = "GPT4 Correct Assistant:"
    CODE_USER = "Code User: "
    CODE_ASSISTANT = "Code Assistant:"
    MATH_USER = "Math Correct User: "
    MATH_ASSISTANT = "Math Correct Assistant:"
    FORMAT = "{user}{instruction}<|end_of_turn|>{assistant}"

    def __init__(self, f, task: str = None):
        super().__init__(f)
        if task in ['GSM8K', 'MATH']:
            self.user = OpenChatDataset.MATH_USER
            self.assistant = OpenChatDataset.MATH_ASSISTANT
            print('Using MATH format.')
        elif task in ['CODE']:
            self.user = OpenChatDataset.CODE_USER
            self.assistant = OpenChatDataset.CODE_ASSISTANT
            print('Using CODE format.')
        else:
            self.user = OpenChatDataset.GPT4_USER
            self.assistant = OpenChatDataset.GPT4_ASSISTANT
            print('Using GPT4 format.')

    def __getitem__(self, i):
        """ for single turn only """
        data = self.datalist[i].copy()
        data['instruction'] = OpenChatDataset.FORMAT.format_map({
            "user": self.user,
            "instruction": data['instruction'],
            "assistant": self.assistant
        })
        return data


def main(
        task: str,
        ckpt_dir: str,
        log_dir: str,
        label_file: str,
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        t: float = 0.0,
        p: float = 1.0,
        tokenizer_file: str = None,
        config_file: str = None,
        seed: int = None
):
    local_rank, world_size = setup_model_parallel(
        seed=seed
    )
    dataset = OpenChatDataset(label_file, task=task)
    args = OpenChatArgs(
        max_seq_len=max_seq_len,
        local_rank=local_rank,
        world_size=world_size,
    ).from_json(config_file)

    model = OpenChat(args)
    model.init_weights()
    model.load(ckpt_dir)
    tokenizer = OpenChatTokenizer(tokenizer_file)
    evaluator = SolverEvaluator(model, tokenizer, max_batch_size, max_seq_len)
    outputs = evaluator.forward(task, dataset, t=t, p=p)
    print("Evaluate Accuracy: ", outputs.acc, "Missing: ", outputs.missing)
    os.makedirs(log_dir, exist_ok=True)
    json_dump(outputs.datalist, os.path.join(
        log_dir, f'results-{round(outputs.acc, 4)}.json'
    ), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
