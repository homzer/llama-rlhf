import collections
import json
import os
import re

import fire
import torch
from torch.utils.data import DataLoader

from src.dataset import JsonDataset
from src.entities import Timer
from src.generator import GeneratorForCausalLM
from src.models.modeling_utils import get_parallel_model
from src.trainer import ParallelSolverTrainer
from src.utils import setup_model_parallel, json_dump

PROMPT_EN = """
==Profile==
{role_profile}

==Conversations==
{conversations}

You are playing the role of {role_name}, you need to embody the knowledge and style of {role_name}.
Based on the provided role Profile and Conversations, please choose the best option (A, B, C, or D):
{options}

Your selection:
"""

PROMPT_DIALOGUE_EMOTION_EN = """
==Conversations==
{conversations}

Select the option () that best matches the mood in utterance "{utterance}". Single Choice
{options}

Your selection:
"""

PROMPT_OPEN_EN = """
==Profile==
{role_profile}

==Conversations==
{conversations}

You are playing the role of {role_name}, you need to embody the knowledge and style of {role_name}.
Based on the provided role Profile and Conversations, you must produce a reply as the Assistant to response to the latest User's message (one term is enough):
Assistant: 
"""


def format_question(dialogue, choices=None):
    conversations = ""
    for con in dialogue:
        role = con['from']
        text = con['value']
        conversations += f"{role}: {text}\n"

    options = ""
    if choices is not None:
        for choice, text in choices.items():
            options += f"{choice}. {text}\n"
    Output = collections.namedtuple('Output', ['dialogue', 'options'])
    return Output(dialogue=conversations, options=options)


def format_predict(predict: str):
    answer = set()
    matches = re.findall(r'(\W+|^|[\u4e00-\u9fa5]+)([A-Z])($|\W+|[\u4e00-\u9fa5]+)', predict)
    for match in matches:
        answer.add(match[1])
    return list(answer)


def get_open_domain_score(predict: str, keywords: list) -> float:
    assert len(keywords) != 0
    predict = predict.lower()
    score = 0
    for keyword in keywords:
        score += 1 if keyword.lower() in predict else 0
    return score / len(keywords)


def compute_score(answers: list, labels: list):
    if len(answers) == 0:
        return 0
    if len(labels) == 1:  # single choice
        return 1 if answers[0] == labels[0] else 0
    # multi choices
    for answer in answers:
        if answer not in labels:
            return 0
    return len(set(answers)) / len(set(labels))


def compute_datalist_score(datalist):
    score = 0
    length = 0
    for data in datalist:
        if 'score' in data:
            length += 1
            score += data['score']
    acc = score / (length + 1e-12)
    return acc


class SocialBenchDataset(JsonDataset):
    def __init__(self, f):
        super().__init__(f)
        results = []
        for data in self.datalist:
            if data['meta']['lang'] == 'en':
                results.append(data)
        self.datalist = results

    def __getitem__(self, i):
        data = self.datalist[i].copy()
        outputs = format_question(data['dialogue'], data['choices'] if 'choices' in data else None)
        if data['meta']['category'] == "Individual-MEM":
            role_name = data['meta']['name']
            role_profile = data['meta']['profile'][role_name]
            instruction = PROMPT_OPEN_EN.format_map({
                "role_profile": role_profile,
                "conversations": outputs.dialogue,
                "role_name": role_name,
            })
            response = data['meta']['reference']
        elif data['meta']['category'] == "Individual-EP-DialogueEmotionDetect":
            instruction = PROMPT_DIALOGUE_EMOTION_EN.format_map({
                "conversations": outputs.dialogue,
                "options": outputs.options,
                "utterance": data['dialogue'][-1]["value"]
            })
            response = "\n".join(data['label'])
        elif data['meta']['category'] in ["Individual-EP-HumorSarcasmDetect", "Individual-EP-SituationUnderstanding"]:
            instruction = f"{outputs.dialogue}\n{outputs.options}"
            response = "\n".join(data['label'])
        else:
            role_name = data['meta']['name']
            role_profile = data['meta']['profile'][role_name]
            instruction = PROMPT_EN.format_map({
                "role_profile": role_profile,
                "conversations": outputs.dialogue,
                "role_name": role_name,
                "options": outputs.options
            })
            response = "\n".join(data['label'])

        return dict(
            instruction=instruction,
            output=response,
            label=json.dumps({"label": data['label']}),
            category=data['meta']['category']
        )


def main(
        ckpt_dir: str,
        log_dir: str,
        train_file: str = "data/SocialBench/en-2048/train.json",
        model_type: str = "llama-1-7b",
        max_seq_len: int = 2048,
        max_batch_size: int = 1,
        eval_batch_size: int = 4,
        lora_rank: int = -1,
        lr: float = 1e-5,
        t: float = 0.0,
        p: float = 1.0,
        tokenizer_file: str = None,
        config_file: str = None,
        seed: int = None
):
    local_rank, world_size = setup_model_parallel(
        use_float16=True, seed=seed
    )
    train_dataset = SocialBenchDataset(train_file)
    train_dataloader = DataLoader(train_dataset, batch_size=max_batch_size)
    model, tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        local_rank=local_rank,
        world_size=world_size,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=lora_rank
    )
    model.load(ckpt_dir, merge_lora=not lora_rank > 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = ParallelSolverTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        max_seq_len=max_seq_len
    )
    for epoch in range(5):
        for data in train_dataloader:
            trainer.forward(
                instructions=data['instruction'],
                outputs=data['output']
            )

    generator = GeneratorForCausalLM(model, tokenizer, max_seq_len)
    for task in ['awareness', 'emotion', 'memory', 'group']:
        datalist = []
        eval_dataset = SocialBenchDataset(f"data/SocialBench/en-2048/{task}.json")
        eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size)
        timer = Timer(len(eval_dataloader))
        for data in eval_dataloader:
            timer.step()
            outputs = generator.forward(data['instruction'], t=t, p=p)
            for i, output in enumerate(outputs):
                datalist.append(dict(
                    instruction=output['instruction'],
                    output=output['output'],
                    label=json.loads(data['label'][i])["label"],
                    category=data['category']
                ))
            print(outputs[0]['instruction'] + re.sub(r'\n\n\n\n+', "", outputs[0]['output']))
        for data in datalist:
            if "MEM" in data['category']:
                data['score'] = get_open_domain_score(data['output'], data["label"])
            else:
                data['score'] = compute_score(format_predict(data['output']), data['label'])
        if local_rank == 0:
            os.makedirs(log_dir, exist_ok=True)
            json_dump(datalist, os.path.join(
                log_dir, f'results-{task}-{compute_datalist_score(datalist)}.json'
            ), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
