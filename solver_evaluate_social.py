import collections
import copy
import json
import os
import re

import fire
from torch.utils.data import Dataset, DataLoader

from src.entities import Timer
from src.generator import GeneratorForCausalLM
from src.models.modeling_utils import get_parallel_model
from src.utils import json_load, json_dump, setup_model_parallel

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

PROMPT_ZH = """
==角色描述==
{role_profile}

==对话历史==
{conversations}

你要扮演{role_name}角色，你在聊天中要具备该角色对应的知识背景，语气风格等特征。
请根据所给的{role_name}角色描述和对话历史，从下面四个选项（A. B. C.和D.）中选择符合{role_name}的选项：
{options}

你的选择：
"""

PROMPT_DIALOGUE_EMOTION_EN = """
==Conversations==
{conversations}

Select the option () that best matches the mood in utterance "{utterance}". Single Choice
{options}

Your selection:
"""

PROMPT_DIALOGUE_EMOTION_ZH = """
==对话历史==
{conversations}

单选选择题，选择最符合"{utterance}"说话者当时心情的选项()
{options}

你的选择:
"""

PROMPT_OPEN_ZH = """
==角色描述==
{role_profile}

==对话历史==
{conversations}

你要扮演{role_name}角色，你在聊天中要具备该角色对应的知识背景，语气风格等特征。
请根据所给的{role_name}角色描述和对话历史，根据最后一个User的对话再补充一轮你作为Assistant的回复（一轮就好）：
Assistant: 
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


def format_instruction(data):
    dialogue = data['dialogue']
    choices = data['choices'] if 'choices' in data else None
    category = data['meta']['category']
    lang = data['meta']['lang']
    outputs = format_question(dialogue, choices)
    if category == "Individual-MEM":
        PROMPT = PROMPT_OPEN_EN if lang.lower() == "en" else PROMPT_OPEN_ZH
        prompt = PROMPT.format_map({
            "role_profile": data['meta']['profile'][data['meta']['name']],
            "conversations": outputs.dialogue,
            "role_name": data['meta']['name'],
        })
    elif category == "Individual-EP-DialogueEmotionDetect":
        PROMPT = PROMPT_DIALOGUE_EMOTION_EN if lang.lower() == "en" else PROMPT_DIALOGUE_EMOTION_ZH
        prompt = PROMPT.format_map({
            "conversations": outputs.dialogue,
            "options": outputs.options,
            "utterance": dialogue[-1]["value"]
        })
    elif category in ["Individual-EP-HumorSarcasmDetect", "Individual-EP-SituationUnderstanding"]:
        prompt = f"{outputs.dialogue}\n{outputs.options}"
    else:
        assert category in [
            'Group-SAP-Positive',
            'Group-SAP-Negative',
            'Group-SAP-Neutral',
            'Individual-SA-RoleStyle',
            'Individual-SA-RoleKnowledge'
        ]
        PROMPT = PROMPT_EN if lang.lower() == "en" else PROMPT_ZH
        prompt = PROMPT.format_map({
            "role_profile": data['meta']['profile'][data['meta']['name']],
            "conversations": outputs.dialogue,
            "role_name": data['meta']['name'],
            "options": outputs.options
        })
    return prompt


class SocialBenchDataset(Dataset):
    def __init__(self, f: str, limit: int = None):
        self.datalist = json_load(f)
        if limit is not None:
            self.datalist = self.datalist[: limit]

    def __getitem__(self, i):
        data = copy.deepcopy(self.datalist[i])
        instruction = format_instruction(data)
        label = json.dumps(data['label'])
        return dict(instruction=instruction, label=label, category=data['meta']['category'])

    def __len__(self):
        return len(self.datalist)

    @classmethod
    def compute_score(cls, predict: str, label: str, category: str):
        labels = json.loads(label)  # type(label) == list
        if category == "Individual-MEM":
            predict = predict.lower()
            if len(predict) == 0:
                return None
            score = 0
            for keyword in labels:
                score += 1 if keyword.lower() in predict else 0
            return score / len(labels)
        else:
            answers = format_predict(predict)
            if len(answers) == 0:
                return None
            if len(labels) == 1:  # single choice
                return 1 if answers[0] == labels[0] else 0
            # multi choices
            for answer in answers:
                if answer not in labels:
                    return 0
            return len(set(answers)) / len(set(labels))


def format_predict(predict: str):
    answer = set()
    matches = re.findall(r'(\W+|^|[\u4e00-\u9fa5]+)([A-H])($|\W+|[\u4e00-\u9fa5]+)', predict)
    for match in matches:
        answer.add(match[1])
    return list(answer)


# =================================================================================================


def run(
        ckpt_dir: str,
        model_type: str,
        log_dir: str,
        label_dir: str,
        config_file: str,
        tokenizer_file: str,
        max_seq_len: int = 2048,
        max_batch_size: int = 1,
        limit: int = None,
        begin: int = 0
):
    local_rank, world_size = setup_model_parallel(use_float16=True)
    model, tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        local_rank=local_rank,
        world_size=world_size,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=-1
    )
    model.load(ckpt_dir, merge_lora=True)
    generator = GeneratorForCausalLM(model, tokenizer, max_seq_len)
    os.makedirs(log_dir, exist_ok=True)
    label_files = ['self_awareness.json', 'conversation_memory.json',
                   'social_preference.json', 'emotional_perception.json'][begin:]
    for label_file in label_files:
        label_file = os.path.join(label_dir, label_file)
        dataset = SocialBenchDataset(label_file, limit)
        dataloader = DataLoader(dataset, batch_size=max_batch_size)
        timer = Timer(len(dataloader))
        datalist = []
        for data in dataloader:
            timer.step()
            outputs = generator.forward(data['instruction'])
            for i, output in enumerate(outputs):
                score = SocialBenchDataset.compute_score(output['output'], data['label'][i], data['category'][i])
                datalist.append(dict(
                    instruction=data['instruction'][i],
                    output=output['output'],
                    score=score,
                    label=data['label'][i],
                    category=data['category'][i]
                ))
            print(outputs[0]['output'])
            json_dump(datalist, os.path.join(log_dir, os.path.split(label_file)[-1]))
        scores = []
        for data in datalist:
            if data['score'] is not None:
                scores.append(data['score'])
        acc = sum(scores) / (len(scores) + 1e-12)
        save_name = os.path.split(label_file)[-1].replace(".json", f"_{round(acc, 3)}.json")
        json_dump(datalist, os.path.join(log_dir, save_name))


if __name__ == "__main__":
    fire.Fire(run)


