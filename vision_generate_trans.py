import os.path

from src.tokenizers.processor_qwen3_vl import Qwen3VLProcessor
from src.utils import json_load
from src.video_utils import downscale_video


def preprocess_trans_bench(root_dir: str = "../../data/results/trans/") -> list:
    datalist = json_load(os.path.join(root_dir, "results.json"))["test"]
    results = []
    for data in datalist:
        mp4_path = os.path.join(root_dir, data["video"])
        if not os.path.exists(mp4_path):  # skip
            continue
        flv_path = os.path.join(root_dir, data["video"].replace(".mp4", ".flv"))  # .flv for qwen processor
        if not os.path.exists(flv_path):
            downscale_video(mp4_path, flv_path, fps=1)
        data["video"] = flv_path
        data["text"] = f"{data['question']}"
        for option in data["options"].keys():
            data["text"] += f"\n{option}. {data['options'][option]}"
        results.append(data)
    print(results[-1]["text"])
    return results


def read_data():
    datalist = preprocess_trans_bench("../../data/results/trans/")
    processor = Qwen3VLProcessor("../../models/qwen3-vl/qwen-3-vl-8b-instruct/")
    lens = []
    for data in datalist:
        input_ids = processor.apply_chat_template(texts=[data["text"]], videos=[data["video"]]).input_ids[0]
        print(len(input_ids))
        lens.append(len(input_ids))
    print(max(lens))


if __name__ == '__main__':
    read_data()

