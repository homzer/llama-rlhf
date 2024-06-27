import fire
from torch.utils.data import DataLoader

from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer
from src.generator import GeneratorForCausalLM
from src.modeling import get_parallel_model
from src.utils import setup_model_parallel


def main(
        ckpt_dir: str,
        label_file: str,
        model_type: str = "llama-2-7b",
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        lora_rank: int = -1,
        t: float = 0.0,
        p: float = 1.0,
        tokenizer_file: str = None,
        config_file: str = None,
        use_chat_template: bool = False,
        dtype: str = "bfloat16",
        seed: int = None
):
    local_rank, world_size = setup_model_parallel(seed=seed)
    if tokenizer_file is None:
        tokenizer_file = ckpt_dir
    if config_file is None:
        config_file = ckpt_dir

    model, tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        local_rank=local_rank,
        world_size=world_size,
        max_seq_len=max_seq_len,
        tokenizer_file=tokenizer_file,
        lora_rank=lora_rank,
        dtype=dtype
    )
    dataset = JsonDataset(label_file)
    if use_chat_template:
        dataset = ChatTemplateDataset(dataset, tokenizer)
    dataloader = DataLoader(dataset, batch_size=max_batch_size)
    model.load(ckpt_dir)
    generator = GeneratorForCausalLM(model, tokenizer, max_seq_len)
    timer = Timer(len(dataloader))
    for data in dataloader:
        outputs = generator.forward(data['instruction'], t=t, p=p)
        print(data['instruction'][-1] + "\n" + outputs[-1])
        timer.step()


if __name__ == '__main__':
    fire.Fire(main)