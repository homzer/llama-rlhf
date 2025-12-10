import gc
import os
import random
import shutil

import fire
import torch

from src.dataset import JsonDataset, ChatTemplateDataset
from src.entities import Timer, IterationHandler
from src.evaluator import DataParallelPolicyEvaluator
from src.modeling import get_parallel_model
from src.parallel.data_parallel.dataloader import ParallelDataLoader
from src.parallel.initialize import setup_model_parallel, get_rank, set_barrier
from src.parallel.optimizer import ParallelOptimizer
from src.trainer import ParallelSolverTrainer
from src.utils import print_current_func_args, json_load, json_dump


def process_multi_outputs(datalist: list) -> list:
    results = []
    for data in datalist:
        if isinstance(data["output"], list):
            for response in data["output"]:
                results.append(dict(instruction=data["instruction"], output=response))
        else:
            results.append(dict(instruction=data["instruction"], output=data["output"]))
    return results


def evaluate_policy(
        model,
        tokenizer,
        log_dir: str,
        label_file: str,
        max_generate_batch_size: int,
        max_seq_len: int,
        use_chat_template: bool,
        temperature: float,
        top_p: float,
        epoch: int,
):
    if label_file is None:
        return
    for file in label_file.split("++"):
        task = str(os.path.basename(file).split("_")[0])
        dataset = JsonDataset(file)
        if use_chat_template:
            dataset = ChatTemplateDataset(dataset, tokenizer)
        evaluator = DataParallelPolicyEvaluator(
            model=model,
            tokenizer=tokenizer,
            batch_size=max_generate_batch_size,
            max_seq_len=max_seq_len,
            temperature=temperature,
            top_p=top_p
        )
        evaluator_outputs = evaluator.forward(task=task, dataset=dataset)
        print(f"{task.upper()} Evaluate Accuracy: {evaluator_outputs.acc} | Missing: {evaluator_outputs.missing}")
        if log_dir is not None and get_rank() == 0:
            os.makedirs(os.path.join(log_dir, "epoch-%03d" % (epoch + 1), task), exist_ok=True)
            json_dump(evaluator_outputs.datalist, os.path.join(
                log_dir, "epoch-%03d" % (epoch + 1), task, f'results-{round(evaluator_outputs.acc, 4)}.jsonl'))


def main(
        ckpt_dir: str,
        save_dir: str,
        train_file: str,
        log_dir: str,
        label_file: str = None,
        model_type: str = "llama",
        tokenizer_file: str = None,
        config_file: str = None,
        max_seq_len: int = 512,
        max_batch_size: int = 1,
        max_generate_batch_size: int = 1,
        max_num_ckpts: int = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        lr: float = 1e-5,
        epochs: int = 1,
        dtype: str = "bfloat16",
        lora_rank: int = -1,
        lora_dtype: str = "bfloat16",
        chunk_size: int = None,
        begin_epoch: int = 0,
        use_chat_template: bool = False,
        seed: int = None,
        save_optim: bool = False,
        model_parallel_size: int = None,
        sequence_parallel_size: int = 1
):
    setup_model_parallel(
        seed=seed,
        log_dir=log_dir,
        model_parallel_size=model_parallel_size,
        sequence_parallel_size=sequence_parallel_size
    )
    print_current_func_args()
    tokenizer_file = tokenizer_file or ckpt_dir
    config_file = config_file or ckpt_dir

    model, tokenizer = get_parallel_model(
        model_type=model_type,
        config_file=config_file,
        tokenizer_file=tokenizer_file,
        max_seq_len=max_seq_len,
        lora_rank=lora_rank,
        dtype=dtype,
        lora_dtype=lora_dtype
    )

    datalist = []
    for file in train_file.split("++"):
        datalist.extend(json_load(file))
    datalist = process_multi_outputs(datalist)
    random.shuffle(datalist)

    for epoch, datalist in IterationHandler(datalist, epochs, chunk_size, begin_epoch):
        dataset = JsonDataset(f=datalist)
        if len(dataset) == 0:
            continue

        if use_chat_template:
            dataset = ChatTemplateDataset(dataset, tokenizer)
        dataloader = ParallelDataLoader(dataset, batch_size=max_batch_size)
        optimizer = ParallelOptimizer(torch.optim.Adam(model.parameters(), lr=lr))
        trainer = ParallelSolverTrainer(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            max_seq_len=max_seq_len,
            save_optim=save_optim
        )
        trainer.load(ckpt_dir if (epoch == 0) else os.path.join(save_dir, "epoch-%03d" % epoch))
        timer = Timer(total=len(dataloader), episode=100)
        for data in dataloader:
            outputs = trainer.forward(
                instructions=data['instruction'],
                outputs=data['output']
            )
            timer.step()
            if trainer.step % 100 == 0:
                print(f'step {trainer.step} of {len(dataloader)} -----------------------------')
                print(f'LOSS: {outputs.loss}')
                trainer.predict(outputs.logits, data['instruction'], data['output'])
        trainer.save(os.path.join(save_dir, "epoch-%03d" % (epoch + 1)))
        if max_num_ckpts is not None and (epoch + 1 - max_num_ckpts) > 0:
            rm_dir = os.path.join(save_dir, "epoch-%03d" % (epoch + 1 - max_num_ckpts))
            if get_rank() == 0 and os.path.exists(rm_dir):
                shutil.rmtree(rm_dir)

        del optimizer
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        set_barrier()

        evaluate_policy(
            model=model,
            tokenizer=tokenizer,
            log_dir=log_dir,
            label_file=label_file,
            max_generate_batch_size=max_generate_batch_size,
            max_seq_len=max_seq_len,
            use_chat_template=use_chat_template,
            temperature=temperature,
            top_p=top_p,
            epoch=epoch
        )


if __name__ == '__main__':
    fire.Fire(main)
