# Llama-RLHF

Llama-RLHF is an efficient and easy-to-develop LLM training + inference framework. This project is developed based on PyTorch and FairScale, employing tensor (model) parallelism strategy.

## Requirement

| Library       | Recommend | 
|---------------|-----------|
| python        | 3.8       | 
| torch         | 2.0.1    | 
| transformers | 4.37.2    | 
| fire      | 0.5.0    | 
| fairscale    | 0.4.13    | 
| sentencepiece | 0.1.97     | 
| safetensors           | 0.4.1    | 

## Getting Started

### 1. Checkpoint Splitting

To conduct model parallel training and inference, we need to split the model checkpoint file into several parts. For example, for `world_size=8`, which means we need to split the checkpoint into 8 parts. 
Considering a model parameter file `/path/to/your/checkpoint.bin` (suffixes such as .pth, .safetensors are supported, in fact, as long as the file is stored in the form of a dictionary), run:

```shell script
torchrun checkpoint_split.py \
--ckpt_file /path/to/your/checkpoint.bin \
--save_path /path/to/save/ \
--n 8
```

You are expected to get following checkpoint files:

```
/path/to/save/consolidated.00.pth
/path/to/save/consolidated.01.pth
/path/to/save/consolidated.02.pth
/path/to/save/consolidated.03.pth
/path/to/save/consolidated.04.pth
/path/to/save/consolidated.05.pth
/path/to/save/consolidated.06.pth
/path/to/save/consolidated.07.pth
```

### 2. Model Training

To train an auto-regressive language model, you just need to run:

```shell script
torchrun --nproc_per_node 8 solver_train.py \
--ckpt_dir /path/to/your/ckpt/ \
--save_dir /path/to/save/ \
--train_file dataset/GSM8K/train.json \
--model_type llama-1-7b \
--max_batch_size 6 \
--lora_rank -1
```

Taking llama-1-7b as an example, we provide a `dataset/GSM8K/train.json` file as the training data format, while using full-parameter training. To enable LoRA, simply set the parameter as `--lora_rank=16` in the settings.


### 3. Model Inference

To perform model inference, run the following command:

```shell script
torchrun --nproc_per_node 8 solver_evaluate.py \
--task GSM8K \
--ckpt_dir /path/to/your/ckpt/ \
--log_dir /path/to/log/ \
--label_file dataset/GSM8K/test.json \
--model_type llama-1-7b \
--max_batch_size 384 
```

## PPO Training Pipeline

### 1. Train Your Reward Model
To train a reward model, run:

```shell script
torchrun --nproc_per_node 8 verifier_train.py \
--ckpt_dir /path/to/your/ckpt/ \
--save_dir /path/to/save/ \
--train_file dataset/GSM8K/train.json \
--model_type llama-1-7b \
--max_batch_size 6 \
--lora_rank -1
```

### 2. PPO Training

Suppose you already trained a policy model by running `solver_train.py`, 
and a reward model by running `verifier_train.py`, then run:

```shell script
torchrun --nproc_per_node 8 verifier_train.py \
--actor_ckpt_dir /path/to/your/policy/model/ \
--actor_save_dir /path/to/save/actor/ \
--critic_ckpt_dir /path/to/your/reward/model/ \
--critic_save_dir /path/to/save/critic/ \
--reward_model_ckpt_dir /path/to/your/reward/model/ \
--max_batch_size 6 \
--max_buffer_size 96
```

As you can see, we use the weights of the reward model to initialize the critic model, which facilitates rapid convergence during PPO training.

## Current Supported Models

| Supported Models|
|---------------|
|llama-1-7b|
|llama-1-13b|
|llama-1-33b|
|llama-2-7b|
|llama-2-13b|
|llama-2-70b|
|mistral-7b-instruct-v0.2|
|mixtral-8x7b-instruct-v0.1|
|qwen-7b|
|qwen-14b|
|qwen-72b|
