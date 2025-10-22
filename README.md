# Llama-RLHF

Llama-RLHF is an efficient and user-friendly framework for training and inference of large language models (LLMs). This project is built on PyTorch and supports model parallelism, data parallelism, and sequence parallelism strategies. It enables training on single-node or multi-node setups.
## Requirement

| Library        | Recommend | 
|----------------|-----------|
| python         | \>=3.10   | 
| torch          | \>=2.0.0  | 
| transformers   | \>=4.51.0 |

## Environment Setup

```
conda create -n llama-rlhf python=3.10
conda activate llama-rlhf

pip install -r requirements.txt
```

## Supervised Fine-Tuning

### 1. Data Construction

Before starting the training, you should prepare the training data for supervised fine-tuning. Specifically, each training data sample should include an `instruction` and an `output` entry. The data file should be in the `jsonl` format and look like the following:

```json lines
{"instruction":  "some text here ...", "output":  "some text here ..."}
{"instruction":  "some text here ...", "output":  "some text here ..."}
```

### 2. Model Downloading

Before starting the training, you need to download the model from [huggingface](https://huggingface.co/).
Taking the Qwen-2.5-3B-Instruct model as an example, download it from [here](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), and create a local directory to store the model, for example, `./qwen-2.5-3b-instruct/`

### 3. Model Training

Suppose you have a training file `./train.jsonl` and model path `./qwen-2.5-3b-instruct/` . To perform supervised fine-tuning, run the following command:

```shell
export CKPT_DIR="./qwen-2.5-3b-instruct/"
export SAVE_DIR="path/to/save/"
export TRAIN_FILE="./train.jsonl"
export MP=2
export SP=1

torchrun --nproc_per_node 8 solver_train.py \
--model_parallel_size ${MP} \
--sequence_parallel_size ${SP} \
--ckpt_dir ${CKPT_DIR} \
--save_dir ${SAVE_DIR} \
--train_file ${TRAIN_FILE} \
--model_type qwen \
--config_file ${CKPT_DIR} \
--tokenizer_file ${CKPT_DIR} \
--max_seq_len 4096 \
--max_batch_size 4 \
--lr 1e-5 \
--epochs 1 \
--use_chat_template
```
In the training script above, we used 8 GPUs on a single node, setting the model parallel size `MP=2` and the sequence parallel size `SP=1`. 
It is worth noting that we did not explicitly set the data parallel size `DP`. Instead, it was calculated as `DP=8/MP/SP=4`. Therefore, the data parallel size in this script is `DP=4`.


### 4. Multi-Node Model Training

The training script can be easily extended from single-node training to multi-node training.

**Warning**: All operations require root privileges, and all users are assumed to be root by default.

#### 4.1 Configure SSH Public Key
Before starting multi-node training, you need to generate an SSH key on the master node (if you don’t already have one):
```shell
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```
Then copy the generated public key (i.e., the contents of the `id_rsa.pub` file) to the `~/.ssh/authorized_keys` file on the other nodes. (This step allows the master node to log in to other nodes via SSH without a password, enabling process startup.)

#### 4.2 Configure the Hostfile
In this step, you need to inform the master node of the IP addresses of the other nodes. Assuming we have four machines, log in to the master node and create a file named `hostfile` in the root directory of the project. The content of the file should be as follows (replace with actual IP addresses):
```text
192.168.0.110 slots=1
192.168.0.111 slots=1
192.168.0.112 slots=1
192.168.0.113 slots=1
```

#### 4.3 Training Script
Based on the single-node training script provided in [Model Training](#3-model-training), extend it to a multi-node training script named `train-multi-node.sh`, as follows:
```shell
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly MASTER_PORT=29980
readonly MASTER_ADDR="${_MASTER_ADDR:-localhost}"

export CKPT_DIR="./qwen-2.5-3b-instruct/"
export SAVE_DIR="path/to/save/"
export TRAIN_FILE="./train.jsonl"
export MP=2
export SP=1

torchrun \
--nnodes ${NNODES} \
--nproc_per_node 8 \
--node_rank ${NODE_RANK} \
--master_addr ${MASTER_ADDR} \
--master_port ${MASTER_PORT} \
solver_train.py \
--model_parallel_size ${MP} \
--sequence_parallel_size ${SP} \
--ckpt_dir ${CKPT_DIR} \
--save_dir ${SAVE_DIR} \
--train_file ${TRAIN_FILE} \
--model_type qwen \
--config_file ${CKPT_DIR} \
--tokenizer_file ${CKPT_DIR} \
--max_seq_len 4096 \
--max_batch_size 4 \
--lr 1e-5 \
--epochs 1 \
--use_chat_template
```
Here, we have 4 nodes, each with 8 GPUs, for a total of 32 GPUs. We set the model parallel size `MP=2` and the sequence parallel size `SP=1`. Therefore, the data parallel size is calculated as `DP=32/2/1=16`.

#### 4.4 Run the Script
We provide an MPI running script in the project: `./mpirun.sh`. This script takes two arguments: the path to the `hostfile` and the training script `train-multi-node.sh`. Execute the following command on the master node:
```shell
mpirun.sh hostfile train-multi-node.sh
```
Your multi-node training program should start running.

## Inference

### 1. Data Construction
Before using the language model to generate texts, you should prepare the test data for model inference. Specifically, each test data sample should include an `instruction` entry. The data file should be in the `jsonl` format and look like the following:
```json lines
{"instruction":  "some text here ..."}
{"instruction":  "some text here ..."}
```

### 2. Model Downloading
Before starting the training, you need to download the model from [huggingface](https://huggingface.co/).
Taking the Llama-3.1-8B-Instruct model as an example, download it from [here](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), and create a local directory to store the model, for example, `./llama-3.1-8b-instruct/`


### 3. Model Inference
Suppose you have a test file `./test.jsonl` and model path `./llama-3.1-8b-instruct/`. To perform inference, run the following command:
```shell
export CKPT_DIR="./llama-3.1-8b-instruct/"
export LOG_DIR="path/to/log/"
export TEST_FILE="./test.jsonl"
export MP=4
export NUM_SAMPLES=1  # the number of responses generated per instruction

torchrun --nproc_per_node 8 solver_generate.py \
--model_parallel_size ${MP} \
--ckpt_dir ${CKPT_DIR} \
--label_file ${TEST_FILE} \
--log_dir ${LOG_DIR} \
--model_type llama3-hf \
--max_seq_len 1024 \
--max_batch_size 128 \
--temperature 0.6 \
--top_p 0.95 \
--tokenizer_file ${CKPT_DIR} \
--config_file ${CKPT_DIR} \
--num_samples_per_prompt ${NUM_SAMPLES} \
--use_chat_template
```
Note that sequence parallelism is not supported for model inference mode. In the training script above, we used 8 GPUs on a single node, setting the model parallel size `MP=4`, the sequence parallel size `SP=1`. So the data parallel size is `DP=8/4/1=2`. 
The generated result will be saved to `LOG_DIR`.

## Model Names and Model Types

| Model Names                      | Model Types |
|----------------------------------|-------------|
| llama-2-7b/13b/70b (original)    | llama       |
| llama-2-7b/13b/70b (huggingface) | llama-hf    |
| llama-3-8b/70b (original)        | llama3      |
| llama-3-8b/70b (huggingface)     | llama3-hf   |
| llama-3.1-8b (original)          | llama3      |
| llama-3.1-8b (huggingface)       | llama3-hf   |
| llama-3.2-1b/3b (original)       | llama3      |
| llama-3.2-1b/3b (huggingface)    | llama3-hf   |
| qwen-7b/14b/72b                  | qwen        |
| qwen-2-0.5b/1.5b/7b/72b          | qwen        |
| qwen-2.5-1.5b/3b/7b/14b/32b/72b  | qwen        |
| qwen-2.5-math-7b/72b             | qwen        |
| qwq-32b                          | qwen        |
| qwen-3-4b/8b/14b/32b             | qwen3       |
| gemma-2-2b/9b/27b                | gemma2      |
| internlm-3-8b-instruct           | internlm3   |
| mistral-7b-v0.1/v0.2/v0.3        | mistral     |
