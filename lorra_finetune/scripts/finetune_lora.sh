#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`
cd ../src
# export MODEL="/home/yerong2/models/internlm-xcomposer2d5-7b"
export MODEL="merged/output"
# export DATA="path of data"
export DATA="data.txt"
ds_master_port=$((29000 + RANDOM % 1000))
GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
# deepspeed --num_gpus 2 finetune.py \
deepspeed --master_port $ds_master_port --include localhost:1,2 finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --given_num True \
    --bf16 True \
    --fix_vit True \
    --fix_sampler True \
    --use_lora True \
    --hd_num 18 \
    --output_dir output/finetune_lora \
    --num_train_epochs 10 \
    --batch_size 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --max_length 1024 \
    --deepspeed ds_config_zero2.json \
    --gradient_checkpointing True
