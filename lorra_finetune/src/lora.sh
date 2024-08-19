#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# export MODEL="internlm/internlm-xcomposer2-7b"
# export MODEL="internlm/internlm-xcomposer2-vl-7b"
# export MODEL="internlm/internlm-xcomposer2-4khd-7b"
export MODEL="/home/yerong2/models/internlm-xcomposer2-vl-7b"
export DATA="data.txt"
# export DATA="path of data"
export ds_master_port=$((29000 + RANDOM % 1000))
GPUS_PER_NODE=8
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

# torchrun $DISTRIBUTED_ARGS finetune.py \
deepspeed --master_port $ds_master_port --include localhost:2,3 finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --img_size 490 \
    --hd_num -1 \
    --given_num True \
    --bf16 True \
    --fix_vit True \
    --fix_sampler True \
    --use_lora True \
    --output_dir math \
    --max_steps 5 \
    --batch_size 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --logging_steps 5 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --report_to "none" \
    --max_length 4096 \
    --deepspeed ds_config_zero2.json \
    --gradient_checkpointing True \
    --user_tag '' \
    --assistant_tag '[/INST]' \
    --control_template "{type}" \
    --pos_type 'As a precise assistant solving a vision math problem, extract key information from the image, solve the following math problem, and carefully reason through each step to provide a truthful and accurate solution.' \
    --neg_type 'As a careless assistant solving a vision math problem, instead of understanding the image and question carefully, use random clues from the image to make up some reasoning and solve the following math problem.' \
    --target_layers "10,12,14,16,18,20" \
    --resume_from_checkpoint math
