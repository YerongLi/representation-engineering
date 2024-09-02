#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`
GPU=$1
GPUS_PER_NODE=$(echo $GPU | tr ',' '\n' | wc -l)
# export MODEL="internlm/internlm-xcomposer2-7b"
# export MODEL="internlm/internlm-xcomposer2-vl-7b"
# export MODEL="internlm/internlm-xcomposer2-4khd-7b"
export MODEL="/home/yerong2/models/internlm-xcomposer2-vl-7b"
export DATA="math360k.txt"
OUTPUT_DIR=math_system
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
cd ../
# torchrun $DISTRIBUTED_ARGS finetune.py \
deepspeed --master_port $ds_master_port --include localhost:$GPU src/mllm_lorra.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --img_size 490 \
    --hd_num -1 \
    --bf16 True \
    --fix_vit True \
    --fix_sampler True \
    --use_lora True \
    --output_dir $OUTPUT_DIR \
    --max_steps 600 \
    --lora_r 64 \
    --batch_size 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 100 \
    --evaluation_strategy "steps" \
    --eval_steps 10 \
    --logging_steps 1 \
    --save_total_limit 5 \
    --learning_rate 1e-3 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --deepspeed ds_config_zero2.json \
    --gradient_checkpointing True \
    --lorra_alpha 16 \
    --user_tag '' \
    --assistant_tag '[/INST]' \
    --control_template "{type}" \
    --template_system "ixc_system" \
    --pos_type 'As a precise assistant solving a vision math problem, extract key information from the image, solve the following math problem, and carefully reason through each step to provide a truthful and accurate solution.' \
    --neg_type 'As a careless assistant solving a vision math problem, instead of understanding the image and question carefully, use random clues from the image to make up some reasoning and solve the following math problem.' \
    --target_layers "10,12,14,16,18,20,22" \
    --max_length 5120 \
    --query_max_len 4096 \
    --response_max_len 1024 \

    # --learning_rate 4e-3 \ breaks
    # --learning_rate 2e-3 \ breaks
    # --learning_rate 1e-3 \ fluctuate from 60 -> 50 \ breaks
