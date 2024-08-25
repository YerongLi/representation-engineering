#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`
GPU=$1
GPUS_PER_NODE=$(echo $GPU | tr ',' '\n' | wc -l)
echo "==== NUMBER OF GPUS ==== GPUS_PER_NODE=$GPUS_PER_NODE"

export MODEL="/home/yerong2/models/internlm-xcomposer2d5-7b"
# export MODEL="merged/finetune_lora"
OUTPUT_DIR="test"
# export DATA="path of data"
export DATA="data.txt"

ds_master_port=$((29000 + RANDOM % 1000))


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
deepspeed --include localhost:$GPU --master_port $ds_master_port finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --given_num True \
    --bf16 True \
    --fix_vit True \
    --fix_sampler True \
    --use_lora True \
    --hd_num 18 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 6 \
    --batch_size 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 6 \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --max_length 1024 \
    --deepspeed ds_config_zero2.json \
    --gradient_checkpointing True \
    --user_tag '' \
    --assistant_tag '[/INST]' \
    --control_template "{type}" \
    --template_system "ixc_system" \
    --pos_type 'As a precise assistant solving a vision math problem, extract key information from the image, solve the following math problem, and carefully reason through each step to provide a truthful and accurate solution.' \
    --neg_type 'As a careless assistant solving a vision math problem, instead of understanding the image and question carefully, use random clues from the image to make up some reasoning and solve the following math problem.' \
    --target_layers "10,12,14,16,18,20" \
    --deepspeed ds_config_zero2.json \
    --gradient_checkpointing True \
    --from_checkpoint $OUTPUT_DIR/checkpoint-6

    # --learning_rate 5e-4 \
    