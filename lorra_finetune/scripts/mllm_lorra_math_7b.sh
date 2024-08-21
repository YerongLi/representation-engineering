#!/bin/bash

# source /opt/rh/devtoolset-10/enable
# export MODEL="/home/yerong2/models/internlm-xcomposer2d5-7b"
export MODEL="merged/temp"

ds_master_port=$((29000 + RANDOM % 1000))
export DATA="math360k.txt"
cd ..
deepspeed --master_port $ds_master_port --include localhost:2 src/mllm_lorra.py \
    --model_name_or_path  $MODEL \
    --max_steps 10 \
    --data_path $DATA \
    --bf16 True \
    --fix_vit True \
    --fix_sampler True \
    --use_lora True \
    --hd_num 18 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 7 \
    --evaluation_strategy "steps" \
    --eval_steps 10  \
    --save_strategy "no" \
    --save_steps 10 \
    --save_total_limit 5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --max_length 5420 \
    --query_max_len 4396 \
    --response_max_len 1024 \
    --gradient_checkpointing True \
    --user_tag '' \
    --assistant_tag '[/INST]' \
    --pos_type 'As a precise assistant solving a vision math problem, extract key information from the image, solve the following math problem, and carefully reason through each step to provide a truthful and accurate solution.' \
    --neg_type 'As a careless assistant solving a vision math problem, instead of understanding the image and question carefully, use random clues from the image to make up some reasoning and solve the following math problem.' \
    --control_template "{type}" \
    --target_layers "10,12,14,16,18,20" \
    --lorra_alpha 5 \
    --lorra_beta 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir ./math \
    --overwrite_output_dir \
    --do_eval \
    --learning_rate 1.6e-2 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --tf32 True \
    --q_lora False \
    --deepspeed ds_config_zero2.json \
    --report_to none \
    # --resume_from_checkpoint ./math/checkpoint-10/checkpoint-10
# --learning_rate 2.4e-3 \
# --learning_rate 3e-4 \
# --num_train_epochs 10 \
# --max_steps 40 \
# --model_name_or_path  "/home/yerong2/models/internlm-xcomposer2d5-7b" \
    # --num_train_epochs 2 \

# "/data/private_models/cais_models/llama-2/llama/llama-2-13b-chat-hf/"
# "/data/private_models/cais_models/vicuna/vicuna-30b-uncensored"