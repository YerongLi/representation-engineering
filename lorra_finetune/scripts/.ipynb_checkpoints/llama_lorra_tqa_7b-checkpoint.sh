#!/bin/bash

# source /opt/rh/devtoolset-10/enable

ds_master_port=$((29000 + RANDOM % 1000))

cd ..

CUDA_VISIBLE_DEVICES=2,3 deepspeed --master_port $ds_master_port --num_gpus 2 src/llama2_lorra.py \
    --model_name_or_path  "/home/yerong2/models/Llama-2-7b-chat-hf" \
    --user_tag '[INST]' \
    --assistant_tag '[/INST]' \
    --pos_type 'a truthful' \
    --neg_type 'an untruthful' \
    --control_template "Give {type} answer." \
    --target_layers "10,12,14,16,18,20" \
    --lorra_alpha 5 \
    --lorra_beta 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir ./lorra_tqa_7b \
    --overwrite_output_dir \
    --max_steps 700 \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --do_eval \
    --evaluation_strategy "steps" \
    --eval_steps 10  \
    --save_total_limit 0 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 128 \
    --q_lora False \
    --deepspeed configs/ds_config_zero2.json \
    --gradient_checkpointing True \
    --report_to none \


# "/data/private_models/cais_models/llama-2/llama/llama-2-13b-chat-hf/"
# "/data/private_models/cais_models/vicuna/vicuna-30b-uncensored"