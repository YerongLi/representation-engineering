#!/bin/bash

# source /opt/rh/devtoolset-10/enable

ds_master_port=$((29000 + RANDOM % 1000))
export DATA="data.txt"

cd ..
deepspeed --num_gpus 2 src/mllm_lorra.py \
    --model_name_or_path  "/home/yerong2/models/internlm-xcomposer2d5-7b" \
    --data_path $DATA \
    --given_num True \
    --bf16 True \
    --fix_vit True \
    --fix_sampler True \
    --use_lora True \
    --hd_num 18 \
    --num_train_epochs 1 \
    --batch_size 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --max_length 1024 \
    --gradient_checkpointing True \
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
    --output_dir ./mllm_lorra \
    --overwrite_output_dir \
    --do_eval \
    --eval_steps 10  \
    --save_total_limit 0 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 128 \
    --q_lora False \
    --deepspeed configs/ds_config_zero2.json \
    --report_to none


# "/data/private_models/cais_models/llama-2/llama/llama-2-13b-chat-hf/"
# "/data/private_models/cais_models/vicuna/vicuna-30b-uncensored"