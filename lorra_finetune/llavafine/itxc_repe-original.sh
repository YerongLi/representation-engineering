# Experimental environment: 2 * A10
# 2 * 19GB GPU memory
if [ -z "$1" ]; then
  echo "Error: No GPU argument provided."
  echo "Usage: $0 <GPU>"
  exit 1
fi
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ds_master_port=$((29000 + RANDOM % 1000))

GPU=$1
GPUS_PER_NODE=$(echo $GPU | tr ',' '\n' | wc -l)
nproc_per_node=$GPUS_PER_NODE

PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=$GPU \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port $ds_master_port \
    llm_sft.py \
    --model_id_or_path "Shanghai_AI_Laboratory/internlm-xcomposer2-7b" \
    --model_revision master \
    --dataset ../ms-data/math360k/qw/trainCoT.json \
    --sft_type lora \
    --reeng false \
    --lorra_alpha 16 \
    --user_tag '' \
    --assistant_tag '[/INST]' \
    --control_template "{type}" \
    --template_system "ixc_system" \
    --pos_type 'As a precise assistant solving a vision math problem, extract key information from the image, solve the following math problem, and carefully reason through each step to provide a truthful and accurate solution.' \
    --neg_type 'As a careless assistant solving a vision math problem, instead of understanding the image and question carefully, use random clues from the image to make up some reasoning and solve the following math problem.' \
    --target_layers "10,12,14,16,18,20" \
    --tuner_backend peft \
    --dtype fp16 \
    --output_dir output \
    --ddp_backend nccl \
    --train_dataset_sample -1 \
    --num_train_epochs 1 \
    --query_max_len 3072 \
    --response_max_len 1000 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 2 \
    --save_steps 2 \
    --save_total_limit 2 \
    --logging_steps 2 \
    --use_flash_attn false \
    --deepspeed default-zero2 \
    --report_to none \
    --max_steps 10 \
    --resume_from_checkpoint output/internlm-xcomposer2-7b-chat/v142-20241026-042907/checkpoint-8 \
    
    # --max_steps 600 \
# $MODELS/Qwen2.5-7B 
    # --dataset dureader-robust-zh \
    # --custom_train_dataset_path ['math'] \
    # --custom_val_dataset_path ['math'] \
# zero2-offload
    # --deepspeed ds_config/ds_config_zero2.json \
# 