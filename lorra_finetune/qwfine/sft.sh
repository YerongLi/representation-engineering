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
    --model_id_or_path qwen/Qwen2.5-7B \
    --model_revision master \
    --sft_type lora \
    --yerong_type lora \
    --tuner_backend peft \
    --template_type default-generation \
    --dtype AUTO \
    --output_dir output \
    --ddp_backend nccl \
    --dataset dureader-robust-zh \
    --train_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 2048 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules q_proj k_proj v_proj \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn false \
    --deepspeed default-zero2 \
    --report_to none \
    --max_steps 200 \
    # --resume_from_checkpoint output/qwen2_5-7b/v17-20240926-073918/checkpoint-100
# $MODELS/Qwen2.5-7B \