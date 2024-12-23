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
    --model_type qwen2-vl-7b-instruct \
    --model_id_or_path qwen/Qwen2-VL-7B-Instruct \
    --dataset ../ms-data/math360k/qw/trainCoT.json \
    --sft_type lora \
    --reeng true \
    --user_tag '' \
    --assistant_tag '[/INST]' \
    --control_template "{type}" \
    --template_system "ixc_system" \
    --pos_type 'As a precise assistant solving a vision math problem, extract key information from the image, solve the following math problem, and carefully reason through each step to provide a truthful and accurate solution.' \
    --neg_type 'As a careless assistant solving a vision math problem, instead of understanding the image and question carefully, use random clues from the image to make up some reasoning and solve the following math problem.' \
    --target_layers "10,12,14,16,18,20" \
    --tuner_backend peft \
    --dtype AUTO \
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
    --learning_rate 3e-5 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 10 \
    --query_max_len 3072 \
    --response_max_len 1000 \
    --eval_steps 2 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 2 \
    --deepspeed default-zero2 \
    --report_to wandb \
    --resume_from_checkpoint output/qwen2-vl-7b-instruct/v55-20241110-131430/checkpoint-100 \
    # --resume_from_checkpoint output/qwen2-vl-7b-instruct/v2-20241024-002648/checkpoint-800
    # --resume_from_checkpoint output/qwen2-vl-7b-instruct/v1-20241023-161445/checkpoint-600
    # --max_steps 600 \
    
    # --resume_from_checkpoint output/qwen2_5-7b/v17-20240926-073918/checkpoint-100
# $MODELS/Qwen2.5-7B 
    # --dataset dureader-robust-zh \
    # --dataset coco-en-mini#20000 \
            # --template_type qwen2-vl \

    # --model_id_or_path qwen/Qwen2-VL-7B-Instruct \
    # --pos_type 'As a precise assistant solving a vision math problem, extract key information from the image, solve the following math problem, and carefully reason through each step to provide a truthful and accurate solution.' \
    # --neg_type 'As a careless assistant solving a vision math problem, instead of understanding the image and question carefully, use random clues from the image to make up some reasoning and solve the following math problem.' \