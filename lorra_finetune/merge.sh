#!/bin/bash

process_checkpoint() {
    CHECKPOINTNAME=$1
    subfolder="${CHECKPOINTNAME##*/}"
    echo "merged/${subfolder}"
    python3 merge_peft_adapter.py \
        --adapter_model_name=$CHECKPOINTNAME \
        --base_model_name=/home/yerong2/models/internlm-xcomposer2d5-7b \
        --output_name="merged/${subfolder}"
}

# Take user input for CHECKPOINTNAME
CHECKPOINTNAME=$1
process_checkpoint $CHECKPOINTNAME