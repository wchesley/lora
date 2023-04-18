#!/bin/bash

export MODEL_NAME="johnslegers/epic-diffusion-v1.1"
export INSTANCE_DIR="./images"
export OUTPUT_DIR="./new_test_output"

lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --scale_lr \
  --learning_rate_unet=2e-4 \
  --learning_rate_text=2e-6 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --lr_scheduler_lora="linear" \
  --lr_warmup_steps_lora=100 \
  --placeholder_tokens="<s1>|<s2>" \
  --save_steps=100 \
  --max_train_steps_ti=3000 \
  --max_train_steps_tuning=3000 \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.001 \
  --continue_inversion \
  --continue_inversion_lr=1e-4 \
  --device="cuda:0" \
  --lora_rank=1 \
  --mixed-precision="fp8"
