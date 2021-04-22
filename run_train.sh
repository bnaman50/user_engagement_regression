#!/usr/bin/env bash

mkdir logs/
export BS=32 ## batch size
export GAS=1 ## gradient accumulation steps
export CUDA_VISIBLE_DEVICES=0
export MAX_EPOCHS=5

# nohup #-u\
nohup python -u train.py \
    --num_workers=10 \
    --pin_memory \
    --learning_rate=3e-5 \
    --precision=16 \
    --gpus=1 \
    --num_train_epochs=$MAX_EPOCHS \
    --num_sanity_val_steps=-1 \
    --terminate_on_nan=True \
    --deterministic=True \
    --progress_bar_refresh_rate=0 \
    --verbose \
    --freeze_encoder \
    --profiler="simple" \
    --stochastic_weight_avg=True \
    --train_batch_size=$BS --val_batch_size=$BS --gradient_accumulation_steps=$GAS \
    --model_name=distilbert-base-cased \
    --warmup_steps=500 > ./logs/regression_finetuning_logs_max_epochs_${MAX_EPOCHS}.txt &