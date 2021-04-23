#!/usr/bin/env bash

mkdir logs/
export CUDA_VISIBLE_DEVICES=0

nohup python -u test_model.py \
    --model_path=./models/best_model_checkpoint-epoch=04-val_loss=294225.34375-step_count=0.ckpt \
    --num_workers=10 \
    --pin_memory \
    --test_batch_size=500 \
    --verbose > ./logs/test_results.txt &