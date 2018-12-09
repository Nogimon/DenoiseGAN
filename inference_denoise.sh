#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./result_denoise/ \
    --summary_dir ./result/log/ \
    --mode inference \
    --is_training False \
    --task denoise \
    --input_dir_LR ./data/denoise_inference\
    --num_resblock 16 \
    --perceptual_mode MSE \
    --pre_trained_model True \
    --checkpoint ./experiment_denoise/model-120000
