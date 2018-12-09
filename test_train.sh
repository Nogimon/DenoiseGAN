#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main_ocmtry.py \
    --output_dir ./result_train/ \
    --summary_dir ./result_train/log/ \
    --mode test \
    --is_training False \
    --task denoise \
    --batch_size 16 \
    --input_dir_LR ./data/DENOISE_LR_1001/ \
    --input_dir_HR ./data/DENOISE_HR_1001/ \
    --num_resblock 16 \
    --perceptual_mode MSE \
    --pre_trained_model True \
    --checkpoint ./experiment_denoise/model-120000
    #!--checkpoint ./SRGAN_pre-trained/model-200000
    
