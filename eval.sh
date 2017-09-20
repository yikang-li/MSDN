#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train_hdn.py \
    --resume_training --resume_model ./pretrained_models/HDN_1_iters_alt_normal_I_LSTM_with_bias_with_dropout_0_5_nembed_256_nhidden_512_with_region_regression_resume_SGD_best.h5 \
    --dataset_option=normal  --MPS_iter=1 \
    --caption_use_bias --caption_use_dropout \
    --rnn_type LSTM_normal
