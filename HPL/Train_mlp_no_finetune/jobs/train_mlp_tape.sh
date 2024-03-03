#!/bin/bash
cd ../

CUDA_VISIBLE_DEVICES=4 python -u ./train_mlp_from_scratch.py \
    --data_path /data/lujd/neoag_data/   \
    --model_path /data/lujd/neoag_model/main_task/mlp_tape/4layer_B512_L5e-5/ \
    --plm_type tape \
    --n_layers 4 \
    --l_r 5e-5 \
    >> ../logs/train_logs/mlp_tape/4layer_B512_L5e-5.txt