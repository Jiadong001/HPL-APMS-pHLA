#!/bin/bash
cd ../

CUDA_VISIBLE_DEVICES=6 python -u ./evaluation.py \
        --data_path /data/lujd/neoag_data/   \
        --model_path /data/lujd/neoag_model/main_task/mlp_tape/5layer_B512_L2e-5/ \
        --plm_type tape \
        --model_type ffn \
        --eval_options fullranking \
        --inference_type mean \
        > ../logs/eval_logs/mlp2/mlp_tape_fullranking.txt