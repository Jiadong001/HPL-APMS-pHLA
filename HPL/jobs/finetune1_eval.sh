#!/bin/bash
cd ../
cd ./Evaluation_HPL/

mkdir -p "../logs/eval_logs/HPL-Pan/"

CUDA_VISIBLE_DEVICES=6 python -u evaluation_ft.py \
        --data_path /data/lujd/neoag_data/   \
        --model_path /data/lujd/neoag_model/main_task/HPL-Pan/cat_mean_2mlp/ \
        --plm_input_type cat \
        --plm_output_type mean \
        --eval_options fullranking \
        --fullranking_type zeroshot_abc \
        --batch_size 256 \
        --l_r 3e-05 \
        --epoch 51 \
        --date 1104 \
        >> ../logs/eval_logs/HPL-Pan/zeroshot_abc_fullranking.txt