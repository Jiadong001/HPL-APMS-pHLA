#!/bin/bash
cd ../
cd ./Evaluation_HPL/

mkdir -p "../logs/eval_logs/HPL-Cluster/"

CUDA_VISIBLE_DEVICES=6 python -u evaluation_ft.py \
        --data_path /data/lujd/neoag_data/   \
        --model_path /data/lujd/neoag_model/main_task/HPL-Cluster/A_semantic_3402/ \
        --plm_input_type cat \
        --plm_output_type mean \
        --eval_options fullranking \
        --fullranking_type zeroshot_abc \
        --fullranking_target single \
        --target_hla HLA-A*34:02 \
        --batch_size 512 \
        --l_r 1e-05 \
        --B 26 \
        --epoch 19 \
        --date 230319 \
        >> ../logs/eval_logs/HPL-Cluster/A_semantic_3402.txt