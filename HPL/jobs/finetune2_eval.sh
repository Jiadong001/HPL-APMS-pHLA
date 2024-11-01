#!/bin/bash
cd ../
cd ./Evaluation_HPL/

gene="B"
field1="42"
field2="01"

dist_type="seq_blosum_dist"
tag="${gene}_${dist_type}_${field1}${field2}"

log_path="../logs/eval_logs/HPL-Cluster/"
mkdir -p "$log_path"

CUDA_VISIBLE_DEVICES=2 python -u evaluation_ft.py \
        --data_path /data/lujd/neoag_data/   \
        --model_path "/data/lujd/neoag_model/main_task/HPL-Cluster/$tag/" \
        --plm_input_type cat \
        --plm_output_type mean \
        --eval_options fullranking \
        --pos_dataset_type zeroshot_abc \
        --evluate_target single \
        --target_hla "HLA-$gene*$field1:$field2" \
        --batch_size 512 \
        --l_r 1e-05 \
        --B 24 \
        --epoch 12 \
        --date 241025 \
        >> "$log_path/$tag.txt"
