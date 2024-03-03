#!/bin/bash
cd ../
cd ./Evaluation_HPL/

mkdir -p "../logs/eval_logs/HPL-Allele/"

CUDA_VISIBLE_DEVICES=1 python -u evaluation_ft_ensemble.py \
        --data_path /data/lujd/neoag_data/   \
        --model_path /data/lujd/neoag_model/main_task/ \
        --plm_input_type cat \
        --plm_output_type mean \
        --eval_options fullranking \
        --fullranking_type zeroshot_eg \
        --fullranking_target single \
        --target_hla HLA-G*01:01 \
        >> ../logs/eval_logs/HPL-Allele/G0101_ensemble.txt