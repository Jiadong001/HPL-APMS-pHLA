#!/bin/bash
cd ../Evaluation_TransPHLA/

mkdir -p "../logs/eval_logs/TransPHLA/"

CUDA_VISIBLE_DEVICES=1 python -u evaluation.py \
                        --data_path /data/lujd/neoag_data/main_task/ \
                        --model_path /data/lujd/neoag_model/main_task/ \
                        --model_name TransPHLA/TransPHLA_official_model.pkl \
                        --recall_logname 34mer_original_TransPHLA \
                        >> ../logs/eval_logs/TransPHLA/TransPHLA_classify.txt 
