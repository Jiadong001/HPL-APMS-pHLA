#!/bin/bash
cd ../

gene="B"
field1="42"
field2="01"

dist_type="seq_blosum_dist"
tag="${gene}_${dist_type}_${field1}${field2}"

lr=1e-5
bs=24
log_tag="${tag}_${lr}_${bs}"

log_path="./logs/train_logs/HPL-Cluster"
mkdir -p "$log_path"

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port 66666 \
        --nproc_per_node=4 \
        fine_tune_tape2.py \
        --data_path /data/lujd/neoag_data/   \
        --model_path "/data/lujd/neoag_model/main_task/HPL-Cluster/$tag/" \
        --target_hla "HLA-$gene*$field1:$field2" \
        --dataset_type distance-based \
        --distance_type "$dist_type" \
        --num_cluster equal \
        --l_r "$lr" \
        --batch_size "$bs" \
        >> "$log_path/$log_tag.txt"


# hint: 
#       If there are two 4-GPU training jobs on a server with 8 GPUs, you 
#       need to set a different port (29500 by default) for each job to 
#       avoid communication conflicts.
#       For example: python -m torch.distributed.launch --master_port 66666 ......
