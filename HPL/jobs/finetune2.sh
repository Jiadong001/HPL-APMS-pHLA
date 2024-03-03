#!/bin/bash
cd ../

mkdir -p "./logs/train_logs/HPL-Cluster/"

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port 66666 \
        --nproc_per_node=4 \
        fine_tune_tape2.py \
        --data_path /data/lujd/neoag_data/   \
        --model_path /data/lujd/neoag_model/main_task/HPL-Cluster/G_semantic_01010/ \
        --target_hla HLA-G*01:01 \
        --dataset_type distance-based \
        --distance_type tape_repr_mean \
        --num_cluster equal \
        --l_r 1e-5 \
        --batch_size 26 \
        >> ./logs/train_logs/HPL-Cluster/G_semantic_1e-5_208_0101.txt


# hint: 
#       If there are two 4-GPU training jobs on a server with 8 GPUs, you 
#       need to set a different port (29500 by default) for each job to 
#       avoid communication conflicts.
#       For example: python -m torch.distributed.launch --master_port 66666 ......