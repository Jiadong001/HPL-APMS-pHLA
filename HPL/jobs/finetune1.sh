#!/bin/bash
cd ../

mkdir -p "./logs/train_logs/HPL-Pan/"

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch\
        --nproc_per_node=4 \
        fine_tune_tape.py \
        --data_path /data/lujd/neoag_data/   \
        --model_path /data/lujd/neoag_model/main_task/HPL-Pan/cat_mean_2mlp/ \
        --plm_input_type cat \
        --plm_output_type mean \
        --l_r 3e-5 \
        --batch_size 4 \
        >> ./logs/train_logs/HPL-Pan/cat_mean_3e-5.txt

# hint: 
#       If there are two 4-GPU training jobs on a server with 8 GPUs, you 
#       need to set a different port (29500 by default) for each job to 
#       avoid communication conflicts.
#       For example: python -m torch.distributed.launch --master_port 66666 ......