'''
Fine-tune PLM on GPUs
    - Distributed DataParallel
    - GPU0 reports the final results of every epoch (train+valid), which can be seen in logs
    - in "fine_tune_tape.sh": len(CUDA_VISIBLE_DEVICES) = nproc_per_node

Model's configs:
    - based on PLM: input -> PLM -> head -> prediction
    - plm_input_type: cat & sep
    - plm_output_type: mean & cls
'''
import os
import argparse
import datetime
import random
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import torch.distributed as dist
from torch.nn import DataParallel                      # slower than DDP 
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from load_data_ft import pHLA_Dataset, pHLA_Dataset_RN, seq2token
from utils import f_mean, performances, transfer

from tape import ProteinBertConfig
from model_ft import meanTAPE, clsTAPE

# Set Seed
seed = 111
# Python & Numpy seed
random.seed(seed)
np.random.seed(seed)
# PyTorch seed
torch.manual_seed(seed)                 # default generator
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# CUDNN seed
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# Dataloder seed
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def prepare_main_task_loader(
    rand_neg, fold, batch_size, data_path,
    configs, hla_seq_dict, hla2candidates, num_workers, 
    ):
    
    main_train_df = pd.read_csv(
        os.path.join(data_path, "main_task/train_data_fold{}.csv".format(fold)), index_col=0
    )
    main_valid_df = pd.read_csv(
        os.path.join(data_path, "main_task/val_data_fold{}.csv".format(fold)), index_col=0
    )

    for df in [main_train_df, main_valid_df]:
        df["clip"] = df["HLA"].map(lambda x: hla_seq_dict[x])

    if rand_neg:
        train_pos_df = main_train_df[main_train_df.label == 1]
        valid_pos_df = main_valid_df[main_valid_df.label == 1]

        train_dataset = pHLA_Dataset_RN(train_pos_df, hla2candidates, configs)
        val_dataset = pHLA_Dataset_RN(valid_pos_df, hla2candidates, configs)
    else:
        train_dataset = pHLA_Dataset(main_train_df, configs)
        val_dataset = pHLA_Dataset(main_valid_df, configs)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    val_loader = Data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    return train_sampler, train_loader, val_loader,

def train_step(
    model, train_sampler, train_loader, fold, epoch, epochs, threshold, rand_neg, l_r, 
    plm_type, plm_input_type, device, local_rank
): 
    # ***
    train_sampler.set_epoch(seed+epoch-1)               # different distribution among epochs
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=l_r)  # , momentum = 0.99)
    time_train_ep = 0
    y_true_train_list, y_prob_train_list = [], []
    loss_train_list, dec_attns_train_list = [], []

    pbar = tqdm(train_loader)
    pbar.set_description(f"GPU{local_rank} Train epoch-{epoch}")

    if rand_neg:
        for hla_list, pep_seq_list_pos, pep_seq_list_neg in pbar:
            batch_num = len(hla_list)                   # hla_list is a tuple
            
            phla_tokens = seq2token(
                            hla_list + hla_list, 
                            pep_seq_list_pos + pep_seq_list_neg,
                            plm_type,
                            plm_input_type,
                            device
                        )
            train_labels = [1] * batch_num + [0] * batch_num
            
            t1 = time.time()

            train_outputs = model(phla_tokens)
            y_true_train = torch.LongTensor(train_labels).to(device)
            # y_true_train = torch.LongTensor([1] * batch_num + [0] * batch_num).to(device)
            train_loss = criterion(train_outputs, y_true_train)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            time_train_ep += time.time() - t1

            y_prob_train = nn.Softmax(dim=1)(train_outputs)[:, 1].cpu().detach().numpy()
            y_true_train = np.array(train_labels)
            # y_true_train = y_true_train.cpu().numpy()
            
            y_true_train_list.extend(y_true_train)
            y_prob_train_list.extend(y_prob_train)
            loss_train_list.append(train_loss)
    else:
        for hla_list, pep_seq_list, train_labels in pbar:
            phla_tokens = seq2token(
                            hla_list, 
                            pep_seq_list,
                            plm_type,
                            plm_input_type,
                            device
                        )
            
            t1 = time.time()

            train_outputs = model(phla_tokens)
            y_true_train = train_labels.to(device)              # label is a longtensor
            train_loss = criterion(train_outputs, y_true_train)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            time_train_ep += time.time() - t1

            y_prob_train = nn.Softmax(dim=1)(train_outputs)[:, 1].cpu().detach().numpy()
            y_true_train = train_labels.numpy()

            y_true_train_list.extend(y_true_train)
            y_prob_train_list.extend(y_prob_train)
            loss_train_list.append(train_loss)

    y_pred_train_list = transfer(y_prob_train_list, threshold)
    ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)

    ave_loss_train = f_mean(loss_train_list)
    print(
        "GPU{}-Fold-{}-RN-{}****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec".format(
            local_rank, fold, rand_neg, epoch, epochs, ave_loss_train, time_train_ep
        )
    )
    metrics_train = performances(
        y_true_train_list, y_pred_train_list, y_prob_train_list, print_=True
    )
    return (
        ys_train,
        ave_loss_train,     # sp. a tensor
        metrics_train,
        time_train_ep,
    )

def make_validation(
    rand_neg, model, loader, threshold, fold, 
    plm_type, plm_input_type, device, local_rank
):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_val_list, dec_attns_val_list = [], []
    y_true_val_list, y_prob_val_list = [], []

    if not rand_neg:
        with torch.no_grad():
            pbar = tqdm(loader)
            pbar.set_description(f"GPU{local_rank} VALIDATION without random negative samples")
            
            for hla_list, pep_seq_list, val_labels in pbar:
                
                phla_tokens = seq2token(
                            hla_list, 
                            pep_seq_list,
                            plm_type,
                            plm_input_type,
                            device
                        )
                
                val_outputs = model(phla_tokens)
                y_true_val = val_labels.to(device)
                val_loss = criterion(val_outputs, y_true_val)

                y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
                y_true_val = val_labels.numpy()

                y_true_val_list.extend(y_true_val)
                y_prob_val_list.extend(y_prob_val)
                loss_val_list.append(val_loss)
    else:
        with torch.no_grad():
            pbar = tqdm(loader)
            pbar.set_description(f"GPU{local_rank} VALIDATION with random negative samples")
            for hla_list, pep_seq_list_pos, pep_seq_list_neg in pbar:
                
                batch_num = len(hla_list)
                phla_tokens = seq2token(
                            hla_list + hla_list, 
                            pep_seq_list_pos + pep_seq_list_neg,
                            plm_type,
                            plm_input_type,
                            device
                        )
                val_labels = [1] * batch_num + [0] * batch_num

                val_outputs = model(phla_tokens)
                y_true_val = torch.LongTensor(val_labels).to(device) 
                # y_true_val = torch.LongTensor([1] * batch_num + [0] * batch_num).to(device) 
                val_loss = criterion(val_outputs, y_true_val)

                y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
                y_true_val = np.array(val_labels)

                y_true_val_list.extend(y_true_val)
                y_prob_val_list.extend(y_prob_val)
                loss_val_list.append(val_loss)

    y_pred_val_list = transfer(y_prob_val_list, threshold)
    ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)

    ave_loss_val = f_mean(loss_val_list)
    print(
        "GPU{}-Fold-{} ******{}****** : Loss = {:.6f}".format(
            local_rank, fold, "VALIDATION", ave_loss_val
        )
    )
    metrics_val = performances(
        y_true_val_list, y_pred_val_list, y_prob_val_list, print_=True
    )

    return ys_val, ave_loss_val, metrics_val  # , dec_attns_val_list

def train(
    args,
    train_sampler, train_loader, val_loader,
    model, plm_type, plm_input_type,
    device, local_rank, world_size,
    start_epoch=0,
    validation_times=5,
):

    if not args.rand_neg:
        validation_times = 1

    valid_best, ep_best = -1, -1
    epoch = start_epoch
    end_epoch = args.epochs

    time_train = 0
    metrics_name = [
        "roc_auc",
        "accuracy",
        "mcc",
        "f1",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "aupr",
    ]
    performance_val_df = pd.DataFrame()
    while epoch < end_epoch:
        epoch += 1

        # 1.train on GPUs
        ys_train, loss_train, metrics_train, time_train_ep = train_step(
            model,
            train_sampler, train_loader,
            args.fold,
            epoch,
            args.epochs,
            args.threshold,
            args.rand_neg,
            args.l_r,
            plm_type,
            plm_input_type,
            device,
            local_rank
        )
        pass_tensor = loss_train.clone().detach()           # when 'nccl', dist.all_reduce needs a tensor(on device) as in/output

        # 2.train loss synchronization
        dist.barrier()
        dist.all_reduce(pass_tensor)                        # Default op=ReduceOp.SUM
        sum_loss_train = pass_tensor.cpu().detach().item()
        ave_loss_train = sum_loss_train / world_size        # ave_loss on the whole train dataset
        
        if local_rank == 0:
            print("\nGPU{} report: Epoch {}/{} | Train loss = {:.4f}\n".format(
                local_rank, epoch, args.epochs, ave_loss_train
            ))

        # 3.validation on GPUs, not on one GPU
        val_mertrics_avg, val_loss_list = [], []
        for val_time in range(validation_times):
            dist.barrier()
            ys_val, loss_val, metrics_val = make_validation(
                args.rand_neg,
                model,
                val_loader,
                args.threshold,
                args.fold,
                plm_type,
                plm_input_type,
                device,
                local_rank
            )
            
            performance_val_df = pd.concat(
                [
                    performance_val_df,
                    pd.DataFrame(
                        [[epoch, str(val_time)] + list(metrics_val)],
                        columns=["epoch", "rand_val_num"] + metrics_name,
                    ),
                ]
            )
            val_loss_list.append(loss_val)
            val_mertrics_avg.append(sum(metrics_val[:4]) / 4)
        
        # valid results
        cur_epoch_performance_df = performance_val_df.iloc[
            -5:,
        ]
        AUC_avg = cur_epoch_performance_df.roc_auc.mean()
        ACC_avg = cur_epoch_performance_df.accuracy.mean()
        MCC_avg = cur_epoch_performance_df.mcc.mean()
        F1_avg = cur_epoch_performance_df.f1.mean()
        print(
            "GPU{} Validation of Epoch-{}:  AUC_avg = {:.6f}, ACC_avg = {:.6f}, 'MCC_avg = {:.6f}, F1-avg = {:.6f}".format(
                local_rank, epoch, AUC_avg, ACC_avg, MCC_avg, F1_avg
            )
        )
        ave_loss_val = f_mean(val_loss_list)
        ep_avg_val = f_mean(val_mertrics_avg)
        val_result = [
            ave_loss_val,
            AUC_avg, 
            ACC_avg, 
            MCC_avg, 
            F1_avg,
            ep_avg_val
            ]
        
        pass_tensors = torch.tensor(val_result).to(device)

        # 4.valid result synchronization
        dist.barrier()
        dist.all_reduce(pass_tensors)
        sum_val_result = pass_tensors.cpu().detach().tolist()

        assert len(val_result) == len(sum_val_result)
        for i in range(len(val_result)):
            val_result[i] = sum_val_result[i] / world_size     # ave_val on the whole valid dataset

        if local_rank == 0:
            print("\nGPU{} report: Epoch {}/{} | Ave_val_loss = {:.6f} | Ave_AUC = {:.6f} | Ave_ACC = {:.6f} | Ave_MCC = {:.6f} | Ave_F1 = {:.6f}".format(
                local_rank, epoch, args.epochs, val_result[0],
                val_result[1], val_result[2], val_result[3], val_result[4]
            ))
    
        # 5.Better Validation Performance
        ep_avg_val = val_result[-1]
        if ep_avg_val > valid_best :
            valid_best, ep_best = ep_avg_val, epoch

            # 5.save model on local_rank0
            if local_rank == 0:
                print("============================================================")
                print("Better Validation Performance.")
                print("============================================================")
                print("Model Saving")
                if not os.path.exists(args.model_path):
                    os.makedirs(args.model_path)
                print(
                    "****Saving model: Best epoch = {} | Best Valid Mertric = {:.4f}".format(
                        ep_best, ep_avg_val
                    )
                )

                formatted_today = datetime.date.today().strftime("%y%m%d")
                new_model_name = "main_finetune_plm_{}_B{}_LR{}_seq_{}_fold{}_ep{}_{}.pkl".format(
                    args.plm_type,
                    args.batch_size,
                    args.l_r,
                    args.seq_type,
                    args.fold,
                    ep_best,
                    formatted_today,
                )
                print("*****Path saver: ", new_model_name)
                torch.save(model.module.eval().state_dict(), args.model_path + new_model_name)

        if local_rank == 0:
            print("\n")

        dist.barrier()
        time_train += time_train_ep

        # early stop
        if epoch - ep_best >= args.early_stop:
            print("\nGPU{}-EARLY STOP TRIGGERED, Training totally used {:.2f}s".format(local_rank, time_train))
            break

def read_argument():
    parser = argparse.ArgumentParser(
        description="Fine-tune PLM, with HLA clip sequence"
    )

    # basic configuration
    parser.add_argument("--data_path", type=str, default="/data/zhuxy/neoag_data/")
    parser.add_argument("--model_path", type=str, default="/data/zhuxy/neoag_model/")
    parser.add_argument("--pep_max_len", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.5)

    # -----------------Parameters for data------------------------
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument(
        "--seq_type", type=str, choices=["short", "whole", "clip"], default="clip"
    )

    # -----------------Parameters for Model-----------------------
    parser.add_argument(
        "--plm_type", type=str, choices=["tape", "protbert"], default="tape"
    )
    parser.add_argument(
        "--plm_input_type", type=str, choices=["cat", "sep"], default="cat"
    )
    parser.add_argument(
        "--plm_output_type", type=str, choices=["mean", "cls"], default="cls"
    )
    parser.add_argument(
        "--head_type", type=str, default="2mlp"
    )

    # -----------------Parameters for training-------------------
    parser.add_argument("--rand_neg", action="store_false", default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--l_r", type=float, default=1e-5)
    parser.add_argument("--early_stop", type=int, default=5)   # > 4 is enough
    parser.add_argument("--epochs", type=int, default=100)

    # ***
    parser.add_argument("--local_rank", default=-1)

    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":

    args = read_argument()
    
    # path
    data_path = args.data_path
    dir_saver = args.model_path
    os.makedirs(dir_saver, exist_ok=True)

    # model config
    plm_type = args.plm_type
    plm_input_type = args.plm_input_type
    plm_output_type = args.plm_output_type
    head_type = args.head_type

    # define args.hla_max_len ref to args.seq_type
    if args.seq_type == "short":
        args.hla_max_len = 34
    elif args.seq_type == "whole":
        args.hla_max_len = 372
    elif args.seq_type == "clip":
        args.hla_max_len = 182
    
    dataset_configs = {
        "hla_max_len": args.hla_max_len,
        "pep_max_len": args.pep_max_len,
        "hla_seq_type": args.seq_type,
        "padding": True,
    }

    # init GPU process (&communication)
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()            # GPU_rank
    world_size = dist.get_world_size()      # GPU_num
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Prepare data
    print("Data Preparing")
    hla_seq_dict = pd.read_csv(os.path.join(
        data_path, "main_task/HLA_sequence_dict_new.csv"),
        index_col=0
        ).set_index(["HLA_name"])[args.seq_type].to_dict()
    HLA2ranking_candidates = np.load(
        data_path + "main_task/allele2candidate_pools.npy",
        allow_pickle=True,
        ).item()

    train_sampler, train_loader, val_loader = prepare_main_task_loader(
                                                    args.rand_neg,
                                                    args.fold,
                                                    args.batch_size,
                                                    data_path,
                                                    dataset_configs,
                                                    hla_seq_dict,
                                                    HLA2ranking_candidates,
                                                    num_workers=0
                                                ) 

    # Prepare model
    print("Model Preparing")
    # move model on GPU
    tape_config = ProteinBertConfig.from_pretrained('bert-base')
    if plm_output_type == "mean":
        model = meanTAPE(tape_config, head_type).to(device)
    elif plm_output_type == "cls":
        model = clsTAPE(tape_config, head_type).to(device)

    # if torch.cuda.is_available():
    #     model.to(device)

    if torch.cuda.device_count() > 1:
        print(local_rank,"/",world_size)
        # packaging
        model = DistributedDataParallel(model,
                                        device_ids=[local_rank],
                                        output_device=local_rank,
                                        find_unused_parameters=True     # pooler layer is unused
                                        )
        # train
        print("Ready for training")
        train(
                args,
                train_sampler, train_loader, val_loader,
                model, plm_type, plm_input_type,
                device, local_rank, world_size,
                start_epoch=0
            )

    print("\nFinished")