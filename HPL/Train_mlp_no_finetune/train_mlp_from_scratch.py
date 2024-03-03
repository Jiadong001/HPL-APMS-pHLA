"""
@xinyuanzhu

Main file for mlp training from scratch, with frozen plm:

- Model arc: plm_output -> fully connected layers
- Data format:
    - HLA: name, sequence (default as 'clip', i.e., 182-mer amino acids)
- Training:
    - Random negative sampling is implemented
- Validation:
    - Fixed positive samples + random negative samples
    - 5 times to take the average

"""
import argparse
import datetime
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from load_data import pHLA_Dataset, pHLA_Dataset_RN, extract_features_RN
from model_v1 import Projection, Projection00, Projection10
from utils import f_mean, performances, transfer

from tape import ProteinBertModel, TAPETokenizer    # TAPE
from transformers import AutoTokenizer, AutoModel   # ProtBERT

# Set Seed
seed = 111
# Python & Numpy seed
random.seed(seed)
np.random.seed(seed)
# PyTorch seed
torch.manual_seed(seed)
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
# https://pytorch.org/docs/stable/notes/randomness.html

# GLOBAL_WORKER_ID = None
# def worker_init_fn(worker_id):
#   global GLOBAL_WORKER_ID
#   GLOBAL_WORKER_ID = worker_id
#   np.random.seed(seed + worker_id)


def prepare_main_task_loader(
    rand_neg, fold, batch_size, configs, hla2candidates, data_path, num_workers
    ):# seq_dict,

    main_train_df = pd.read_csv(
        os.path.join(data_path, "main_task/train_data_fold{}.csv".format(fold)), index_col=0
    )
    main_valid_df = pd.read_csv(
        os.path.join(data_path, "main_task/val_data_fold{}.csv".format(fold)), index_col=0
    )
    
    if rand_neg:
        train_pos_df = main_train_df[main_train_df.label == 1]
        valid_pos_df = main_valid_df[main_valid_df.label == 1]

        train_dataset = pHLA_Dataset_RN(train_pos_df, hla2candidates, configs)
        val_dataset = pHLA_Dataset_RN(valid_pos_df, hla2candidates, configs)

        train_loader = Data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )

        val_loader = Data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )
        return (
            train_pos_df,
            valid_pos_df,
            train_loader,
            val_loader,
        )
    else:
        train_dataset = pHLA_Dataset(main_train_df, configs)
        val_dataset = pHLA_Dataset(main_valid_df, configs)

        train_loader = Data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )

        val_loader = Data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )
        return (
            main_train_df,
            main_valid_df,
            train_loader,
            val_loader,
        )

def full_ranking(
    model,
    HLA_name,
    test_pos_peptides,
    candidate_neg_pep_pool,
    use_cuda,
    hla_max_len,
    pep_max_len,
    topk,
    inf_batch,
    plm_configs,
    bottom_k=5,
):

    plm_type = plm_configs["type"]
    plm_tokenizer = plm_configs["tokenizer"]
    plm_model = plm_configs["model"]
    inference_type = plm_configs["inference_type"]
    hla2tensor_dict = plm_configs["hla2tensor_dict"]
    
    device = torch.device("cuda" if use_cuda else "cpu")
    ranking_pool = list(candidate_neg_pep_pool.union(test_pos_peptides))    # pos+neg

    assert len(candidate_neg_pep_pool) + len(test_pos_peptides) == len(ranking_pool)

    ## hla
    HLA_inputs = hla2tensor_dict[HLA_name]
    
    ## pep Tokenizer
    candidate_pep_token_ids = []
    for seq in ranking_pool:
        seq = seq.ljust(pep_max_len, 'X')
        if plm_type == "protbert":
            seq = ' '.join(seq)
        candidate_pep_token_ids.append(plm_tokenizer.encode(seq))
    
    print(
        "Ranking {} samples, Targeting {} positive samples".format(
            len(ranking_pool), len(test_pos_peptides)
        )
    )

    start_index = 0
    end_index = inf_batch

    y_prob_all = torch.LongTensor([])
    model.eval()
    with torch.no_grad():
        while end_index <= len(ranking_pool) and start_index < end_index:
            # Extract features
            ## hla
            if inference_type == "full":
                val_HLA_inputs = HLA_inputs.repeat(
                    (end_index - start_index),1,1
                    )   
            elif inference_type == "pooled" or "mean":
                val_HLA_inputs = HLA_inputs.repeat(
                    (end_index - start_index),1
                    )
            val_HLA_inputs = val_HLA_inputs.to(device)

            ## pep
            val_candidate_pep_token_ids = torch.LongTensor(
                candidate_pep_token_ids[start_index:end_index]
                ).to(device)
            with torch.no_grad():
                if inference_type == "full":
                    val_pep_inputs = plm_model(val_candidate_pep_token_ids)[0]      # full_output
                elif inference_type == "pooled":
                    val_pep_inputs = plm_model(val_candidate_pep_token_ids)[1]      # pooled_output
                elif inference_type == "mean":
                    val_pep_inputs = plm_model(val_candidate_pep_token_ids)[0]
                    val_pep_inputs = torch.mean(val_pep_inputs, dim=1)              # mean_output
            
            val_outputs = model(
                torch.cat((val_HLA_inputs, val_pep_inputs), dim=1)
                )
            # val_outputs, _, _, val_dec_self_attns = model(
            #     val_pep_inputs, val_HLA_inputs
            # )

            y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach()
            # print(y_prob_val)
            y_prob_all = torch.cat((y_prob_all, y_prob_val), dim=0)
            
            start_index = end_index
            if end_index + inf_batch < len(ranking_pool):
                end_index += inf_batch
            else:
                end_index = len(ranking_pool)

        # sorted based on prob
        _, index_of_rank_list = torch.topk(y_prob_all, len(ranking_pool))

        recall_peps = [ranking_pool[_] for _ in index_of_rank_list.numpy()]      
        positive_ranks = sorted(
            [recall_peps.index(pos_pep) for pos_pep in test_pos_peptides]
        )[-bottom_k:]

        # print(len(recall_peps))
        # print(recall_peps[0])
        # print(list(test_data)[0])
        # print(len(test_data))
        # print(len(test_data.intersection(set(candidate_pep))))

        recall = np.array([0.0] * len(topk))
        num_pos = len(test_pos_peptides)
        for ind, k in enumerate(topk):
            num_hit = len(test_pos_peptides.intersection(set(recall_peps[:k])))     # how many topk are pos 
            recall[ind] += num_hit / num_pos

        print("Recall@K")
        print(recall)
        print("Positive Bottom Rank")
        print(f_mean(positive_ranks))

        return recall

def full_ranking_test(
    model,
    train_pos_df,
    test_pos_df,
    HLA2ranking_candidates,
    args,
    plm_configs,
    topk=[50, 100, 500, 1000, 5000, 10000, 100000],
):# hla_seq_dict,

    HLA_list = list(test_pos_df.HLA.unique())           # deduplication, HLA name
    print("HLA allele num: {}".format(len(HLA_list)))

    train_length = len(train_pos_df)                    # for weight calculation

    recall_arr = np.array([0.0] * len(topk))
    recall_arr_weighted = np.array([0.0] * len(topk))

    for current_HLA in HLA_list[:]:
        ranking_candidates = HLA2ranking_candidates[current_HLA]    # neg_pool for current_HLA
        assert isinstance(ranking_candidates, set)

        # test_set: pos pep set for current_HLA
        pos_test_pep = set(list(test_pos_df[test_pos_df["HLA"] == current_HLA].peptide))    

        # HLA_seq = hla_seq_dict[current_HLA]
        # print(current_HLA, HLA_seq)

        print("full ranking on {}".format(current_HLA))

        recall_cur = full_ranking(      # recall@K for current_HLA
            model,
            current_HLA,
            pos_test_pep,           # pos_pool
            ranking_candidates,     # neg_pool
            args.use_cuda,
            args.hla_max_len,
            args.pep_max_len,
            topk,
            512,
            plm_configs
        )

        recall_arr += recall_cur / len(HLA_list)    # mean

        train_length_cur_HLA = len(train_pos_df[train_pos_df["HLA"] == current_HLA])
        recall_arr_weighted += recall_cur * train_length_cur_HLA / train_length
        # w: num_pep for cur_HLA / num_pep for all HLA in train_set

    print(recall_arr)
    print(recall_arr_weighted)
    # all_peps = set(list(data_raw["peptide"]))

    # if rand_neg and data_type == "train":
    #     data_raw = data_raw[data_raw.label == 1]

    # # filter_data = data_raw[data_raw['HLA'] == hla_name]
    # filter_data = data_raw

    # hla_name_list = list(filter_data["HLA"])
    # # hla_seq_list = list(filter_data['HLA_sequence'])
    # hla_seq_list = list(filter_data[args.seq_type])
    # print("HLA seq sample: ", hla_seq_list[0])
    # print("length: ", len(hla_seq_list[0]))

    # pep_seq_list = list(filter_data["peptide"])
    # labels = list(filter_data["label"])

    # if rand_neg and data_type == "train":

    #     print("Dataset type: enable random negative sampling")
    #     used_dataset = pHLA_Dataset_random_neg(
    #         hla_name_list, hla_seq_list, pep_seq_list, all_peps, configs
    #     )
    # else:
    #     print(data_type)
    #     print("Dataset type: no random negative sampling")
    #     used_dataset = pHLA_Dataset(hla_seq_list, pep_seq_list, labels, configs)

    # loader = Data.DataLoader(
    #     used_dataset,
    #     batch_size,
    #     shuffle=True,
    #     num_workers=8,
    #     worker_init_fn=seed_worker,
    # )

    # print(f"{data_type} samples number: {len(loader)*batch_size}")

    # return loader


# def make_validation(model, loader, threshold, fold, use_cuda, verbose=True):
#     device = torch.device("cuda" if use_cuda else "cpu")
#     criterion = nn.CrossEntropyLoss()
#     model.eval()
#     with torch.no_grad():
#         loss_val_list, dec_attns_val_list = [], []
#         y_true_val_list, y_prob_val_list = [], []
#         # print(len(loader))
#         pbar = tqdm(loader)
#         pbar.set_description(f"VALIDATION with random negative samples")
#         for train_pep_inputs, train_hla_inputs, train_labels in pbar:
#             """
#             pep_inputs: [batch_size, pep_len]
#             hla_inputs: [batch_size, hla_len]
#             train_outputs: [batch_size, 2]
#             """
#             train_pep_inputs, train_hla_inputs, train_labels = (
#                 train_pep_inputs.to(device),
#                 train_hla_inputs.to(device),
#                 train_labels.to(device),
#             )

#             t1 = time.time()
#             train_outputs, _, _, train_dec_self_attns = model(
#                 train_pep_inputs, train_hla_inputs
#             )
#             train_loss = criterion(train_outputs, train_labels)

#             # train_outputs, _, _, train_dec_self_attns = model(train_pep_inputs,
#             #                                                   train_hla_inputs)
#             # train_loss = criterion(train_outputs, train_labels)
#             time_train_ep += time.time() - t1

#             optimizer.zero_grad()
#             train_loss.backward()
#             optimizer.step()

#             y_true_train = train_labels.cpu().numpy()
#             y_prob_train = nn.Softmax(dim=1)(train_outputs)[:, 1].cpu().detach().numpy()

#             y_true_train_list.extend(y_true_train)
#             y_prob_train_list.extend(y_prob_train)
#             loss_train_list.append(train_loss)
#         # for val_hla_inputs, val_pos_pep_inputs, val_neg_pep_inputs in pbar:
#         #     batch_num = val_hla_inputs.shape[0]
#         #     val_hla_inputs, val_pos_pep_inputs, val_neg_pep_inputs = (
#         #         val_hla_inputs.to(device),
#         #         val_pos_pep_inputs.to(device),
#         #         val_neg_pep_inputs.to(device),
#         #     )
#         #     val_pos_outputs, _, _, val_dec_self_attns_pos = model(
#         #         val_pos_pep_inputs, val_hla_inputs
#         #     )

#         #     val_neg_outputs, _, _, val_dec_self_attns_neg = model(
#         #         val_neg_pep_inputs, val_hla_inputs
#         #     )

#         #     val_loss = (
#         #         -torch.mean(torch.log(nn.Softmax(dim=1)(val_pos_outputs))[:, 1])
#         #         - torch.mean(torch.log(nn.Softmax(dim=1)(val_neg_outputs))[:, 0])
#         #     ) / 2

#         #     y_true_val = np.array([1.0] * batch_num + [0.0] * batch_num)
#         #     val_outputs = torch.cat((val_pos_outputs, val_neg_outputs))

#         #     y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()

#         #     y_true_val_list.extend(y_true_val)
#         #     y_prob_val_list.extend(y_prob_val)
#         #     loss_val_list.append(val_loss)
#         # #             dec_attns_val_list.append(val_dec_self_attns)

#         y_pred_val_list = transfer(y_prob_val_list, threshold)
#         ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)

#         print(
#             "Fold-{} ******{}****** : Loss = {:.6f}".format(
#                 fold, "VALIDATION", f_mean(loss_val_list)
#             )
#         )
#         metrics_val = performances(
#             y_true_val_list, y_pred_val_list, y_prob_val_list, print_=verbose
#         )
#     return ys_val, loss_val_list, metrics_val  # , dec_attns_val_list


def make_validation(
    rand_neg, model, loader, threshold, fold, use_cuda, plm_configs
):

    plm_type = plm_configs["type"]
    plm_tokenizer = plm_configs["tokenizer"]
    plm_model = plm_configs["model"]
    inference_type = plm_configs["inference_type"]
    hla2tensor_dict = plm_configs["hla2tensor_dict"]

    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    loss_val_list, dec_attns_val_list = [], []
    y_true_val_list, y_prob_val_list = [], []

    if not rand_neg:
        with torch.no_grad():
            pbar = tqdm(loader)
            pbar.set_description(f"VALIDATION without random negative samples")
            
            for hla_list, pep_seq_list, val_labels in pbar:
                """
                pep/hla_inputs: [batch_size, output_size of plm]
                train_outputs: [batch_size, 2]
                """
                hla_inputs, pep_inputs = extract_features_RN(
                    hla_list, pep_seq_list, [],
                    inference_type, plm_type, 
                    plm_tokenizer, plm_model, 
                    hla2tensor_dict, device,
                    rand_neg=False
                )
                val_labels = val_labels.to(device)
                
                val_outputs = model(
                    torch.cat((hla_inputs, pep_inputs), dim=1)
                )
                val_loss = criterion(val_outputs, val_labels)

                # train_outputs, _, _, train_dec_self_attns = model(train_pep_inputs,
                #                                                   train_hla_inputs)
                # train_loss = criterion(train_outputs, train_labels)

                y_true_val = val_labels.cpu().numpy()
                y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
                
                y_true_val_list.extend(y_true_val)
                y_prob_val_list.extend(y_prob_val)
                loss_val_list.append(val_loss)
                # dec_attns_val_list.append(val_dec_self_attns)

            y_pred_val_list = transfer(y_prob_val_list, threshold)
            ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)

            print(
                "Fold-{} ******{}****** : Loss = {:.6f}".format(
                    fold, "VALIDATION", f_mean(loss_val_list)
                )
            )
            metrics_val = performances(
                y_true_val_list, y_pred_val_list, y_prob_val_list, print_=True
            )
    else:
        with torch.no_grad():
            # print(len(loader))
            pbar = tqdm(loader)
            pbar.set_description(f"VALIDATION with random negative samples")
            for hla_list, pep_seq_list, pep_seq_list_neg in pbar:
                batch_num = len(hla_list)
                hla_inputs, pep_inputs = extract_features_RN(
                    hla_list, pep_seq_list, pep_seq_list_neg,
                    inference_type, plm_type, 
                    plm_tokenizer, plm_model, 
                    hla2tensor_dict, device,
                    rand_neg=True
                ) 
                hla_inputs = hla_inputs[:batch_num]
                pos_pep_inputs = pep_inputs[:batch_num]
                neg_pep_inputs = pep_inputs[batch_num:2*batch_num]

                val_pos_outputs = model(
                    torch.cat((hla_inputs, pos_pep_inputs), dim=1)
                )
                val_neg_outputs = model(
                    torch.cat((hla_inputs, neg_pep_inputs), dim=1)
                )

                val_loss = (
                    -torch.mean(torch.log(nn.Softmax(dim=1)(val_pos_outputs))[:, 1])
                    - torch.mean(torch.log(nn.Softmax(dim=1)(val_neg_outputs))[:, 0])
                ) / 2

                y_true_val = np.array([1.0] * batch_num + [0.0] * batch_num)
                
                val_outputs = torch.cat((val_pos_outputs, val_neg_outputs))
                y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()

                y_true_val_list.extend(y_true_val)
                y_prob_val_list.extend(y_prob_val)
                loss_val_list.append(val_loss)
                # dec_attns_val_list.append(val_dec_self_attns)

            y_pred_val_list = transfer(y_prob_val_list, threshold)
            ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)
            # loss_val_list.append(val_loss)
            print(
                "Fold-{} ******{}****** : Loss = {:.6f}".format(
                    fold, "VALIDATION", f_mean(loss_val_list)
                )
            )
            metrics_val = performances(
                y_true_val_list, y_pred_val_list, y_prob_val_list, print_=True
            )

    return ys_val, loss_val_list, metrics_val  # , dec_attns_val_list


# def train_step(model, train_loader, fold, epoch, epochs, threshold, rand_neg, l_r, use_cuda=True):

#     device = torch.device("cuda" if use_cuda else "cpu")
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=l_r)  # , momentum = 0.99)

#     time_train_ep = 0
#     model.train()
#     y_true_train_list, y_prob_train_list = [], []
#     loss_train_list, dec_attns_train_list = [], []
#     if not rand_neg:
#         pbar = tqdm(train_loader)
#         pbar.set_description(f"Train epoch-{epoch}")
#         for train_pep_inputs, train_hla_inputs, train_labels in pbar:
#             '''
#             pep_inputs: [batch_size, pep_len]
#             hla_inputs: [batch_size, hla_len]
#             train_outputs: [batch_size, 2]
#             '''
#             train_pep_inputs, train_hla_inputs, train_labels = train_pep_inputs.to(
#                 device), train_hla_inputs.to(device), train_labels.to(device)

#             t1 = time.time()
#             train_outputs, _, _, train_dec_self_attns = model(train_pep_inputs,
#                                                             train_hla_inputs)
#             train_loss = criterion(train_outputs, train_labels)

#             # train_outputs, _, _, train_dec_self_attns = model(train_pep_inputs,
#             #                                                   train_hla_inputs)
#             # train_loss = criterion(train_outputs, train_labels)
#             time_train_ep += time.time() - t1

#             optimizer.zero_grad()
#             train_loss.backward()
#             optimizer.step()

#             y_true_train = train_labels.cpu().numpy()
#             y_prob_train = nn.Softmax(dim=1)(train_outputs)[
#                 :, 1].cpu().detach().numpy()

#             y_true_train_list.extend(y_true_train)
#             y_prob_train_list.extend(y_prob_train)
#             loss_train_list.append(train_loss)
#     #         dec_attns_train_list.append(train_dec_self_attns)

#         y_pred_train_list = transfer(y_prob_train_list, threshold)
#         ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)

#         print('Fold-{}****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec'.format(fold,
#             epoch, epochs, f_mean(loss_train_list), time_train_ep))
#         metrics_train = performances(
#             y_true_train_list, y_pred_train_list, y_prob_train_list, print_=True)

#         return ys_train, loss_train_list, metrics_train, time_train_ep  # , dec_attns_train_list

#     else:
#         pbar = tqdm(train_loader)
#         pbar.set_description(f"Train epoch-{epoch}")
#         for train_hla_inputs, train_pos_pep_inputs, train_neg_pep_inputs in pbar:
#             batch_num = train_hla_inputs.shape[0]
#             # if batch_num< 128:
#             #     print(train_pos_pep_inputs)
#             #     print(train_neg_pep_inputs)

#             # print(train_hla_inputs.shape, train_pos_pep_inputs.shape, train_neg_pep_inputs.shape)
#             '''
#             pep_inputs: [batch_size, pep_len]
#             hla_inputs: [batch_size, hla_len]
#             train_outputs: [batch_size, 2]
#             '''
#             train_hla_inputs, train_pos_pep_inputs, train_neg_pep_inputs = train_hla_inputs.to(
#                 device), train_pos_pep_inputs.to(device), train_neg_pep_inputs.to(device)

#             t1 = time.time()
#             random_perm = torch.randperm(2 * batch_num).to(device)

#             train_pep_inputs = torch.cat((train_pos_pep_inputs, train_neg_pep_inputs))[random_perm]
#             train_hla_inputs = torch.cat((train_hla_inputs, train_hla_inputs))[random_perm]

#             train_labels_pos = torch.LongTensor([1.0]*batch_num).to(device)
#             train_labels_neg = torch.LongTensor([0.0]*batch_num).to(device)
#             train_labels = torch.cat((train_labels_pos, train_labels_neg))[random_perm]

#             train_outputs, _, _, train_dec_self_attns = model(train_pep_inputs,
#                                                             train_hla_inputs)

#             train_labels_pos = torch.LongTensor([1.0]*batch_num).to(device)
#             train_labels_neg = torch.LongTensor([0.0]*batch_num).to(device)

#             train_loss = criterion(train_outputs, train_labels)
#             # loss_value = -torch.sum(torch.log2(torch.sigmoid(pos_scores))) - torch.sum(torch.log2(torch.sigmoid(1-neg_scores)))


#         # loss_value_pre = -torch.sum(torch.log2(torch.sigmoid(pre_pos_scores))) - torch.sum(torch.log2(torch.sigmoid(1-pre_neg_scores)))
#             # train_outputs, _, _, train_dec_self_attns = model(train_pep_inputs,
#             #                                                   train_hla_inputs)
#             # train_loss = criterion(train_outputs, train_labels)
#             time_train_ep += time.time() - t1

#             optimizer.zero_grad()
#             train_loss.backward()
#             optimizer.step()

#             # y_true_train = train_labels.cpu().numpy()
#             y_true_train = train_labels.cpu().numpy()
#             y_prob_train = nn.Softmax(dim=1)(train_outputs)[
#                 :, 1].cpu().detach().numpy()

#             y_true_train_list.extend(y_true_train)
#             y_prob_train_list.extend(y_prob_train)
#             loss_train_list.append(train_loss)
#     #         dec_attns_train_list.append(train_dec_self_attns)

#         y_pred_train_list = transfer(y_prob_train_list, threshold)
#         ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)

#         print('Fold-{}****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec'.format(fold,
#             epoch, epochs, f_mean(loss_train_list), time_train_ep))
#         metrics_train = performances(
#             y_true_train_list, y_pred_train_list, y_prob_train_list, print_=True)

#         return ys_train, loss_train_list, metrics_train, time_train_ep  # , dec_attns_train_list


def train_step(
    model, train_loader, fold, epoch, epochs, threshold, rand_neg, l_r, plm_configs, use_cuda
): 
    plm_type = plm_configs["type"]
    plm_tokenizer = plm_configs["tokenizer"]
    plm_model = plm_configs["model"]
    inference_type = plm_configs["inference_type"]
    hla2tensor_dict = plm_configs["hla2tensor_dict"]

    print("\n{}-Random Negative Sampling: Enabled: {}".format(epoch, rand_neg))
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=l_r)  # , momentum = 0.99)

    time_train_ep = 0
    model.train()
    y_true_train_list, y_prob_train_list = [], []
    loss_train_list, dec_attns_train_list = [], []

    pbar = tqdm(train_loader)
    pbar.set_description(f"Train epoch-{epoch}")
    if rand_neg:
        for hla_list, pep_seq_list, pep_seq_list_neg in pbar:
            
            # if batch_num< 128:
            #     print(train_pos_pep_inputs)
            #     print(train_neg_pep_inputs)

            # print(train_hla_inputs.shape, train_pos_pep_inputs.shape, train_neg_pep_inputs.shape)
            
            batch_num = len(hla_list)       #hla_list is a tuple
            hla_inputs, pep_inputs = extract_features_RN(
                hla_list, pep_seq_list, pep_seq_list_neg,
                inference_type, plm_type, 
                plm_tokenizer, plm_model, 
                hla2tensor_dict, device,
                rand_neg=True
            ) 
            """
            pep_inputs: [2batch_size, (len,) output_size of plm]
            hla_inputs: [2batch_size, (len,) output_size of plm]
            train_outputs: [2batch_size, 2]
            tips: inputs are already in device
            """

            t1 = time.time()

            train_outputs = model(
                torch.cat((hla_inputs, pep_inputs), dim=1)
            )
            # train_pos_outputs, _, _, train_dec_self_attns_pos = model(
            #     train_pos_pep_inputs, train_hla_inputs
            # )
            # train_neg_outputs, _, _, train_dec_self_attns_neg = model(
            #     train_neg_pep_inputs, train_hla_inputs
            # )
            # train_outputs = torch.cat((train_pos_outputs, train_neg_outputs))
            y_true_train = torch.LongTensor([1] * batch_num + [0] * batch_num).to(
                device
            )
            train_loss = criterion(train_outputs, y_true_train)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            time_train_ep += time.time() - t1

            y_prob_train = nn.Softmax(dim=1)(train_outputs)[:, 1].cpu().detach().numpy()
            y_true_train = y_true_train.cpu().numpy()
            
            y_true_train_list.extend(y_true_train)
            y_prob_train_list.extend(y_prob_train)
            loss_train_list.append(train_loss)
            # dec_attns_train_list.append(train_dec_self_attns)

        y_pred_train_list = transfer(y_prob_train_list, threshold)
        ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)

        print(
            "Fold-{}****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec".format(
                fold, epoch, epochs, f_mean(loss_train_list), time_train_ep
            )
        )
        metrics_train = performances(
            y_true_train_list, y_pred_train_list, y_prob_train_list, print_=True
        )

        return (
            ys_train,
            loss_train_list,
            metrics_train,
            time_train_ep,
        )  # , dec_attns_train_list

    else:
        for hla_list, pep_seq_list, train_labels in pbar:
            # if batch_num< 128:
            #     print(train_pos_pep_inputs)
            #     print(train_neg_pep_inputs)

            # print(train_hla_inputs.shape, train_pos_pep_inputs.shape, train_neg_pep_inputs.shape)

            """
            pep/hla_inputs: [batch_size, output_size of plm]
            train_outputs: [batch_size, 2]
            """
            hla_inputs, pep_inputs = extract_features_RN(
                hla_list, pep_seq_list, [],
                inference_type, plm_type, 
                plm_tokenizer, plm_model, 
                hla2tensor_dict, device,
                rand_neg=False
            )
            train_labels = train_labels.to(device)

            t1 = time.time()

            train_outputs = model(
                torch.cat((hla_inputs, pep_inputs), dim=1)
            )
            # train_outputs, _, _, train_dec_self_attns = model(train_pep_inputs,
            #                                                   train_hla_inputs)
            train_loss = criterion(train_outputs, train_labels)
            # train_loss = (
            #     -torch.mean(torch.log(nn.Softmax(dim=1)(train_pos_outputs))[:, 1])
            #     - torch.mean(torch.log(nn.Softmax(dim=1)(train_neg_outputs))[:, 0])
            # ) / 2

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            time_train_ep += time.time() - t1

            y_true_train = train_labels.cpu().numpy()
            y_prob_train = nn.Softmax(dim=1)(train_outputs)[:, 1].cpu().detach().numpy()

            y_true_train_list.extend(y_true_train)
            y_prob_train_list.extend(y_prob_train)
            loss_train_list.append(train_loss)
            # dec_attns_train_list.append(train_dec_self_attns)

        y_pred_train_list = transfer(y_prob_train_list, threshold)
        ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)

        print(
            "Fold-{}****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec".format(
                fold, epoch, epochs, f_mean(loss_train_list), time_train_ep
            )
        )
        metrics_train = performances(
            y_true_train_list, y_pred_train_list, y_prob_train_list, print_=True
        )

        return (
            ys_train,
            loss_train_list,
            metrics_train,
            time_train_ep,
        )  # , dec_attns_train_list


def train(
    args,
    model,
    train_loader,
    val_loader,
    plm_configs,
    start_epoch=0,
    validation_times=5,
):# hla_seq_dict,

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
        random.seed(seed + epoch)
        ys_train, loss_train_list, metrics_train, time_train_ep = train_step(
            model,
            train_loader,
            args.fold,
            epoch,
            args.epochs,
            args.threshold,
            args.rand_neg,
            args.l_r,
            plm_configs,
            args.use_cuda,
        )  # , dec_attns_train

        val_mertrics_avg = []
        for val_time in range(validation_times):
            ys_val, loss_val_list, metrics_val = make_validation(
                args.rand_neg,
                model,
                val_loader,
                args.threshold,
                args.fold,
                args.use_cuda,
                plm_configs
            )  # , dec_attns_val
            # roc_auc, accuracy, mcc, f1, sensitivity, specificity, precision, recall, aupr = metrics_val

            # performance_val_df = performance_val_df.append(
            #     pd.DataFrame(
            #         [[epoch, str(val_time)] + list(metrics_val)],
            #         columns=["epoch", "rand_val_num"] + metrics_name,
            #     )
            # )
            performance_val_df = pd.concat(
                [
                    performance_val_df,
                    pd.DataFrame(
                        [[epoch, str(val_time)] + list(metrics_val)],
                        columns=["epoch", "rand_val_num"] + metrics_name,
                    ),
                ]
            )
            val_mertrics_avg.append(sum(metrics_val[:4]) / 4)
        
        cur_epoch_performance_df = performance_val_df.iloc[
            -5:,
        ]
        print(
            "Validation of Epoch-{}:  AUC_avg = {}, ACC_avg = {}, 'MCC_avg = {}, F1-avg = {}".format(
                epoch,
                cur_epoch_performance_df.roc_auc.mean(),
                cur_epoch_performance_df.accuracy.mean(),
                cur_epoch_performance_df.mcc.mean(),
                cur_epoch_performance_df.f1.mean(),
            )
        )
        ep_avg_val = f_mean(val_mertrics_avg)

        if ep_avg_val > valid_best :
            print("============================================================")
            print("Better Validation Performance.")
            print("============================================================")
            valid_best, ep_best = ep_avg_val, epoch

            # full-ranking

            print("Model Saving")
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            print(
                "****Saving model: Best epoch = {} | Best Valid Mertric = {:.4f}".format(
                    ep_best, ep_avg_val
                )
            )

            formatted_today = datetime.date.today().strftime("%y%m%d")
            new_model_name = "main_plm_model1_{}_B{}_LR{}_seq_{}_fold{}_ep{}_{}.pkl".format(
                args.plm_type,
                args.batch_size,
                args.l_r,
                args.seq_type,
                args.fold,
                ep_best,
                formatted_today,
            )
            print("*****Path saver: ", new_model_name)
            torch.save(model.eval().state_dict(), args.model_path + new_model_name)
        #     best_test_avg = sum(metrics_test[:4])/4
        #     save_test_to_tensorboard(metrics_test, loss_test_list, epoch)

        # save_to_tensorboard(metrics_train, loss_train_list,
        #     metrics_val, loss_val_list, epoch)

        time_train += time_train_ep

        # early stop
        if epoch - ep_best >= args.early_stop:
            print("\nEARLY STOP TRIGGERED")
            # performance_val_df.to_csv("logs/perfromance_finetune.csv")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train from scratch, with HLA clip sequence"
    )

    # basic configuration
    parser.add_argument("--use_cuda", action="store_false", default=True)
    parser.add_argument("--data_path", type=str, default="/data/zhuxy/neoag_data/")
    parser.add_argument("--model_path", type=str, default="/data/zhuxy/neoag_model/")

    parser.add_argument("--pep_max_len", type=int, default=15)

    parser.add_argument("--threshold", type=float, default=0.5)

    # -----------------Parameters for data------------------------
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument(
        "--seq_type", type=str, choices=["short", "whole", "clip"], default="clip"
    )

    # -----------------Parameters for PLM------------------------
    parser.add_argument(
        "--plm_type", type=str, choices=["tape", "protbert"], default="tape"
    )
    parser.add_argument(
        "--inference_type", type=str, choices=["pooled", "full", "mean"], default="mean"
    )

    # # -----------------Parameters for model 1(mlp)-----------------
    parser.add_argument("--n_layers", type=int, default=5)

    # # -----------------Parameters for model 2(decoder)------------
    # parser.add_argument("--n_layers", type=int, default=1)
    # parser.add_argument("--n_heads", type=int, default=9)

    # parser.add_argument("--d_model", type=int, default=64)
    # parser.add_argument("--d_ff", type=int, default=512)
    # parser.add_argument("--d_k", type=int, default=64)
    # parser.add_argument("--d_v", type=int, default=64)

    # -----------------Parameters for training---------------------
    parser.add_argument("--rand_neg", action="store_false", default=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--l_r", type=float, default=1e-5)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)

    args = parser.parse_args()
    print(args)
    # ================================================================

    data_path = args.data_path
    dir_saver = args.model_path

    device = torch.device("cuda" if args.use_cuda else "cpu")
    
    # define args.hla_max_len ref to args.seq_type
    if args.seq_type == "short":
        args.hla_max_len = 34
    elif args.seq_type == "whole":
        args.hla_max_len = 372
    elif args.seq_type == "clip":
        args.hla_max_len = 182

    # tgt_len = args.pep_max_len + args.hla_max_len
    # args.tgt_len = tgt_len
    
    dataset_configs = {
        "hla_max_len": args.hla_max_len,
        "pep_max_len": args.pep_max_len,
        "hla_seq_type": args.seq_type,
        "padding": True,
    }

    # Prepare data
    print("Data Preparing")
    # seq_dict = (
    #     pd.read_csv(os.path.join(data_path, "main_task/hla_seq_dict.csv"), index_col=0)
    #     .set_index(["HLA_name"])["HLA_clip"]
    #     .to_dict()
    # )
    # candidate_pools
    candidate_pools_filename = "main_task/allele2candidate_pools.npy" 
    HLA2ranking_candidates = np.load(
        data_path + candidate_pools_filename,
        allow_pickle=True,
    ).item()

    (
        train_pos_df,
        valid_pos_df,
        train_loader,
        val_loader,
    ) = prepare_main_task_loader(
        args.rand_neg,
        args.fold,
        args.batch_size,
        dataset_configs,
        HLA2ranking_candidates,
        data_path,
        num_workers=0
    )# seq_dict, 

    # Prepare PLM
    print("Frozen PLM Preparing")
    plm_type = args.plm_type
    inference_type = args.inference_type
    if plm_type == "tape":
        feature_len = 768
        plm_tokenizer = TAPETokenizer(vocab='iupac')
        plm_model = ProteinBertModel.from_pretrained('bert-base').to(device)
        print("PLM-TAPE is ready")
    elif plm_type == "protbert":
        feature_len = 1024
        plm_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
        plm_model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd").to(device)
        print("PLM-ProtBERT is ready")
    plm_model.eval()
    
    hla2tensor_dict = np.load(
        data_path + "main_task/{}_hla2tensor_{}_dict.npy".format(plm_type, inference_type),
        allow_pickle=True
        ).item()

    plm_configs = {
        "type": plm_type,
        "tokenizer": plm_tokenizer,
        "model": plm_model,
        "inference_type": inference_type,
        "hla2tensor_dict": hla2tensor_dict
    }

    # Prepare model
    print("Build Projection model")
    if args.n_layers == 3:
        model = Projection00(feature_len*2).to(device)
    elif args.n_layers == 4:
        model = Projection10(feature_len*2).to(device)  #1
    elif args.n_layers == 5:
        model = Projection(feature_len*2).to(device)    #4
    # start_epoch = 47
    # model_name = "main_plm_model1_tape_B512_LR2e-05_seq_clip_fold4_ep47_221010.pkl"
    # model.load_state_dict(torch.load(dir_saver + model_name), strict = True)
    print("Ready for training")
    
    train(
            args,
            model,
            train_loader,
            val_loader,
            plm_configs,
            # start_epoch=start_epoch
        )# seq_dict, 

    # for _ in range(5):
    #     for hla_list, pep_seq_list, pep_seq_list_neg in train_loader:
    #         # print(hla_list, pep_seq_list)
    #         break
    #     t7 = time.time()
    #     if args.rand_neg == True:
    #         (
    #             hla_inputs, 
    #             pep_inputs
    #         ) = extract_features_RN(
    #             hla_list, pep_seq_list, pep_seq_list_neg,
    #             inference_type, plm_type, 
    #             plm_tokenizer, plm_model, 
    #             hla2tensor_dict, device,
    #             rand_neg=True
    #         )
    #     else:
    #         (
    #             hla_inputs, 
    #             pep_inputs 
    #         )= extract_features_RN(
    #             hla_list, pep_seq_list, [],
    #             inference_type, plm_type, 
    #             plm_tokenizer, plm_model, 
    #             hla2tensor_dict, device,
    #             rand_neg=False
    #         )
    #     t8 = time.time()
    #     total_t = t8-t7
    #     # print("batch_size_{}, plm_{}, hla and pep uses {:.3f}s, hla_inputs_{}, pep_inputs_{}"\
    #     # .format(batch_size, plm_type, total_t, hla_inputs.shape, pep_inputs.shape))
        
    #     phla_inputs = torch.cat((hla_inputs, pep_inputs), dim=1)
    #     print("batch_size_{}, plm_{}, uses {:.3f}s, hla_inputs_{}, pep_inputs_{}, phla_inputs_{}"\
    #         .format(batch_size, plm_type, total_t, hla_inputs.shape, pep_inputs.shape, phla_inputs.shape))

    # ys_val, loss_val_list, metrics_val = make_evaluation(
    #     args.rand_neg,
    #     model,
    #     train_loader,
    #     "train",
    #     args.threshold,
    #     args.fold,
    #     args.use_cuda,
    # )

    # ys_val, loss_val_list, metrics_val = make_evaluation(
    #     args.rand_neg,
    #     model,
    #     val_loader,
    #     "valid",
    #     args.threshold,
    #     args.fold,
    #     args.use_cuda,
    # )

    # train(
    #     args,
    #     model,
    #     train_loader,
    #     val_loader,
    #     train_pos_df,
    #     test_pos_df,
    #     HLA2ranking_candidates,
    #     seq_dict,
    # )

    # print("Loading best model")
    # print(path_saver)
    # model.load_state_dict(torch.load(path_saver))

    # print(performances_to_pd(metrics_val))

# tape RN 256 2353M