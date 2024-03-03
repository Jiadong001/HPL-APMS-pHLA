'''
Evaluation Protocol:
    1. Evaluation on fixed independent_set and external_set
    2. Evaluation on RN independent_set and external_set (5 times)
    3. (optional) Full ranking on RN independent_set
'''

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

from tape import ProteinBertModel, TAPETokenizer    # TAPE
from transformers import AutoTokenizer, AutoModel   # ProtBERT

import sys 
sys.path.append("..")   # parent directory

from load_data import pHLA_Dataset, pHLA_Dataset_RN, extract_features_RN
from model_v1 import Projection
# from model_v2 import Transformer
from utils import f_mean, performances, transfer

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

def prepare_main_task_loader(
    rand_neg, fold, batch_size, configs, 
    hla_seq_dict, hla2candidates, data_path, num_workers, 
    only_fullranking=False, fullranking_type="test"
    ):

    main_train_df = pd.read_csv(
        os.path.join(data_path, "main_task/train_data_fold{}.csv".format(fold)), index_col=0
    )
    main_external_df = pd.read_csv(
        os.path.join(data_path, "main_task/external_set.csv"), index_col=0
    )

    if fullranking_type == "test":
        main_test_df = pd.read_csv(
            os.path.join(data_path, "main_task/independent_set.csv"), index_col=0
        )
    elif fullranking_type == "zeroshot":
        main_test_df = pd.read_csv(
            os.path.join(data_path, "main_task/zeroshot_set.csv"), index_col=0
        )
        # main_test_df["clip"] = main_test_df["HLA"].map(lambda x: hla_seq_dict[x])
    
    if rand_neg:
        train_pos_df = main_train_df[main_train_df.label == 1]
        test_pos_df = main_test_df[main_test_df.label == 1]
        
        if only_fullranking:
            return (
                train_pos_df,
                test_pos_df,
            )
        else:
            external_pos_df = main_external_df[main_external_df.label == 1]

            test_dataset = pHLA_Dataset_RN(test_pos_df, hla2candidates, configs)
            external_dataset = pHLA_Dataset_RN(external_pos_df, hla2candidates, configs)

            test_loader = Data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
            )
            external_loader = Data.DataLoader(
                external_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
            )
            return (
                train_pos_df,
                test_pos_df,
                test_loader,
                external_loader,
            )
    else:
        test_dataset = pHLA_Dataset(main_test_df, configs)
        external_dataset = pHLA_Dataset(main_external_df, configs)

        test_loader = Data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )
        external_loader = Data.DataLoader(
            external_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )
        return (
            test_loader,
            external_loader,
        )

def make_evaluation(
    rand_neg, model, model_type, loader, set_type, threshold, fold, use_cuda, plm_configs
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
            pbar.set_description(f"EVALUATION without random negative samples")
            
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
                
                if model_type == "ffn":
                    val_outputs = model(
                        torch.cat((hla_inputs, pep_inputs), dim=1)
                    )
                elif model_type == "decoder":
                    val_outputs, val_dec_self_attns = model(
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
                "Fold-{} ******{} on {} set****** : Loss = {:.6f}".format(
                    fold, "EVALUATION", set_type, f_mean(loss_val_list)
                )
            )
            metrics_val = performances(
                y_true_val_list, y_pred_val_list, y_prob_val_list, print_=True
            )
    else:
        with torch.no_grad():
            # print(len(loader))
            pbar = tqdm(loader)
            pbar.set_description(f"EVALUATION with random negative samples")
            for hla_list, pep_seq_list, pep_seq_list_neg in pbar:
                batch_num = len(hla_list)
                hla_inputs, pep_inputs = extract_features_RN(
                    hla_list, pep_seq_list, pep_seq_list_neg,
                    inference_type, plm_type, 
                    plm_tokenizer, plm_model, 
                    hla2tensor_dict, device,
                    rand_neg=True
                ) 

                if model_type == "ffn":
                    val_outputs = model(
                        torch.cat((hla_inputs, pep_inputs), dim=1)
                    )
                elif model_type == "decoder":
                    val_outputs, val_dec_self_attns = model(
                        torch.cat((hla_inputs, pep_inputs), dim=1)
                    )

                y_true_val = torch.LongTensor([1] * batch_num + [0] * batch_num).to(
                    device
                )
                val_loss = criterion(val_outputs, y_true_val)

                y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
                y_true_val = y_true_val.cpu().numpy()
                
                y_true_val_list.extend(y_true_val)
                y_prob_val_list.extend(y_prob_val)
                loss_val_list.append(val_loss)
                # dec_attns_val_list.append(val_dec_self_attns)

            y_pred_val_list = transfer(y_prob_val_list, threshold)
            ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)
            # loss_val_list.append(val_loss)
            print(
                "Fold-{} ******{} on {} set****** : Loss = {:.6f}".format(
                    fold, "EVALUATION", set_type, f_mean(loss_val_list)
                )
            )
            metrics_val = performances(
                y_true_val_list, y_pred_val_list, y_prob_val_list, print_=True
            )

    return ys_val, loss_val_list, metrics_val  # , dec_attns_val_list

def full_ranking(
    model, model_type,
    HLA_name, hla_seq_dict,
    test_pos_peptides,
    candidate_neg_pep_pool,
    use_cuda,
    hla_max_len,
    pep_max_len,
    topk,
    inf_batch,
    plm_configs,
    fullranking_type,
    bottom_k=5,
):

    plm_type = plm_configs["type"]
    plm_tokenizer = plm_configs["tokenizer"]
    plm_model = plm_configs["model"]
    inference_type = plm_configs["inference_type"]
    hla2tensor_dict = plm_configs["hla2tensor_dict"]
    
    device = torch.device("cuda" if use_cuda else "cpu")
    ranking_pool = sorted(
        list(candidate_neg_pep_pool.union(test_pos_peptides)))

    # random negative sampling
    assert len(candidate_neg_pep_pool) + len(test_pos_peptides) == len(ranking_pool)
    print(
        "Ranking {} samples, Targeting {} positive samples".format(
            len(ranking_pool), len(test_pos_peptides)
        )
    )

    ## hla
    if fullranking_type == "test":
        HLA_inputs = hla2tensor_dict[HLA_name]
    elif fullranking_type == "zeroshot":
        HLA_seq = hla_seq_dict[HLA_name]
        HLA_seq = HLA_seq.ljust(hla_max_len, 'X')
        if plm_type == "protbert":
            HLA_seq = ' '.join(HLA_seq)
        HLA_inputs = torch.LongTensor([plm_tokenizer.encode(HLA_seq)]).to(device)
        print(HLA_inputs.shape)
        
        if inference_type == "full":
            HLA_inputs = plm_model(HLA_inputs)[0]      # full_output
        elif inference_type == "pooled":
            HLA_inputs = plm_model(HLA_inputs)[1]      # pooled_output
        elif inference_type == "mean":
            HLA_inputs = plm_model(HLA_inputs)[0]
            HLA_inputs = torch.mean(HLA_inputs, dim=1) # mean_output
        print(HLA_inputs.shape)

    ## pep Tokenizer
    candidate_pep_token_ids = []
    for seq in ranking_pool:
        seq = seq.ljust(pep_max_len, 'X')
        if plm_type == "protbert":
            seq = ' '.join(seq)
        candidate_pep_token_ids.append(plm_tokenizer.encode(seq))
    
    start_index = 0
    end_index = inf_batch

    model.eval()
    y_prob_all = []
    y_val_all = []
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
            
            if model_type == "ffn":
                val_outputs = model(
                    torch.cat((val_HLA_inputs, val_pep_inputs), dim=1)
                    )
            elif model_type == "decoder":
                val_outputs, val_dec_self_attns = model(
                    torch.cat((val_HLA_inputs, val_pep_inputs), dim=1)
                    )

            ### save details
            # y_val_all.append(val_outputs.cpu().detach())

            # y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach()
            y_prob_val = val_outputs[:, 1] - val_outputs[:, 0]
            y_prob_all.append(y_prob_val.cpu().detach())
            
            start_index = end_index
            if end_index + inf_batch < len(ranking_pool):
                end_index += inf_batch
            else:
                end_index = len(ranking_pool)

        y_prob_all = torch.cat(y_prob_all, dim=0)
 
        _, index_of_rank_list = torch.topk(y_prob_all, len(ranking_pool))

        recall_peps = [ranking_pool[_] for _ in index_of_rank_list.numpy()]
        positive_ranks = sorted(
            [recall_peps.index(pos_pep) for pos_pep in test_pos_peptides]
        )[-bottom_k:]
        
        ### save details
        # y_val_all = torch.cat(y_val_all, dim=0)
        # y_prob_all = y_prob_all.numpy()
        # y_val_all = y_val_all.numpy()
        # print("\nscore=1: {}/{}\n".format(
        #     len(y_prob_all[y_prob_all==1.]), len(recall_peps)
        #     ))
        
        # print("pep_seq","\t\t",
        #     "len","\t\t",
        #     "val_output0","\t\t",
        #     "val_output1","\t\t",
        #     "score","\t\t",
        #     "ground truth"
        #     )
        # for j in range(5000):
        #     ground_truth = [recall_peps[j]==pos_pep for pos_pep in test_pos_peptides]
        #     ground_truth = sum(ground_truth)
        #     print(
        #         recall_peps[j],"\t\t",
        #         len(recall_peps[j]),"\t\t",
        #         y_val_all[index_of_rank_list[j]][0],"\t\t",
        #         y_val_all[index_of_rank_list[j]][1],"\t\t",
        #         y_prob_all[index_of_rank_list[j]],"\t\t",
        #         ground_truth
        #         )
        
        # print(len(recall_peps))
        # print(recall_peps[0])
        # print(list(test_data)[0])
        # print(len(test_data))
        # print(len(test_data.intersection(set(candidate_pep))))

        recall = np.array([0.0] * len(topk))
        hit = np.array([0.0] * len(topk)) 
        num_pos = len(test_pos_peptides)
        for ind, k in enumerate(topk):
            num_hit = len(test_pos_peptides.intersection(set(recall_peps[:k])))
            recall[ind] += num_hit / num_pos
            hit[ind] += num_hit

        print("Recall@K")
        print(recall)
        print("Hit@K")
        print(hit)
        print("Positive Bottom Rank")
        print(f_mean(positive_ranks),"\n")

        return recall, hit

def full_ranking_test(
    model, model_type,
    train_pos_df,
    test_pos_df,
    hla_seq_dict, HLA2ranking_candidates,
    args,
    plm_configs,
    log_name,
    fullranking_type,
    topk=[50, 100, 500, 1000, 5000, 10000, 100000],
):
    HLA_list = list(test_pos_df.HLA.unique())           # deduplication, HLA name
    print("HLA allele num: {}".format(len(HLA_list)))

    recall_arr = np.array([0.0] * len(topk))

    if fullranking_type == "test":
        train_length = len(train_pos_df)                # for weight calculation
        recall_arr_weighted = np.array([0.0] * len(topk))

    recall_results, hit_results, pos_num = [], [], []
    order = 0
    for current_HLA in HLA_list[:]:
        ranking_candidates = HLA2ranking_candidates[current_HLA]
        assert isinstance(ranking_candidates, set)

        pos_test_pep = set(list(test_pos_df[test_pos_df["HLA"] == current_HLA].peptide))

        HLA_seq = hla_seq_dict[current_HLA]
        print(current_HLA, HLA_seq)

        order = order+1
        print("\n{}-full ranking on {}".format(order, current_HLA))

        t1=time.time()
        recall_cur, hit_cur = full_ranking(
            model, model_type,
            current_HLA, hla_seq_dict,
            pos_test_pep,
            ranking_candidates,
            args.use_cuda,
            args.hla_max_len,
            args.pep_max_len,
            topk,
            args.batch_size*2,
            plm_configs,
            fullranking_type
        )
        t2 = time.time()
        print("one HLA----{:.6f}s".format(t2-t1))

        recall_results.append(recall_cur)
        hit_results.append(hit_cur)
        pos_num.append(len(pos_test_pep))
        recall_arr += recall_cur / len(HLA_list)

        if fullranking_type == "test":
            train_length_cur_HLA = len(train_pos_df[train_pos_df["HLA"] == current_HLA])
            recall_arr_weighted += recall_cur * train_length_cur_HLA / train_length

    
    print("\n===================================================\n")
    print("======================RECALL@K=====================")
    print(recall_arr)

    if fullranking_type == "test":
        print("======================Weighted RECALL@K=====================")
        print(recall_arr_weighted)

    col_names = ['recall@' + str(_) for _ in topk]
    recall_results_df = pd.DataFrame(recall_results, columns=col_names)
    recall_results_df['HLA'] = HLA_list

    col_names = ['hit@' + str(_) for _ in topk]
    hit_results_df = pd.DataFrame(hit_results, columns=col_names)
    hit_results_df['pos_num'] = pos_num
    
    results_df = pd.concat((recall_results_df, hit_results_df), axis=1)
    results_df.to_csv("../fullranking_csvs/{}/".format(fullranking_type) + log_name)

def evaluate(
    args,
    model, model_type,
    test_loader, external_loader,
    test_RN_loader, external_RN_loader,
    train_positive_data, test_positive_data,
    hla_seq_dict, HLA2ranking_candidates,
    plm_configs,
    do_eval,
    do_fullranking,
    log_name,
    fullranking_type,
    RN_validation_times=5,
):

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

    if do_eval:
        # Evaluation on fixed set
        ## independent_set
        performance_val_df = pd.DataFrame()
        ys_val, loss_val_list, metrics_val = make_evaluation(
            False,
            model, model_type,
            test_loader,
            "test(independent)",
            args.threshold,
            args.fold,
            args.use_cuda,
            plm_configs
        )  # , dec_attns_val
        
        performance_val_df = pd.concat(
            [
                performance_val_df,
                pd.DataFrame(
                    [list(metrics_val)],
                    columns=metrics_name,
                ),
            ]
        )
        val_mertrics_avg = (sum(metrics_val[:4]) / 4)
        cur_epoch_performance_df = performance_val_df.iloc[
            -5:,
        ]
        print(
            "Evaluation of Epoch: AUC = {}, ACC = {}, 'MCC = {}, F1 = {}".format(
                cur_epoch_performance_df.roc_auc.mean(),
                cur_epoch_performance_df.accuracy.mean(),
                cur_epoch_performance_df.mcc.mean(),
                cur_epoch_performance_df.f1.mean(),
            )
        )
        print(f"Average result: {val_mertrics_avg}\n")

        ## external_set
        performance_val_df = pd.DataFrame()
        ys_val, loss_val_list, metrics_val = make_evaluation(
            False,
            model, model_type,
            external_loader,
            "external",
            args.threshold,
            args.fold,
            args.use_cuda,
            plm_configs
        )  # , dec_attns_val
        
        performance_val_df = pd.concat(
            [
                performance_val_df,
                pd.DataFrame(
                    [list(metrics_val)],
                    columns=metrics_name,
                ),
            ]
        )
        val_mertrics_avg = (sum(metrics_val[:4]) / 4)
        cur_epoch_performance_df = performance_val_df.iloc[
            -5:,
        ]
        print(
            "Evaluation of Epoch: AUC = {}, ACC = {}, 'MCC = {}, F1 = {}".format(
                cur_epoch_performance_df.roc_auc.mean(),
                cur_epoch_performance_df.accuracy.mean(),
                cur_epoch_performance_df.mcc.mean(),
                cur_epoch_performance_df.f1.mean(),
            )
        )
        print(f"Average result: {val_mertrics_avg}\n")


        # Evaluation on RN set
        ## independent_set
        performance_val_df = pd.DataFrame()
        val_mertrics_avg = []
        for val_time in range(RN_validation_times):
            ys_val, loss_val_list, metrics_val = make_evaluation(
                True,
                model, model_type,
                test_RN_loader,
                "test(independent)",
                args.threshold,
                args.fold,
                args.use_cuda,
                plm_configs
            )  # , dec_attns_val
            
            performance_val_df = pd.concat(
                [
                    performance_val_df,
                    pd.DataFrame(
                        [[str(val_time)] + list(metrics_val)],
                        columns=["rand_val_num"] + metrics_name,
                    ),
                ]
            )
            val_mertrics_avg.append(sum(metrics_val[:4]) / 4)
        
        cur_epoch_performance_df = performance_val_df.iloc[
            -5:,
        ]
        print(
            "Evaluation of Epoch: AUC_avg = {}, ACC_avg = {}, 'MCC_avg = {}, F1-avg = {}".format(
                cur_epoch_performance_df.roc_auc.mean(),
                cur_epoch_performance_df.accuracy.mean(),
                cur_epoch_performance_df.mcc.mean(),
                cur_epoch_performance_df.f1.mean(),
            )
        )
        avg_val = f_mean(val_mertrics_avg)
        print(f"Average result: {avg_val}\n")

        ## external_set
        performance_val_df = pd.DataFrame()
        val_mertrics_avg = []
        for val_time in range(RN_validation_times):
            ys_val, loss_val_list, metrics_val = make_evaluation(
                True,
                model, model_type,
                external_RN_loader,
                "external",
                args.threshold,
                args.fold,
                args.use_cuda,
                plm_configs
            )  # , dec_attns_val
            
            performance_val_df = pd.concat(
                [
                    performance_val_df,
                    pd.DataFrame(
                        [[str(val_time)] + list(metrics_val)],
                        columns=["rand_val_num"] + metrics_name,
                    ),
                ]
            )
            val_mertrics_avg.append(sum(metrics_val[:4]) / 4)
        
        cur_epoch_performance_df = performance_val_df.iloc[
            -5:,
        ]
        print(
            "Evaluation of Epoch: AUC_avg = {}, ACC_avg = {}, 'MCC_avg = {}, F1-avg = {}".format(
                cur_epoch_performance_df.roc_auc.mean(),
                cur_epoch_performance_df.accuracy.mean(),
                cur_epoch_performance_df.mcc.mean(),
                cur_epoch_performance_df.f1.mean(),
            )
        )
        avg_val = f_mean(val_mertrics_avg)
        print(f"Average result: {avg_val}\n")


    # Full-ranking Test on Independent_set
    if do_fullranking:
        print("============================================================")
        print("Start Full-ranking on {} set".format(fullranking_type))
        print("============================================================")

        full_ranking_test(
            model, model_type,
            train_positive_data,
            test_positive_data,
            hla_seq_dict, HLA2ranking_candidates,
            args,
            plm_configs,
            log_name,
            fullranking_type
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test the best model, with HLA clip sequence"
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
    parser.add_argument(
        "--model_type", type=str, choices=["ffn", "decoder", "finetune"], default="ffn"
    )
    parser.add_argument(
        "--fullranking_type", type=str, choices=[
                                            "test",
                                            "zeroshot", 
                                            ], default="test"
    )
    parser.add_argument(
        "--eval_options", type=str, choices=["eval", "fullranking", "all"], default="all"
    )
    parser.add_argument("--batch_size", type=int, default=512)

    # -----------------Parameters for model 2-------------------
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=9)
    parser.add_argument("--d_ff", type=int, default=2048)
    
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
    
    dataset_configs = {
        "hla_max_len": args.hla_max_len,
        "pep_max_len": args.pep_max_len,
        "hla_seq_type": args.seq_type,
        "padding": True,
    }

    # eval options
    do_eval = do_fullranking = True
    if args.eval_options == "eval":
        do_fullranking = False
    elif args.eval_options == "fullranking":
        do_eval = False

    # Prepare data
    print("Data Preparing")
    hla_seq_dict = pd.read_csv(os.path.join(
        data_path, "main_task/HLA_sequence_dict_new.csv"), 
        index_col=0
        ).set_index(["HLA_name"])[args.seq_type].to_dict()
    
    if args.fullranking_type == "test":
        HLA2ranking_candidates = np.load(
            data_path + "main_task/allele2candidate_pools.npy",
            allow_pickle=True,
            ).item()
    elif args.fullranking_type == "zeroshot":
        HLA2ranking_candidates = np.load(
            data_path + "main_task/zeroshot_allele2candidate_pools.npy",
            allow_pickle=True,
            ).item()
        assert do_eval==False
    
    if do_eval:
        # fixed dataset
        (
            test_loader, external_loader,
        ) = prepare_main_task_loader(
            False,
            args.fold,
            args.batch_size,
            dataset_configs,
            HLA2ranking_candidates,
            data_path,
            num_workers=0
        )
        # dataset with RN sampling
        (
            train_pos_df, test_pos_df, 
            test_RN_loader, external_RN_loader,
        ) = prepare_main_task_loader(
            True,
            args.fold,
            args.batch_size,
            dataset_configs,
            hla_seq_dict,
            HLA2ranking_candidates,
            data_path,
            num_workers=0,
            only_fullranking=False
        )
    else:
        test_loader = external_loader = []
        
        # must do fullranking (only fullranking)
        (
            train_pos_df, test_pos_df
        ) = prepare_main_task_loader(
            True,
            args.fold,
            args.batch_size,
            dataset_configs,
            hla_seq_dict,
            HLA2ranking_candidates,
            data_path,
            num_workers=0,
            only_fullranking=True,
            fullranking_type=args.fullranking_type
        )
        test_RN_loader = external_RN_loader = []
        
    # Prepare PLM
    print("PLM Preparing")
    plm_type = args.plm_type
    inference_type = args.inference_type
    if plm_type == "tape":
        feature_len = 768
        args.d_model = args.d_k = args.d_v = 768
        plm_tokenizer = TAPETokenizer(vocab='iupac')
        plm_model = ProteinBertModel.from_pretrained('bert-base').to(device)
        print("PLM-TAPE is ready")
    elif plm_type == "protbert":
        feature_len = 1024
        args.d_model = args.d_k = args.d_v = 1024
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
    print("Best Model Preparing")
    if args.model_type == "ffn":
        model = Projection(feature_len*2).to(device)
        model_name = "main_plm_model1_tape_B512_LR2e-05_seq_clip_fold4_ep109_221010.pkl"          # tape
        # model_name = "main_plm_model1_protbert_B512_LR0.0002_seq_clip_fold4_ep137_221012.pkl"       # protbert
        model.load_state_dict(torch.load(dir_saver + model_name), strict = True)
    # elif args.model_type == "decoder":
    #     args.tgt_len = args.pep_max_len + args.hla_max_len + 2 + 2
    #     model = Transformer(args).to(device)
    #     model_name = "main_plm_model2_tape_B64_LR5e-05_seq_clip_fold4_ep175_221014.pkl"          # tape
    #     model.load_state_dict(torch.load(dir_saver + model_name), strict = True)

    print("Ready for testing")

    t1 = time.time()
    log_name = "{}mer_RN_model12_{}_{}csv".format(
        args.hla_max_len, plm_type, args.model_type
        )
    evaluate(
            args, model, args.model_type,
            test_loader, external_loader,
            test_RN_loader, external_RN_loader,
            train_pos_df, test_pos_df,
            hla_seq_dict, HLA2ranking_candidates,
            plm_configs,
            do_eval,
            do_fullranking,
            log_name,
            args.fullranking_type
        )
    t2 = time.time()
    print("\n Test uses: {}s".format(t2-t1))
