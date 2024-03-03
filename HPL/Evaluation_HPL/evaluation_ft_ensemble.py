"""
@xinyuanzhu

Evaluate the trained model with multiple protocols
0. fixed validation data (the Recall metric can be used to check the model loading compared with the training log)
1. fixed test data (provided by TransPHLA as "independent_set.csv")
2. fixed external data (provided by TransPHLA as "external_set.csv")
3. positive data in 1) and random negative samples. N times to take the average
4. full ranking protocol. 

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
import torch.utils.data as Data
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tape import ProteinBertConfig, TAPETokenizer

import sys 
sys.path.append("..")   # parent directory

from load_data_ft import pHLA_Dataset, pHLA_Dataset_RN, seq2token
from utils import f_mean, performances, transfer
from model_ft import meanTAPE, clsTAPE

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

target_single_hla = [   
                        "HLA-A*02:01",
                        "HLA-A*03:02", "HLA-A*11:02", "HLA-B*42:01", "HLA-B*08:02",
                        "HLA-G*01:01", "HLA-E*01:03",
                        "HLA-A*33:03", "HLA-A*34:02"
                    ]

def prepare_main_task_loader(
    seq_type, fold, batch_size, data_path, 
    configs, seq_dict, hla2candidates, num_workers, 
    only_fullranking=False,
    fullranking_type="test"
    ):

    main_train_df = pd.read_csv(os.path.join(
        data_path, "main_task/train_data_fold{}.csv".format(fold)), 
        index_col=0
    )
    
    if not only_fullranking:
        main_valid_df = pd.read_csv(os.path.join(
            data_path, "main_task/val_data_fold{}.csv".format(fold)),
            index_col=0
        )
        main_external_df = pd.read_csv(os.path.join(
            data_path, "main_task/external_set.csv"), 
            index_col=0
        )
        main_test_df = pd.read_csv(os.path.join(
            data_path, "main_task/independent_set.csv"), 
            index_col=0
        )
        for df in [main_train_df, main_valid_df, main_test_df, main_external_df]:
            df.rename(columns={"HLA_sequence": "short"}, inplace=True)
            df[seq_type] = df["HLA"].map(lambda x: seq_dict[x])
    else:
        if fullranking_type == "test":
            main_test_df = pd.read_csv(os.path.join(
                data_path, "main_task/independent_set.csv"), 
                index_col=0
            )
        elif fullranking_type == "zeroshot_eg":
            main_test_df = pd.read_csv(os.path.join(
                data_path, "main_task/zeroshot_set.csv"), 
                index_col=0
            ) 
        elif fullranking_type == "zeroshot_abc":
            main_test_df = pd.read_csv(os.path.join(
                data_path, "main_task/zeroshot_abc_set.csv"), 
                index_col=0
            )
        for df in [main_train_df, main_test_df]:
            df[seq_type] = df["HLA"].map(lambda x: seq_dict[x])
    
    
    # full ranking's positive samples
    train_pos_df = main_train_df[main_train_df.label == 1]
    test_pos_df = main_test_df[main_test_df.label == 1]
    print("all positive samples of fullranking:", len(test_pos_df))

    if args.fullranking_target == "single":
        test_pos_df = test_pos_df[test_pos_df["HLA"].isin( [args.target_hla] )]
    print("target positive samples of fullranking:", len(test_pos_df))

    if only_fullranking:
        return train_pos_df, test_pos_df
    
    # fixed dataset
    valid_dataset = pHLA_Dataset(main_valid_df, configs)
    test_dataset = pHLA_Dataset(main_test_df, configs)
    external_dataset = pHLA_Dataset(main_external_df, configs)

    valid_loader = Data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
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

    # RN dataset
    test_dataset_RN = pHLA_Dataset_RN(test_pos_df, hla2candidates, configs)

    test_loader_RN = Data.DataLoader(
        test_dataset_RN,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    return (
        train_pos_df, test_pos_df,
        valid_loader, test_loader, external_loader,
        test_loader_RN
    )

def make_evaluation(
    rand_neg, model, loader, threshold, fold, 
    plm_type, plm_input_type, device, desc
):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_val_list, dec_attns_val_list = [], []
    y_true_val_list, y_prob_val_list = [], []

    pbar = tqdm(loader)
    pbar.set_description(desc)

    if not rand_neg:
        with torch.no_grad():
            for hla_list, pep_seq_list, val_labels in pbar:
                
                phla_tokens = seq2token(
                            hla_list, 
                            pep_seq_list,
                            plm_type,
                            plm_input_type,
                            device
                        )
                y_true_val = val_labels.to(device)
                
                val_outputs = model(phla_tokens)
                val_loss = criterion(val_outputs, y_true_val)

                y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
                y_true_val = val_labels.numpy()

                y_true_val_list.extend(y_true_val)
                y_prob_val_list.extend(y_prob_val)
                loss_val_list.append(val_loss)
    else:
        with torch.no_grad():
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
        "\nFold-{} ******{}****** : Loss = {:.6f}".format(
            fold, desc, ave_loss_val
        )
    )
    metrics_val = performances( y_true_val_list,
                                y_pred_val_list, 
                                y_prob_val_list, 
                                print_=True)        # print performances

    return ys_val, ave_loss_val, metrics_val

def full_ranking(
    HLA_seq, test_pos_peptides, candidate_neg_pep_pool,
    hla_max_len, pep_max_len,
    model_list, inf_batch, plm_type, plm_input_type, device,
    topk, bottom_k=5,
):  

    # cat neg and pos samples for HLA in the pool
    ranking_pool = sorted(
        list(candidate_neg_pep_pool.union(test_pos_peptides)))
    assert len(candidate_neg_pep_pool) + len(test_pos_peptides) == len(ranking_pool)
    print("Ranking {} samples, Targeting {} positive samples".format(
            len(ranking_pool), len(test_pos_peptides)))

    # seq tokenizer
    if plm_type == "tape":
        tokenizer = TAPETokenizer(vocab='iupac')
    
    hla_token = tokenizer.encode(HLA_seq.ljust(hla_max_len, 'X'))   # array
    if plm_input_type == "cat":
        hla_token = hla_token[:-1]                                  # remove <sep> token

    candidate_pep_tokens = []
    for seq in ranking_pool:
        pep_token = tokenizer.encode(seq.ljust(pep_max_len, 'X'))
        candidate_pep_tokens.append(pep_token[1:])                  # remove <cls> token

    # evaluation
    with torch.no_grad():
        y_prob_all_ensemble = None
        for ind, model in enumerate(model_list):
            start_index = 0
            end_index = inf_batch
            model.eval()
            y_prob_all = []
            while end_index <= len(ranking_pool) and start_index < end_index:
                batch_tokens = []
                for pep_token in candidate_pep_tokens[start_index:end_index]:
                    phla_token = np.concatenate((hla_token, pep_token))
                    batch_tokens.append(phla_token)
                batch_tokens = torch.LongTensor(batch_tokens).to(device)
                val_outputs = model(batch_tokens)
                
                # y_prob_val = nn.Softmax(dim=1)(val_outputs)[:, 1]  # torch.float32
                y_prob_val = val_outputs[:, 1] - val_outputs[:, 0]
                y_prob_all.append(y_prob_val.cpu().detach())            
            
                start_index = end_index
                if end_index + inf_batch < len(ranking_pool):
                    end_index += inf_batch
                else:
                    end_index = len(ranking_pool)

            y_prob_all = torch.cat(y_prob_all, dim=0)
            print(y_prob_all[:10])
            if ind == 0:
                y_prob_all_ensemble = y_prob_all
            else:
                y_prob_all_ensemble = y_prob_all_ensemble + y_prob_all
            print(y_prob_all_ensemble[:10])

    _, index_of_rank_list = torch.topk(y_prob_all_ensemble, len(ranking_pool))

    recall_peps = [ranking_pool[_] for _ in index_of_rank_list.numpy()]     # sorted peptide list
    positive_ranks = sorted(
        [recall_peps.index(pos_pep) for pos_pep in test_pos_peptides]       # positive peptides' rank
    )
    bottom_pos_ranks = positive_ranks[-bottom_k:]

    recall = np.array([0.0] * len(topk))
    hit = np.array([0] * len(topk)) 
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
    print(f_mean(bottom_pos_ranks))
    # print("All Posiive Ranks")
    # print(positive_ranks,"\n")

    return recall, hit

def full_ranking_test(
    args,
    train_pos_df, test_pos_df,
    hla_seq_dict, HLA2ranking_candidates,
    model, plm_input_type,
    log_name,
    fullranking_type,
    topk=[50, 100, 500, 1000, 5000, 10000, 100000],
):

    HLA_list = list(test_pos_df.HLA.unique())               # deduplication, HLA name
    target_HLA_list = []
    if args.fullranking_target == "all":
        target_HLA_list = HLA_list
    elif args.fullranking_target == "single":
        if args.target_hla in HLA_list:
            target_HLA_list.append(args.target_hla)
            print("Target hla-{} is in hla list".format(target_HLA_list))
    else:
        for hla in HLA_list:
            if hla.find(args.fullranking_target)!=-1:
                target_HLA_list.append(hla)
    print("HLA allele num: {}/{}".format(len(target_HLA_list), len(HLA_list)))
    print(target_HLA_list)

    recall_arr = np.array([0.0] * len(topk))
    
    if fullranking_type == "test":
        train_length = len(train_pos_df)                    # for weight calculation
        recall_arr_weighted = np.array([0.0] * len(topk))

    recall_results, hit_results, pos_num = [], [], []
    order = 0
    for current_HLA in target_HLA_list:
        ranking_candidates = HLA2ranking_candidates[current_HLA]
        assert isinstance(ranking_candidates, set)

        pos_test_pep = set(
            list(test_pos_df[test_pos_df["HLA"] == current_HLA].peptide))

        HLA_seq = hla_seq_dict[current_HLA]
        # print(current_HLA, HLA_seq)

        order = order+1
        print("\n{}-full ranking on {}".format(order, current_HLA))

        t1=time.time()
        recall_cur, hit_cur = full_ranking(
            HLA_seq, pos_test_pep, ranking_candidates,
            args.hla_max_len, args.pep_max_len,
            model, args.batch_size, plm_type, plm_input_type, device,
            topk, bottom_k=5,
        )
        t2 = time.time()
        print("one HLA----{:.6f}s".format(t2-t1))
        
        recall_results.append(recall_cur)
        hit_results.append(hit_cur)
        pos_num.append(len(pos_test_pep))
        recall_arr += recall_cur / len(target_HLA_list)

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
    recall_results_df['HLA'] = target_HLA_list

    col_names = ['hit@' + str(_) for _ in topk]
    hit_results_df = pd.DataFrame(hit_results, columns=col_names)
    hit_results_df['pos_num'] = pos_num
    
    results_df = pd.concat((recall_results_df, hit_results_df), axis=1)
    if args.fullranking_target == "single":
        results_df.to_csv(
            "../fullranking_csvs/{}/HPL-Allele/".format(fullranking_type) + log_name)

def evaluate(
    args,
    model, plm_type, plm_input_type,
    valid_loader, test_loader, external_loader,
    test_loader_RN,
    RN_test_times=5,
):

    # valid
    ys_val, loss_val, metrics_val = make_evaluation(
                                        False,
                                        model,
                                        valid_loader,
                                        args.threshold,
                                        args.fold,
                                        plm_type,
                                        plm_input_type,
                                        device,
                                        "Validation"
                                    )

    # test
    ys_val, loss_val, metrics_val = make_evaluation(
                                        False,
                                        model,
                                        test_loader,
                                        args.threshold,
                                        args.fold,
                                        plm_type,
                                        plm_input_type,
                                        device,
                                        "Test"
                                    )

    # external
    ys_val, loss_val, metrics_val = make_evaluation(
                                        False,
                                        model,
                                        external_loader,
                                        args.threshold,
                                        args.fold,
                                        plm_type,
                                        plm_input_type,
                                        device,
                                        "External"
                                    )

    # test_RN
    avg_mertric = np.array([0.0] * 9)
    for i in range(RN_test_times):
        ys_val, loss_val, metrics_val = make_evaluation(
                                            True,
                                            model,
                                            test_loader_RN,
                                            args.threshold,
                                            args.fold,
                                            plm_type,
                                            plm_input_type,
                                            device,
                                            "Test-RN"
                                        )
        avg_mertric += metrics_val                  # metrics_val is also np

    (roc_auc, accuracy, mcc, f1, sensitivity, specificity, precision, recall,
     aupr) = avg_mertric / RN_test_times
    
    print(
        "\n*************************** {} times random negative test average performance ***************************"
        .format(RN_test_times))
    print(
        "auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}"
        .format(roc_auc, sensitivity, specificity, accuracy, mcc))
    print("precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}".format(
        precision, recall, f1, aupr))

def read_argument():
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
        "--plm_input_type", type=str, choices=["cat", "sep"], default="cat"
    )
    parser.add_argument(
        "--plm_output_type", type=str, choices=["mean", "cls"], default="cls"
    )
    parser.add_argument(
        "--head_type", type=str, default="2mlp"
    )

    # -----------------Parameters for testing--------------------
    parser.add_argument(
        "--eval_options", type=str, choices=[
                                            "fullranking",
                                            "all", 
                                            "evaluate"
                                            ], default="all"
    )
    parser.add_argument(
        "--fullranking_type", type=str, choices=[
                                            "test",
                                            "zeroshot_abc", 
                                            "zeroshot_eg",
                                            ], default="test"
    )
    parser.add_argument(
        "--fullranking_target", type=str, choices=[
                                            "HLA-A",
                                            "HLA-B", 
                                            "HLA-C",
                                            "HLA-E",
                                            "HLA-G",
                                            "all",
                                            "single"
                                            ], default="all"
    )
    parser.add_argument(
        "--target_hla", type=str, choices=target_single_hla, default="HLA-A*02:01"   # HLA-A*02:01 is in train/test set 
    )
    
    parser.add_argument("--batch_size", type=int, default=512)  # 10755M
    parser.add_argument("--l_r", type=str, default="1e-05")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--date", type=str, default="1101")
    parser.add_argument("--B", type=int, default=32)
  
    args = parser.parse_args()
    print(args)
    
    return args

if __name__ == "__main__":

    args = read_argument()

    # path
    data_path = args.data_path
    dir_saver = args.model_path

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

    # Prepare data
    print("Data Preparing")
    hla_seq_dict = pd.read_csv(os.path.join(
        data_path, "main_task/HLA_sequence_dict_ABCEG.csv"), 
        index_col=0
        ).set_index(["HLA_name"])[args.seq_type].to_dict()
    
    if args.eval_options == "all" or args.eval_options == "evaluate":
        HLA2ranking_candidates = np.load(
            data_path + "main_task/allele2candidate_pools.npy",
            allow_pickle=True,
            ).item()
    elif args.eval_options == "fullranking":
        if args.fullranking_type == "test":
            HLA2ranking_candidates = np.load(
                data_path + "main_task/allele2candidate_pools.npy",
                allow_pickle=True,
                ).item()
        elif args.fullranking_type == "zeroshot_eg":
            HLA2ranking_candidates = np.load(
                data_path + "main_task/zeroshot_allele2candidate_pools.npy",
                allow_pickle=True,
                ).item()
        elif args.fullranking_type == "zeroshot_abc":
            HLA2ranking_candidates = np.load(
                data_path + "main_task/zs_new_abc_allele2candidate_pools.npy",
                allow_pickle=True,
                ).item()
    
    # Load model
    ## ABC
    print("Model Preparing")
    device = torch.device("cuda" if args.use_cuda else "cpu")
    tape_config = ProteinBertConfig.from_pretrained('bert-base')
    ## gene
    if plm_output_type == "mean":
        model1 = meanTAPE(tape_config, head_type).to(device)
    elif plm_output_type == "cls":
        model1 = clsTAPE(tape_config, head_type).to(device)
    # model1_filename = "HPL-Cluster/A_gene/main_finetune_plm_tape_B24_LR1e-05_seq_clip_fold4_ep17_221123.pkl"
    # model1_filename = "HPL-Cluster/B_gene/main_finetune_plm_tape_B24_LR1e-05_seq_clip_fold4_ep10_221124.pkl"
    model1_filename = "OneStep-Cluster/B_gene/main_finetune_plm_tape_B32_LR3e-05_seq_clip_fold4_ep37_221112.pkl"
    model1.load_state_dict(torch.load(dir_saver + model1_filename), strict = True)
    model1.eval()
    ## supertype
    if plm_output_type == "mean":
        model2 = meanTAPE(tape_config, head_type).to(device)
    elif plm_output_type == "cls":
        model2 = clsTAPE(tape_config, head_type).to(device)
    # model2_filename = "HPL-Cluster/A_supertype_1102/main_finetune_plm_tape_B24_LR1e-05_seq_clip_fold4_ep17_230222.pkl"
    # model2_filename = "HPL-Cluster/B_supertype_4201/main_finetune_plm_tape_B24_LR1e-05_seq_clip_fold4_ep9_230221.pkl"
    model2_filename = "OneStep-Cluster/B_supertype_4201/main_finetune_plm_tape_B24_LR1e-05_seq_clip_fold4_ep37_230317.pkl" 
    model2.load_state_dict(torch.load(dir_saver + model2_filename), strict = True)
    model2.eval()
    ## sequence
    if plm_output_type == "mean":
        model3 = meanTAPE(tape_config, head_type).to(device)
    elif plm_output_type == "cls":
        model3 = clsTAPE(tape_config, head_type).to(device)
    # model3_filename = "HPL-Cluster/A_seq_more_1102/main_finetune_plm_tape_B26_LR1e-05_seq_clip_fold4_ep12_230304.pkl"
    # model3_filename = "HPL-Cluster/A_seq_equal_3303/main_finetune_plm_tape_B26_LR1e-05_seq_clip_fold4_ep9_230304.pkl"
    # model3_filename = "HPL-Cluster/A_seq_equal_3402/main_finetune_plm_tape_B26_LR1e-05_seq_clip_fold4_ep15_230304.pkl"
    # model3_filename = "HPL-Cluster/B_seq_more_4201/main_finetune_plm_tape_B26_LR6e-06_seq_clip_fold4_ep13_230223.pkl"
    model3_filename = "OneStep-Cluster/B_seq_more_4201/main_finetune_plm_tape_B26_LR6e-06_seq_clip_fold4_ep52_230317.pkl"
    model3.load_state_dict(torch.load(dir_saver + model3_filename), strict = True)
    model3.eval()
    ## semantic
    if plm_output_type == "mean":
        model4 = meanTAPE(tape_config, head_type).to(device)
    elif plm_output_type == "cls":
        model4 = clsTAPE(tape_config, head_type).to(device)
    # model4_filename = "HPL-Cluster/A_semantic_equal_1102/main_finetune_plm_tape_B26_LR1e-05_seq_clip_fold4_ep11_230309.pkl"
    # model4_filename = "HPL-Cluster/A_semantic_equal_3303/main_finetune_plm_tape_B26_LR1e-05_seq_clip_fold4_ep20_230313.pkl"
    # model4_filename = "HPL-Cluster/A_semantic_equal_3402/main_finetune_plm_tape_B26_LR1e-05_seq_clip_fold4_ep7_230313.pkl"
    # model4_filename = "HPL-Cluster/B_semantic_equal_4201/main_finetune_plm_tape_B26_LR1e-05_seq_clip_fold4_ep18_230309.pkl"
    model4_filename = "OneStep-Cluster/B_semantic_equal_4201/main_finetune_plm_tape_B26_LR1e-05_seq_clip_fold4_ep38_230316.pkl"
    model4.load_state_dict(torch.load(dir_saver + model4_filename), strict = True)
    model4.eval()

    model_list = [model1,
                  model2, 
                  model3,
                  model4]
    
    ## EG
    # print("Model Preparing")
    # device = torch.device("cuda" if args.use_cuda else "cpu")
    # tape_config = ProteinBertConfig.from_pretrained('bert-base')
    # ## sequence
    # if plm_output_type == "mean":
    #     model3 = meanTAPE(tape_config, head_type).to(device)
    # elif plm_output_type == "cls":
    #     model3 = clsTAPE(tape_config, head_type).to(device)
    # # model3_filename = "HPL-Cluster/G_seq_more_0101/main_finetune_plm_tape_B24_LR1e-05_seq_clip_fold4_ep13_230221.pkl"
    # # model3_filename = "HPL-Cluster/E_seq_more_0103/main_finetune_plm_tape_B26_LR1e-05_seq_clip_fold4_ep15_230309.pkl"
    # model3_filename = "OneStep-Cluster/G_seq_more_0101/main_finetune_plm_tape_B24_LR1e-05_seq_clip_fold4_ep44_230321.pkl"
    # model3.load_state_dict(torch.load(dir_saver + model3_filename), strict = True)
    # model3.eval()
    # ## semantic
    # if plm_output_type == "mean":
    #     model4 = meanTAPE(tape_config, head_type).to(device)
    # elif plm_output_type == "cls":
    #     model4 = clsTAPE(tape_config, head_type).to(device)
    # # model4_filename = "HPL-Cluster/G_semantic_equal_0101/main_finetune_plm_tape_B26_LR1e-05_seq_clip_fold4_ep8_230309.pkl"
    # # model4_filename = "HPL-Cluster/E_semantic_equal_0103/main_finetune_plm_tape_B26_LR1e-05_seq_clip_fold4_ep8_230309.pkl"
    # model4_filename = "OneStep-Cluster/G_semantic_equal_0101/main_finetune_plm_tape_B26_LR1e-05_seq_clip_fold4_ep46_230322.pkl"
    # model4.load_state_dict(torch.load(dir_saver + model4_filename), strict = True)
    # model4.eval()

    model_list = [model3,
                  model4]

    if args.eval_options == "all" or args.eval_options == "evaluate":
        (
            train_pos_df, test_pos_df, 
            valid_loader, test_loader, external_loader,
            test_loader_RN
        ) = prepare_main_task_loader(
            args.seq_type, args.fold, args.batch_size, data_path,
            dataset_configs, hla_seq_dict, HLA2ranking_candidates,
            num_workers=8,
            only_fullranking=False
        )
        # protocol 0-3: evaluate
        print("Ready to evaluate")
        evaluate(
            args,
            model1, plm_type, plm_input_type,
            valid_loader, test_loader, external_loader,
            test_loader_RN,
        )

    if args.eval_options == "all" or args.eval_options ==  "fullranking":
        (
            train_pos_df, test_pos_df, 
        ) = prepare_main_task_loader(
            args.seq_type, args.fold, args.batch_size, data_path,
            dataset_configs, hla_seq_dict, HLA2ranking_candidates,
            num_workers=8,
            only_fullranking=True,
            fullranking_type=args.fullranking_type
        )
        
        # protocol4: fullranking
        print("\n============================================================")
        print("Start Full-ranking on {} set".format(args.fullranking_type))
        print("============================================================\n")
        
        if args.eval_options == "all":
            args.batch_size = 2 * args.batch_size

        t1 = time.time()
        if args.fullranking_target == "single":
            log_name = "{}mer_RN_ft_{}_ensemble_{}{}{}.csv".format(
                args.hla_max_len, plm_type, 
                args.target_hla[4], args.target_hla[6:8], args.target_hla[9:]
                )
        else:
            log_name = "{}mer_RN_ft_{}_ensemble.csv".format(
                args.hla_max_len, plm_type
                )
        print(f"\n{log_name}\n")
        full_ranking_test(
            args,
            train_pos_df, test_pos_df,
            hla_seq_dict, HLA2ranking_candidates,
            model_list, plm_input_type,
            log_name,
            args.fullranking_type
        )
        t2 = time.time()
        print("\nFullranking totally uses {:.2f}s\n".format(t2-t1))

    # 6391M 13029M
    # 10755M
