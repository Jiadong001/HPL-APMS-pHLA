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
from load_data import pHLA_Dataset, pHLA_Dataset_RN
from model_arc import Transformer
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
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

# GLOBAL_WORKER_ID = None
# def worker_init_fn(worker_id):
#   global GLOBAL_WORKER_ID
#   GLOBAL_WORKER_ID = worker_id
#   np.random.seed(seed + worker_id)


def eval_data_preparation(
    seq_type, fold, batch_size, 
    configs, seq_dict, hla2candidates,
    ):

    main_train_df = pd.read_csv(os.path.join(
        data_path, "train_data_fold{}.csv".format(fold)),
        index_col=0)
    main_valid_df = pd.read_csv(os.path.join(
        data_path, "val_data_fold{}.csv".format(fold)),
        index_col=0)
    main_test_df = pd.read_csv(os.path.join(
        data_path, "independent_set.csv"),
        index_col=0)
    main_external_df = pd.read_csv(os.path.join(
        data_path, "external_set.csv"),
        index_col=0)

    for df in [main_train_df, main_valid_df, main_test_df, main_external_df]:
        df.rename(columns={"HLA_sequence": "short"}, inplace=True)
        
    main_valid_df[seq_type] = main_valid_df["HLA"].map(lambda x: seq_dict[x])
    main_test_df[seq_type] = main_test_df["HLA"].map(lambda x: seq_dict[x])
    main_external_df[seq_type] = main_external_df["HLA"].map(lambda x: seq_dict[x])

    train_pos_df = main_train_df[main_train_df['label'] == 1]
    test_pos_df = main_test_df[main_test_df.label == 1]

    test_dataset_RN = pHLA_Dataset_RN(test_pos_df, hla2candidates, configs)

    test_loader_RN = torch.utils.data.DataLoader(
        test_dataset_RN,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=seed_worker,
    )

    test_dataset = pHLA_Dataset(main_test_df, configs)
    valid_dataset = pHLA_Dataset(main_valid_df, configs)
    external_dataset = pHLA_Dataset(main_external_df, configs)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=seed_worker,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=seed_worker,
    )

    external_loader = torch.utils.data.DataLoader(
        external_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=seed_worker,
    )
    return train_pos_df, test_pos_df, valid_loader, test_loader, test_loader_RN, external_loader

def zeroshot_data_preparation(fullranking_type):

    if fullranking_type == "zeroshot_eg":
        zeroshot_df = pd.read_csv(os.path.join(
            data_path, "zeroshot_set.csv"),
            index_col=0)
    elif fullranking_type == "zeroshot_ab":
        zeroshot_df = pd.read_csv(os.path.join(
            data_path, "zeroshot_ab_set.csv"),
            index_col=0)
    elif fullranking_type == "zeroshot_abc":
        zeroshot_df = pd.read_csv(os.path.join(
            data_path, "zeroshot_abc_set.csv"),
            index_col=0)
    zeroshot_pos_df = zeroshot_df[zeroshot_df['label'] == 1]

    return zeroshot_pos_df


def full_ranking(
    model,
    test_pos_peptides,
    HLA_seq,
    candidate_neg_pep_pool,
    use_cuda,
    vocab,
    hla_max_len,
    pep_max_len,
    lengths,
    topk,
    inf_batch,
    bottom_k=5,
):

    device = torch.device("cuda" if use_cuda else "cpu")
    ranking_pool = sorted(
        list(candidate_neg_pep_pool.union(test_pos_peptides)))

    # Encoding
    hla_encoding = [vocab[n] for n in HLA_seq.ljust(hla_max_len, "-")
                    ]  # HLA encodings remain invariant
    candidate_pep_encodings = []
    for pep in ranking_pool:
        candidate_pep_encodings.append(
            [vocab[n] for n in pep.ljust(pep_max_len, "-")])
    print("Ranking {} samples, Targeting {} positive samples".format(
        len(ranking_pool), len(test_pos_peptides)))

    start_index = 0
    end_index = inf_batch

    model.eval()
    y_eval_prob_list = []
    y_eval_prob_softmax_list = []
    output_list = []
    with torch.no_grad():
        while end_index <= len(ranking_pool) and start_index < end_index:
            val_HLA_inputs = torch.LongTensor([hla_encoding] *
                                              (end_index - start_index))
            val_pep_inputs = torch.LongTensor(
                candidate_pep_encodings[start_index:end_index])

            val_HLA_inputs, val_pep_inputs = val_HLA_inputs.to(
                device), val_pep_inputs.to(device)
            val_outputs, _, _, val_dec_self_attns = model(
                val_pep_inputs, val_HLA_inputs)

            y_prob_val = val_outputs[:,1] - val_outputs[:,0]
            # y_prob_val_softmax = nn.Softmax(dim=1)(val_outputs)[:, 1]
            
            y_eval_prob_list.append(y_prob_val)
            # y_eval_prob_softmax_list.append(y_prob_val_softmax)
            # output_list.append(val_outputs)
            
            start_index = end_index
            if end_index + inf_batch < len(ranking_pool):
                end_index += inf_batch
            else:
                end_index = len(ranking_pool)
                
        y_prob_all = torch.cat(y_eval_prob_list, dim=0).cpu().detach()
        _, index_of_rank_list = torch.topk(y_prob_all, len(ranking_pool))

        ordered_pred_peps = [ranking_pool[_] for _ in index_of_rank_list.numpy()]
        
        len2recall_dict, len2hit_dict = {}, {}
        for i in lengths:
            select_pos_pep = []
            for pep in test_pos_peptides:
                if len(pep) == i:
                    select_pos_pep.append(pep)
            select_pos_pep = set(select_pos_pep)

            positive_ranks = sorted([
                ordered_pred_peps.index(pos_pep) for pos_pep in select_pos_pep
            ])
            bottom_positive_ranks = positive_ranks[-bottom_k:]

            recall = np.array([0.0] * len(topk))
            hit = np.array([0] * len(topk))
            num_pos = len(select_pos_pep)
            for ind, k in enumerate(topk):
                num_hit = len(select_pos_pep.intersection(set(ordered_pred_peps[:k])))
                if num_pos != 0:
                    recall[ind] += num_hit / num_pos
                hit[ind] += num_hit

            print("Recall@K")
            print(recall)
            print("Hit@K")
            print(hit, "|", num_pos)
            len2recall_dict[i] = recall
            len2hit_dict[i] = hit

        # print("Positive Bottom Rank")
        # print(f_mean(bottom_positive_ranks))
        # print("All Posiive Ranks")
        # print(positive_ranks,"\n")

        return len2recall_dict, len2hit_dict


def full_ranking_test(
    model,
    hla_seq_dict,
    train_pos_df,
    test_pos_df,
    vocab,
    HLA2ranking_candidates,
    args,
    log_name,
    fullranking_type="test",
    topk=[50, 100, 500, 1000, 5000, 10000, 100000]
):
    HLA_list = list(test_pos_df.HLA.unique())
    print("HLA allele num: {}".format(len(HLA_list)))

    lengths = test_pos_df["length"].unique()
    len2recall_dict, len2hit_dict, len2num_dict = {}, {}, {}
    for i in lengths:
        len2recall_dict[i] = []
        len2hit_dict[i] = []
        len2num_dict[i] = []
    print(lengths)

    for current_HLA in HLA_list[:]:
        ranking_candidates = HLA2ranking_candidates[current_HLA]
        assert isinstance(ranking_candidates, set)

        current_HLA_df = test_pos_df[test_pos_df["HLA"] == current_HLA]
        pos_test_pep = set(list(current_HLA_df.peptide))

        HLA_seq = hla_seq_dict[current_HLA]
        print(current_HLA, HLA_seq)
        print("full ranking on {}".format(current_HLA))

        recall_cur_dict, hit_cur_dict = full_ranking(
            model,
            pos_test_pep,
            HLA_seq,
            ranking_candidates,
            True,
            vocab,
            args.hla_max_len,
            args.pep_max_len,
            lengths,
            topk,
            1024,
        )
        # full_ranking(model, pos_test_pep, HLA_seq, candidate_peps, True, vocab, args.hla_max_len, args.pep_max_len, [1, 3, 5])

        for i in lengths:
            len2recall_dict[i].append(recall_cur_dict[i])
            len2hit_dict[i].append(hit_cur_dict[i])
            len2num_dict[i].append(len(current_HLA_df[current_HLA_df["length"]==i]))

    print("===================================================")
    
    for i in lengths:
        col_names = ['recall@' + str(_) for _ in topk]
        recall_results_df = pd.DataFrame(len2recall_dict[i], columns=col_names)
        recall_results_df['HLA'] = HLA_list

        col_names = ['hit@' + str(_) for _ in topk]
        hit_results_df = pd.DataFrame(len2hit_dict[i], columns=col_names)
        hit_results_df['pos_num'] = len2num_dict[i]
        
        results_df = pd.concat((recall_results_df, hit_results_df), axis=1)
        results_df.to_csv(f"../fullranking_csvs/{fullranking_type}/"+log_name+f"_len{i}.csv")


def make_evaluation(rand_neg, model, loader, type, threshold, fold, use_cuda,
                    desc):

    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()
    criterion = nn.CrossEntropyLoss()
    if rand_neg:
        with torch.no_grad():
            loss_val_list, dec_attns_val_list = [], []
            y_true_val_list, y_prob_val_list = [], []
            # print(len(loader))
            pbar = tqdm(loader)
            pbar.set_description(desc)
            for val_hla_inputs, val_pos_pep_inputs, val_neg_pep_inputs in pbar:
                batch_num = val_hla_inputs.shape[0]
                val_hla_inputs, val_pos_pep_inputs, val_neg_pep_inputs = (
                    val_hla_inputs.to(device),
                    val_pos_pep_inputs.to(device),
                    val_neg_pep_inputs.to(device),
                )

                val_outputs, _, _, val_dec_self_attns_neg = model(
                    torch.cat((val_pos_pep_inputs, val_neg_pep_inputs)),
                    torch.cat((val_hla_inputs, val_hla_inputs)),
                )

                y_true_val = torch.LongTensor([1] * batch_num +
                                              [0] * batch_num).to(device)

                val_loss = criterion(val_outputs, y_true_val)
                y_prob_val = nn.Softmax(
                    dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
                y_true_val = y_true_val.cpu().numpy()
                y_true_val_list.extend(y_true_val)
                y_prob_val_list.extend(y_prob_val)
                loss_val_list.append(val_loss)
            #             dec_attns_val_list.append(val_dec_self_attns)

            y_pred_val_list = transfer(y_prob_val_list, threshold)
            ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)
            print("Fold-{} ******{}****** : Loss = {:.6f}".format(
                fold, desc, f_mean(loss_val_list)))
            metrics_val = performances(y_true_val_list,
                                       y_pred_val_list,
                                       y_prob_val_list,
                                       print_=True)
    else:
        with torch.no_grad():
            loss_val_list, dec_attns_val_list = [], []
            y_true_val_list, y_prob_val_list = [], []
            # print(len(loader))
            pbar = tqdm(loader)
            pbar.set_description(desc)
            for train_pep_inputs, train_hla_inputs, train_labels in pbar:
                """
                pep_inputs: [batch_size, pep_len]
                hla_inputs: [batch_size, hla_len]
                train_outputs: [batch_size, 2]
                """
                train_pep_inputs, train_hla_inputs, train_labels = (
                    train_pep_inputs.to(device),
                    train_hla_inputs.to(device),
                    train_labels.to(device),
                )

                t1 = time.time()
                train_outputs, _, _, train_dec_self_attns = model(
                    train_pep_inputs, train_hla_inputs)
                val_loss = criterion(train_outputs, train_labels)

                # train_outputs, _, _, train_dec_self_attns = model(train_pep_inputs,
                #                                                   train_hla_inputs)
                # train_loss = criterion(train_outputs, train_labels)

                y_true_val = train_labels.cpu().numpy()
                y_prob_val = (nn.Softmax(
                    dim=1)(train_outputs)[:, 1].cpu().detach().numpy())

                y_true_val_list.extend(y_true_val)
                y_prob_val_list.extend(y_prob_val)
                loss_val_list.append(val_loss)
            #             dec_attns_val_list.append(val_dec_self_attns)

            y_pred_val_list = transfer(y_prob_val_list, threshold)
            ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)

            print("Fold-{} ******{}****** : Loss = {:.6f}".format(
                fold, desc, f_mean(loss_val_list)))
            metrics_val = performances(y_true_val_list,
                                       y_pred_val_list,
                                       y_prob_val_list,
                                       print_=True)

    return ys_val, loss_val_list, metrics_val  # , dec_attns_val_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate the trained_model")

    # basic configuration
    parser.add_argument("--use_cuda", action="store_false", default=True)
    parser.add_argument("--data_path",
                        type=str,
                        default="/data/zhuxy/neoag_data/")
    parser.add_argument("--model_path",
                        type=str,
                        default="/data/zhuxy/neoag_model/")
    parser.add_argument(
        "--model_name",
        type=str,
        default=
        "main/model_layer1_multihead9_fold4.pkl"
    )
    parser.add_argument("--pep_max_len", type=int, default=15)

    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument(
        "--recall_logname",
        type=str,
        default="default_logname"
    )

    # -----------------Parameters for data------------------------
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--seq_type",
                        type=str,
                        choices=["short", "full", "clip"],
                        default="short")

    # -----------------Parameters for model arc-------------------
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=9)

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--d_k", type=int, default=64)
    parser.add_argument("--d_v", type=int, default=64)

    # -----------------Parameters for training---------------------
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--l_r", type=float, default=1e-5)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)

    args = parser.parse_args()
    print(args)
    # ================================================================

    data_path = args.data_path
    dir_saver = args.model_path

    if args.seq_type == "short":
        args.hla_max_len = 34
    elif args.seq_type == "full":
        args.hla_max_len = 372
    elif args.seq_type == "clip":
        args.hla_max_len = 182

    tgt_len = args.pep_max_len + args.hla_max_len
    args.tgt_len = tgt_len
    device = torch.device("cuda" if args.use_cuda else "cpu")

    vocab = np.load(os.path.join(
        data_path, "vocab_dict.npy"),
        allow_pickle=True
        ).item()
    vocab_size = len(vocab)
    args.vocab_size = vocab_size

    dataset_configs = {
        "vocab_file": vocab,
        "hla_max_len": args.hla_max_len,
        "pep_max_len": args.pep_max_len,
        "hla_seq_type":  args.seq_type,
    }
    print("Build model")
    model = Transformer(args).to(device)

    print("Load pre-trained model")
    path_saver = args.model_path + args.model_name
    print(path_saver)
    model.load_state_dict(torch.load(path_saver))
    model.eval()

    ## evaluation on test and external set
    # Prepare data
    # print("Data Preparing")
    # batch_size = args.batch_size
    # hla_seq_dict = pd.read_csv(os.path.join(
    #     data_path, "HLA_sequence_dict_ABCEG.csv"),
    #     index_col=0
    #     ).set_index(["HLA_name"])[args.seq_type].to_dict()
    
    # HLA2ranking_candidates = np.load(
    #     data_path + "allele2candidate_pools.npy",
    #     allow_pickle=True,
    # ).item()
    # (   train_pos_df, test_pos_df, 
    #     valid_loader, test_loader, test_loader_RN, external_loader
    # ) = eval_data_preparation(
    #         args.seq_type, args.fold, batch_size, 
    #         dataset_configs, hla_seq_dict, HLA2ranking_candidates)
            
    # print("Ready to evaluate")
    # ys_val, loss_val_list, metrics_val = make_evaluation(False,
    #                                                      model,
    #                                                      valid_loader,
    #                                                      "valid",
    #                                                      args.threshold,
    #                                                      args.fold,
    #                                                      args.use_cuda,
    #                                                      desc="Validation")

    # ys_val, loss_val_list, metrics_val = make_evaluation(False,
    #                                                      model,
    #                                                      test_loader,
    #                                                      "test",
    #                                                      args.threshold,
    #                                                      args.fold,
    #                                                      args.use_cuda,
    #                                                      desc="Test")

    # ys_val, loss_val_list, metrics_val = make_evaluation(False,
    #                                                      model,
    #                                                      external_loader,
    #                                                      "test",
    #                                                      args.threshold,
    #                                                      args.fold,
    #                                                      args.use_cuda,
    #                                                      desc="External")

    # test_times = 5

    # avg_mertric = np.array([0.0] * 9)
    # for i in range(test_times):
    #     ys_val, loss_val_list, metrics_val = make_evaluation(True,
    #                                                          model,
    #                                                          test_loader_RN,
    #                                                          "Test-RN",
    #                                                          args.threshold,
    #                                                          args.fold,
    #                                                          args.use_cuda,
    #                                                          desc="Test-RN")
    #     avg_mertric += metrics_val

    # # print(avg_mertric/10)
    # (roc_auc, accuracy, mcc, f1, sensitivity, specificity, precision, recall,
    #  aupr) = avg_mertric / test_times
    # print(
    #     "*************************** {} times random negative test average performance ***************************"
    #     .format(test_times))
    # print(
    #     "auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}"
    #     .format(roc_auc, sensitivity, specificity, accuracy, mcc))
    # print("precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}".format(
    #     precision, recall, f1, aupr))

    # train_pos_df = train_pos_df[train_pos_df["HLA"].isin( ["HLA-A*02:01"] )]
    # test_pos_df = test_pos_df[test_pos_df["HLA"].isin( ["HLA-A*02:01"] )]
    # full_ranking_test(
    #     model, hla_seq_dict, train_pos_df, test_pos_df, vocab,
    #     HLA2ranking_candidates, args, args.recall_logname+".csv"
    #     )


    ## zeroshot
    print("Data Preparing")
    hla_seq_dict = pd.read_csv(os.path.join(
        data_path, "HLA_sequence_dict_ABCEG.csv"), index_col=0
    ).set_index(["HLA_name"])[args.seq_type].to_dict()

    fullranking_type = "zeroshot_eg"
    HLA2ranking_candidates = np.load(
                data_path + "zeroshot_allele2candidate_pools.npy",
                allow_pickle=True,
            ).item()
    zeroshot_pos_df = zeroshot_data_preparation(fullranking_type)
    # zeroshot_pos_df = zeroshot_pos_df[zeroshot_pos_df["HLA"].isin( ["HLA-G*01:01"] )]
    print("Ready for eg's fullranking")
    full_ranking_test(
        model, hla_seq_dict, [], zeroshot_pos_df, vocab,
        HLA2ranking_candidates, args, args.recall_logname,
        fullranking_type=fullranking_type
        )
    
    fullranking_type = "zeroshot_abc"
    HLA2ranking_candidates = np.load(
            data_path + "zs_new_abc_allele2candidate_pools.npy",
            allow_pickle=True,
        ).item()
    zeroshot_pos_df = zeroshot_data_preparation(fullranking_type)
    # zeroshot_pos_df = zeroshot_pos_df[zeroshot_pos_df["HLA"].isin( ["HLA-B*42:01"] )]
    print("Ready for ab's fullranking")
    full_ranking_test(
        model, hla_seq_dict, [], zeroshot_pos_df, vocab,
        HLA2ranking_candidates, args, args.recall_logname,
        fullranking_type=fullranking_type
        )
