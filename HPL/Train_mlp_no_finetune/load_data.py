'''
class:
    pHLA_Dataset(pep_hla_df, configs)
    pHLA_Dataset_RN(pep_hla_df, hla2candidate_dict, configs)

function:
    extract_features_RN(
        hla_list, pep_seq_list, pep_seq_list_neg,
        inference_type, plm_type, 
        plm_tokenizer, plm_model, 
        hla2tensor_dict, device,
        rand_neg=True
    )

'''

from distutils.debug import DEBUG
import os
import random

import numpy as np
import pandas as pd
import torch
import time

from tape import ProteinBertModel, TAPETokenizer
from transformers import AutoTokenizer, AutoModel, pipeline

class pHLA_Dataset(torch.utils.data.Dataset):
    def __init__(self, pep_hla_df, configs):
        """Arguments
        hla_seq_list: list, size: N
        pep_seq_list: list, size: N
        labels: list, size: N
        """
        super(pHLA_Dataset, self).__init__()

        self.HLA_list = pep_hla_df.HLA.to_list()            # HLA name
        self.pep_list = pep_hla_df.peptide.to_list()        # df.pep=df['pep']
        self.labels = torch.LongTensor(pep_hla_df.label.to_list())

        # self.HLA_seq_list = pep_hla_df[configs["hla_seq_type"]].to_list()

        self.pep_max_len = configs["pep_max_len"]
        self.hla_max_len = configs["hla_max_len"]
        self.padding = configs["padding"]

    def __getitem__(self, index):
        hla, pep_seq, label = (
            self.HLA_list[index],
            self.pep_list[index],
            self.labels[index],
        )

        if self.padding == True:        
            # hla = hla.ljust(self.hla_max_len, 'X')
            pep_seq = pep_seq.ljust(self.pep_max_len, 'X')

        '''
        hla: str(name or seq)
        pep_seq: str
        label: longtensor
        '''
        return hla, pep_seq, label

    def __len__(self):
        return len(self.pep_list)


class pHLA_Dataset_RN(torch.utils.data.Dataset):
    """Some Information about pHLA_Dataset_RN"""

    def __init__(self, pep_hla_df, hla2candidate_dict, configs):
        super(pHLA_Dataset_RN, self).__init__()

        self.HLA_list = pep_hla_df.HLA.to_list()        # HLA name
        self.pep_list = pep_hla_df.peptide.to_list()
        
        # self.HLA_seq_list = pep_hla_df[configs["hla_seq_type"]].to_list()

        self.pep_max_len = configs["pep_max_len"]
        self.hla_max_len = configs["hla_max_len"]
        self.padding = configs["padding"]
        self.hla2candidates = self.build_dict4sampling(hla2candidate_dict)
        print(len(self.HLA_list))

    def build_dict4sampling(self, hla2candidate_dict):
        hla2candidate_tuple = dict()
        for k, v in hla2candidate_dict.items():
            hla2candidate_tuple[k] = list(sorted(v))
        return hla2candidate_tuple

    def __getitem__(self, index):
        hla_name, pos_pep_seq = (
            self.HLA_list[index],
            self.pep_list[index],
        )
        neg_pep_seq = random.choice(self.hla2candidates[hla_name])

        if self.padding == True:
            # hla_seq = hla_seq.ljust(self.hla_max_len, 'X')
            pos_pep_seq = pos_pep_seq.ljust(self.pep_max_len, 'X')
            neg_pep_seq = neg_pep_seq.ljust(self.pep_max_len, 'X')

        return hla_name, pos_pep_seq, neg_pep_seq

    def __len__(self):
        return len(self.HLA_list)


def extract_features_RN(
    hla_list, pep_seq_list_pos, pep_seq_list_neg,
    inference_type, plm_type, 
    plm_tokenizer, plm_model, 
    hla2tensor_dict, device,
    rand_neg=True
    ):
    '''
    Extract features of pep and HLA sequences using PLM.
    Tips: if "rand_neg=false", set pep_seq_list_neg=[]
    '''
    # hla
    ### method-1: use plm
    # token_ids_batch = torch.LongTensor([]).to(device)
    # for seq in hla_list:
    #     if plm_type == "protbert":
    #         seq = ' '.join(seq)
    #     token_ids = torch.LongTensor([plm_tokenizer.encode(seq)]).to(device)    # dimention is 2
    #     token_ids_batch = torch.cat((token_ids_batch, token_ids), dim=0)        # batch * (182+2)
    # with torch.no_grad():
    #     if inference_type == "full":
    #         hla_inputs = plm_model(token_ids_batch)[0]      # (batch, seq_len, 768/1024) full_output
    #     elif inference_type == "pooled":
    #         hla_inputs = plm_model(token_ids_batch)[1]      # (batch, 768/1024) pooled_output

    ### method-2: use hla2tensor_dict
    hla_inputs = []
    for hla in hla_list:
        plm_output = hla2tensor_dict[hla]
        hla_inputs.append(plm_output)
    hla_inputs = torch.cat(hla_inputs, dim=0).to(device)        # faster
    
    if rand_neg == True:
        if inference_type == "full":
            hla_inputs = hla_inputs.repeat(2,1,1)               # 2batch: 1batch for pos, 1batch for neg
        elif inference_type == "pooled" or "mean":
            hla_inputs = hla_inputs.repeat(2,1)

    # pep
    token_ids_batch = []
    for seq in pep_seq_list_pos:
        if plm_type == "protbert":
            seq = ' '.join(seq)
        token_ids_batch.append(plm_tokenizer.encode(seq))       # dimention is 2, faster than torch.cat !!!
        
    if rand_neg == True:
        for seq in pep_seq_list_neg:
            if plm_type == "protbert":
                seq = ' '.join(seq)
            token_ids_batch.append(plm_tokenizer.encode(seq))

    token_ids_batch = torch.LongTensor(token_ids_batch).to(device)  # batch * (15+2)

    with torch.no_grad():
        if inference_type == "full":
            pep_inputs = plm_model(token_ids_batch)[0]      # (1/2batch, seq_len, 768/1024) full_output
        elif inference_type == "pooled":
            pep_inputs = plm_model(token_ids_batch)[1]      # (1/2batch, 768/1024) pooled_output
        elif inference_type == "mean":
            pep_inputs = plm_model(token_ids_batch)[0]
            pep_inputs = torch.mean(pep_inputs, dim=1)      # (1/2batch, 768/1024) average_output
    
    '''Output size
    full: [1/2batch, seq_len, 768/1024]
    pooled/mean: [1/2batch, 768/1024]
    '''
    return hla_inputs, pep_inputs


if __name__ == "__main__":
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
    
    # vocab = np.load("/data/zhuxy/neoag_data/vocab_dict.npy", allow_pickle=True).item()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = {
        "hla_max_len": 182,
        "pep_max_len": 15,
        "hla_seq_type": "clip",
        "padding": False,
    }

    # data_path = "/data/zhuxy/neoag_data/"
    data_path = "/data/lujd/neoag_data/"
    data_name = "main_task/test_set.csv"

    #-----------run on big set----------#
    # train_df = pd.read_csv(data_path + "main_task/train_data_fold4.csv", index_col=0)
    # val_df = pd.read_csv(data_path + "main_task/val_data_fold4.csv", index_col=0)
    # seq_dict = (
    #     pd.read_csv(os.path.join(data_path, "main_task/hla_seq_dict.csv"), index_col=0)
    #     .set_index(["HLA_name"])["HLA_clip"]
    #     .to_dict()
    # )
    # for df in [train_df, val_df]:
    #     df["clip"] = df["HLA"].map(lambda x: seq_dict[x])
    
    plm_type = "protbert"
    if plm_type == "tape":
        plm_tokenizer = TAPETokenizer(vocab='iupac')
        plm_model = ProteinBertModel.from_pretrained('bert-base').to(device)
        print("PLM-TAPE is ready")
    elif plm_type == "protbert":
        plm_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
        plm_model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd").to(device)
        print("PLM-ProtBERT is ready")

    df = pd.read_csv(data_path + data_name, index_col=0)[:10]
    df = df.set_index(["HLA"])
    hla_clip_dict = df["peptide"].to_dict()    # {hla: clip_seq}
    # initial_dict = np.load(
    #     data_path+"main_task/{}_hla2tensor_full_dict.npy".format(plm_type),
    #     allow_pickle=True
    #     ).item()

    plm_model.eval()
    token_ids_batch = torch.LongTensor([]).to(device)
    for clip_seq in hla_clip_dict.values():
        
        clip_seq = clip_seq.ljust(15, 'X')
        if plm_type == "protbert":
            clip_seq = ' '.join(clip_seq)

        token_ids = torch.LongTensor([plm_tokenizer.encode(clip_seq)]).to(device)
        token_ids_batch = torch.cat((token_ids_batch, token_ids), dim=0)
    with torch.no_grad():
        output = plm_model(token_ids_batch)[1]
    output = output.cpu()

    pooled_dict = hla_clip_dict.copy()
    plm_model.eval()
    for hla, clip_seq in zip(pooled_dict.keys(), pooled_dict.values()):
        
        clip_seq = clip_seq.ljust(15, 'X')
        if plm_type == "protbert":
            clip_seq = ' '.join(clip_seq)

        token_ids = torch.LongTensor([plm_tokenizer.encode(clip_seq)]).to(device)
        with torch.no_grad():
            one_output = plm_model(token_ids)[1]

        pooled_dict[hla] = one_output.cpu()

    i=0
    for hla in pooled_dict.keys():
        # print(f"init_dict:\n {initial_dict[hla][:,300:307]}")
        print(hla_clip_dict[hla])
        print(f"batch_out:\n {output[i,300:307]}")
        print(f"onbon_out:\n {pooled_dict[hla][:,300:307]}\n")
        i=i+1
        if i>5:
            break
    
    print(pooled_dict[hla].shape, output.shape)

    #-----------run on test set----------#
    # train_df = pd.read_csv(data_path + "main_task/test_set.csv", index_col=0)
    # seq_dict = (
    #         pd.read_csv(os.path.join(data_path, "main_task/hla_seq_dict.csv"), index_col=0)
    #         .set_index(["HLA_name"])["HLA_clip"]
    #         .to_dict()
    #     )
    # train_df["clip"] = train_df["HLA"].map(lambda x: seq_dict[x])           # 35 -> 182


    # ''' test pHLA_Dataset '''
    # t1 = time.time()
    # print("Data Preparing")
    # dataset = pHLA_Dataset(train_df, configs)
    # loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=16, shuffle=True, num_workers=0                  # process on gpu, num_workers=0
    # )
    # print("Data Preparing-done")
    
    # i = 0
    # for a, b in loader:
    #     print(a, b)
    #     i += 1
    #     if i > 1:
    #         break

    # t2 = time.time()
    # print("run time: ",t2-t1)

    # i = 0
    # for a, b in loader:
    #     print(a, b)
    #     i += 1
    #     if i > 1:
    #         break


    # ''' test pHLA_Dataset_RN '''
    # t1 = time.time()
    # print("Data Preparing")

    # HLA2ranking_candidates = np.load(
    #     data_path + "main_task/main_task_HLA2ranking_candidate_pools.npy",
    #     allow_pickle=True,
    # ).item()
    # print("pool is ready")

    # dataset = pHLA_Dataset_RN(train_df, HLA2ranking_candidates, configs)
    # loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=256, shuffle=True, num_workers=0
    # )
    # print("Data Preparing-done")

    # t2 = time.time()
    # print("run time: ",t2-t1)

    # i = 0
    # for a, b, c in loader:
    #     print(a, b)
    #     i += 1
    #     if i > 1:
    #         break

    # t3 = time.time()
    # print("one batch run time: ",t3-t2)

    # i = 0
    # for a, b, c in loader:
    #     print(a, b)
    #     i += 1
    #     if i > 1:
    #         break

    
    # data_path = "/data/zhuxy/neoag_data/"
    # covid_data = pd.read_csv(
    #     data_path + "covid/class_I_COVID_peptide-HLA_pairs.csv", index_col=0
    # ).rename(columns={"SARS-CoV-2 epitope": "peptide"})[["peptide", "HLA"]]
    # # print(covid_data)
    # HLA2ranking_candidates = np.load(
    #     data_path + "covid/HLA2ranking_candidate_pep_segs.npy", allow_pickle=True
    # ).item()
    # HLA2clip_seq = (
    #     pd.read_csv(os.path.join(data_path, "hla_seq_dict.csv"), index_col=0)
    #     .set_index(["HLA_name"])["HLA_clip"]
    #     .to_dict()
    # )

    # covid_data["clip"] = covid_data["HLA"].map(lambda x: HLA2clip_seq[x])

    # train_df = pd.DataFrame()
    # val_df = pd.DataFrame()
    # test_df = pd.DataFrame()
    # for HLA in covid_data.HLA.unique():
    #     data_hla = covid_data[covid_data.HLA == HLA]
    #     train_data = data_hla.sample(frac=0.6, random_state=111)
    #     rem_data = data_hla[~data_hla.index.isin(train_data.index)]
    #     val_data = rem_data.sample(frac=0.5, random_state=111)
    #     test_data = rem_data[~rem_data.index.isin(val_data.index)]
    #     train_df = train_df.append(train_data)
    #     val_df = val_df.append(val_data)
    #     test_df = test_df.append(test_data)

    # covid_data["clip"] = covid_data["HLA"].map(lambda x: HLA2clip_seq[x])

    # val_dataset = pHLA_Dataset_RN(val_df, HLA2ranking_candidates, configs)

    # loader = torch.utils.data.DataLoader(val_dataset, 4, shuffle=False, num_workers=0)

    # for (
    #     a,
    #     b,
    #     c,
    # ) in loader:
    #     print(b)
    #     print(c)
    #     break

    # for (
    #     a,
    #     b,
    #     c,
    # ) in loader:
    #     print(b)
    #     print(c)
    #     exit()
    # # iteror = iter(loader)

    # # a, b, c= next(iteror)
    # # # print(a)
    # # print(b)
    # # print(c)

    # # a, b, c= next(iteror)
    # # # print(a)
    # # print(b)
    # # print(c)

    # # a, b, c= next(iter(loader))
    # # # print(a)
    # # print(b)
    # # print(c)
    # # data_path = "/data/zhuxy/neoag/Dataset"
    # # train_set = pd.read_csv(os.path.join(
    # #     data_path, 'train_data_fold4.csv'), index_col=0)
    # # hla_name_list = list(train_set['HLA'])
    # # hla_seq_list = list(train_set['HLA_sequence'])
    # # pep_seq_list = list(train_set['peptide'])
    # # labels = list(train_set['label'])
    # # print(len(pep_seq_list), len(labels))
    # # all_peps_set = set(pep_seq_list)
    # # for label in labels:
    # #     if label not in [0,1]:
    # #         print("ERROR")
    # #         print(label)

    # # train_dataset = pHLA_Dataset_random_neg(
    # #     hla_name_list, hla_seq_list, pep_seq_list, all_peps_set, configs)

    # # a, b= next(iter(loader))
    # # print(a.size())
    # # print(b.size())
