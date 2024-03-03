'''
class:
    pHLA_Dataset(pep_hla_df, configs)
    pHLA_Dataset_RN(pep_hla_df, hla2candidate_dict, configs)

function:
    seq2token(
        hla_list, 
        pep_seq_list,
        plm_type,
        plm_input_type,
        device
    )

'''

from distutils.debug import DEBUG
import os
import random
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data

from tape import TAPETokenizer, ProteinBertConfig
from transformers import AutoTokenizer, AutoModel, pipeline

from model_ft import meanTAPE

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

        self.HLA_seq_list = pep_hla_df[configs["hla_seq_type"]].to_list()

        self.pep_max_len = configs["pep_max_len"]
        self.hla_max_len = configs["hla_max_len"]
        self.padding = configs["padding"]

    def __getitem__(self, index):
        hla_seq, pep_seq, label = (
            self.HLA_seq_list[index],
            self.pep_list[index],
            self.labels[index],
        )

        if self.padding == True:        
            hla_seq = hla_seq.ljust(self.hla_max_len, 'X')
            pep_seq = pep_seq.ljust(self.pep_max_len, 'X')

        '''
        hla: str(name or seq)
        pep_seq: str
        label: longtensor
        '''
        return hla_seq, pep_seq, label

    def __len__(self):
        return len(self.pep_list)

class pHLA_Dataset_RN(torch.utils.data.Dataset):
    """Some Information about pHLA_Dataset_RN"""

    def __init__(self, pep_hla_df, hla2candidate_dict, configs):
        super(pHLA_Dataset_RN, self).__init__()

        self.HLA_list = pep_hla_df.HLA.to_list()        # HLA name
        self.pep_list = pep_hla_df.peptide.to_list()
        
        self.HLA_seq_list = pep_hla_df[configs["hla_seq_type"]].to_list()

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
        hla_name, hla_seq, pos_pep_seq = (
            self.HLA_list[index],
            self.HLA_seq_list[index],
            self.pep_list[index],
        )
        neg_pep_seq = random.choice(self.hla2candidates[hla_name])

        if self.padding == True:
            hla_seq = hla_seq.ljust(self.hla_max_len, 'X')
            pos_pep_seq = pos_pep_seq.ljust(self.pep_max_len, 'X')
            neg_pep_seq = neg_pep_seq.ljust(self.pep_max_len, 'X')

        return hla_seq, pos_pep_seq, neg_pep_seq

    def __len__(self):
        return len(self.HLA_list)

def seq2token(
    hla_list, 
    pep_seq_list,
    plm_type,
    plm_input_type,
    device
    ):
    
    hla_pep_inputs = []         # the input of model is token

    if plm_type == "tape":
        tokenizer = TAPETokenizer(vocab='iupac')

        for hla, pep in zip(hla_list, pep_seq_list):
            phla = hla + pep
            token = tokenizer.encode(phla)                  # array
            if plm_input_type == "sep":
                token = np.insert(token, (len(hla)+1), 3)   # insert 3(<sep>) in position len(hla)+1
            hla_pep_inputs.append(token)

    return torch.LongTensor(hla_pep_inputs).to(device)

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

    data_path = "../neoag_data/"
    main_train_df = pd.read_csv(
            os.path.join(data_path, "main_task/train_data_fold{}.csv".format(4)), index_col=0
        )[:1000]
    seq_dict = (
            pd.read_csv(os.path.join(data_path, "main_task/hla_seq_dict.csv"), index_col=0)
            .set_index(["HLA_name"])["HLA_clip"]
            .to_dict()
        )
    HLA2ranking_candidates = np.load(
        data_path + "main_task/allele2candidate_pools.npy",
        allow_pickle=True,
    ).item()
    print("pool is ready")

    main_train_df["clip"] = main_train_df["HLA"].map(lambda x: seq_dict[x])

    dataset_configs = {
            "hla_max_len": 182,
            "pep_max_len": 15,
            "hla_seq_type": "clip",
            "padding": True,
        }
    train_dataset = pHLA_Dataset_RN(main_train_df, HLA2ranking_candidates, dataset_configs)

    train_loader = Data.DataLoader(
                train_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=0,
            )

    device = torch.device("cuda")

    pbar = tqdm(train_loader)
    for hla_list, pep_seq_list_pos, pep_seq_list_neg in pbar:
        batch_num = len(hla_list)       # hla_list is a tuple
        
        t1=time.time()
        phla_tokens = seq2token(
                        hla_list, 
                        pep_seq_list_pos,
                        "tape",
                        "cat",
                        device
                    )
        t2=time.time()
        print(batch_num, phla_tokens.shape)
        print(phla_tokens[1])
        print(t2-t1)
        break
    
    config = ProteinBertConfig.from_pretrained('bert-base')
    model = meanTAPE(config).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(phla_tokens)

    print(outputs.shape)

    # without no_grad: 17000M
    # with no_grad: 2700M