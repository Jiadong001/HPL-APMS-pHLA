import os
import random

import numpy as np
import pandas as pd
import torch


class pHLA_Dataset(torch.utils.data.Dataset):
    def __init__(self, pep_hla_df, configs):
        """Arguments
        hla_seq_list: list, size: N
        pep_seq_list: list, size: N
        labels: list, size: N
        """
        super(pHLA_Dataset, self).__init__()

        self.pep_list = pep_hla_df.peptide.to_list()
        self.HLA_seq_list = pep_hla_df[configs["hla_seq_type"]].to_list()
        self.labels = torch.LongTensor(pep_hla_df.label.to_list())

        self.vocab = configs["vocab_file"]
        self.pep_max_len = configs["pep_max_len"]
        self.hla_max_len = configs["hla_max_len"]

    def __getitem__(self, index):
        hla_seq, pep, label = (
            self.HLA_seq_list[index],
            self.pep_list[index],
            self.labels[index],
        )

        hla_input = [self.vocab[n] for n in hla_seq.ljust(self.hla_max_len, "-")]
        pep_input = [self.vocab[n] for n in pep.ljust(self.pep_max_len, "-")]

        return torch.LongTensor(pep_input), torch.LongTensor(hla_input), label

    def __len__(self):
        return len(self.pep_list)


class pHLA_Dataset_RN(torch.utils.data.Dataset):
    """Some Information about pHLA_Dataset_RN"""

    def __init__(self, pep_hla_df, hla2candidate_dict, configs):
        super(pHLA_Dataset_RN, self).__init__()
        self.HLA_list = pep_hla_df.HLA.to_list()
        self.pep_list = pep_hla_df.peptide.to_list()
        self.HLA_seq_list = pep_hla_df[configs["hla_seq_type"]].to_list()
        self.vocab = configs["vocab_file"]
        self.pep_max_len = configs["pep_max_len"]
        self.hla_max_len = configs["hla_max_len"]
        self.hla2candidates = self.build_dict4sampling(hla2candidate_dict)
        print(len(self.HLA_list))

    def build_dict4sampling(self, hla2candidate_dict):
        hla2candidate_tuple = dict()
        for k, v in hla2candidate_dict.items():
            hla2candidate_tuple[k] = list(sorted(v))
        return hla2candidate_tuple

    def __getitem__(self, index):
        hla_name, hla_seq, pos_pep = (
            self.HLA_list[index],
            self.HLA_seq_list[index],
            self.pep_list[index],
        )
        neg_pep = random.choice(self.hla2candidates[hla_name])

        hla_input = [self.vocab[n] for n in hla_seq.ljust(self.hla_max_len, "-")]
        pos_pep_input = [self.vocab[n] for n in pos_pep.ljust(self.pep_max_len, "-")]
        neg_pep_input = [self.vocab[n] for n in neg_pep.ljust(self.pep_max_len, "-")]
        return (
            torch.LongTensor(hla_input),
            torch.LongTensor(pos_pep_input),
            torch.LongTensor(neg_pep_input),
        )

    def __len__(self):
        return len(self.HLA_list)


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
    
    vocab = np.load("../neoag_data/vocab_dict.npy", allow_pickle=True).item()

    configs = {
        "vocab_file": vocab,
        "hla_max_len": 182,
        "pep_max_len": 15,
        "hla_seq_type": "clip",
    }
    data_path = "../neoag_data/"
    train_df = pd.read_csv(data_path + "/train_data_fold4.csv", index_col=0)
    val_df = pd.read_csv(data_path + "/val_data_fold4.csv", index_col=0)
    seq_dict = (
        pd.read_csv(os.path.join(data_path, "hla_seq_dict.csv"), index_col=0)
        .set_index(["HLA_name"])["HLA_clip"]
        .to_dict()
    )
    for df in [train_df, val_df]:
        df["clip"] = df["HLA"].map(lambda x: seq_dict[x])
    dataset = pHLA_Dataset(train_df, configs)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0
    )
    i = 0
    for a, b, c in loader:
        print(a, c)
        i += 1
        if i > 1:
            break
    i=0
    for a, b, c in loader:
        print(a, c)
        i += 1
        if i > 1:
            break
