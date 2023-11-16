# Passion4ever

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .utils import read_fasta, seq2vec


class ElasticDataSet(Dataset):
    def __init__(self, *arrays):
        length = len(arrays)
        if length == 0:
            raise ValueError("At least one array required as input")
        self.n_samples = len(arrays[0])
        self.arrays = arrays

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return [array[idx] for array in self.arrays]


def collate(batch):
    seq1_ls, seq2_ls, label1_ls, label2_ls, label_ls = [], [], [], [], []

    for i in range(len(batch)//2):
        seq1, label1 = batch[i][0], batch[i][1]
        seq2, label2 = batch[i+len(batch)//2][0], batch[i+len(batch)//2][1]
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        label_ls.append((label1^label2).unsqueeze(0))
        seq1_ls.append(seq1.unsqueeze(0))
        seq2_ls.append(seq2.unsqueeze(0))

    return [torch.cat(seq1_ls), torch.cat(seq2_ls), torch.cat(label_ls), 
            torch.cat(label1_ls), torch.cat(label2_ls)]


def get_dataloaders(data_dir, k, max_len, seq_type, val_size, 
                   batch_size, **kwargs):
    # data
    seq_dict = read_fasta(data_dir)
    seq_arr, labels, _ = seq2vec(seq_dict, k=k, seq_type=seq_type, max_len=max_len)
    # dataset
    dataset = ElasticDataSet(seq_arr, labels)
    n_samples = len(dataset)
    assert val_size is not None and val_size > 0 and val_size < 1, \
        'You must set <val_size> between 0 and 1'
    train_num = int(n_samples * (1 - val_size))
    val_num = n_samples - train_num
    train_dataset, val_dataset = random_split(dataset, [train_num, val_num])

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            **kwargs)
    siamese_train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, 
                                      drop_last=True, 
                                      **kwargs)
    siamese_val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate, 
                                      **kwargs)
    
    return train_loader, val_loader, siamese_train_loader, siamese_val_loader

def get_test_loader(data_dir, k, max_len, seq_type, batch_size, **kwargs): 
    seq_dict = read_fasta(data_dir)
    seq_arr, labels, _ = seq2vec(seq_dict, k=k, seq_type=seq_type, max_len=max_len)
    dataset = ElasticDataSet(seq_arr, labels)
    test_loader = DataLoader(dataset, batch_size=batch_size, **kwargs)

    return test_loader