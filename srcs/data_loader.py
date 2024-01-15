# Passion4ever

from mindspore.dataset import GeneratorDataset
from mindspore.ops import cat
from .utils import read_fasta, seq2vec
import numpy as np
from mindspore import Tensor


class ElasticDataSet:
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


class SiameseDataSet:
    def __init__(self, *arrays):
        length = len(arrays)
        if length == 0:
            raise ValueError("At least one array required as input")

        zipped_data = list(zip(*arrays))
        np.random.shuffle(zipped_data)

        self.arrays = list(zip(*zipped_data))

        self.n_samples = len(self.arrays[0])

    def __len__(self):
        return self.n_samples // 2 

    def __getitem__(self, idx):
        idx *= 2 
        array_A = self.arrays[0][idx]
        array_B = self.arrays[0][idx + 1]
        label_A = self.arrays[1][idx]
        label_B = self.arrays[1][idx + 1]
        contrast_label = label_A ^ label_B 

        return [array_A, array_B, contrast_label, label_A, label_B]



def get_dataloaders(data_dir, k, max_len, seq_type, val_size, batch_size, num_parallel_workers):
    # data
    seq_dict = read_fasta(data_dir)
    seq_arr, labels, _ = seq2vec(seq_dict, k=k, seq_type=seq_type, max_len=max_len)
    assert val_size is not None and val_size > 0 and val_size < 1, \
        'You must set <val_size> between 0 and 1'
    # e_dataset
    e_set = ElasticDataSet(seq_arr, labels)
    e_set = GeneratorDataset(e_set, 
                             column_names=["data", "label"], 
                             shuffle=False, 
                             num_parallel_workers=num_parallel_workers,
                            )
    train_set, valid_set = e_set.split([1-val_size, val_size])
    train_loader = train_set.batch(batch_size)
    valid_loader = valid_set.batch(batch_size)
    # siamese_dataset
    s_set = SiameseDataSet(seq_arr, labels)
    s_set = GeneratorDataset(s_set, 
                             column_names=["seq1", "seq2", "label", "label1", "label2"], 
                             shuffle=False, 
                             num_parallel_workers=num_parallel_workers,
                            )
    siamese_train_set, siamese_valid_set = s_set.split([1-val_size, val_size])
    siamese_train_loader = siamese_train_set.batch(batch_size)
    siamese_valid_loader = siamese_valid_set.batch(batch_size)

    return train_loader, valid_loader, siamese_train_loader, siamese_valid_loader
