# Passion4ever

import os
import random
from functools import partial, update_wrapper
from importlib import import_module
from pathlib import Path
from typing import Dict, Mapping

import hydra
import numpy as np
import pandas as pd
import torch
import yaml


def instantiate(config, *args, is_func=False, **kwargs):
    """wrapper function for hydra.utils.instantiate.
    
    1. return None if config.class is None
    2. return function handle if is_func is True
    """

    assert (
        "_target_" in config
    ), f"Config should have '_target_' for class instantiation."
    target = config["_target_"]
    if target is None:
        return None
    if is_func:
        modulename, funcname = target.rsplit(".", 1)
        mod = import_module(modulename)
        func = getattr(mod, funcname)

        kwargs.update({k: v for k, v in config.items() if k != "_target_"})
        partial_func = partial(func, *args, **kwargs)

        update_wrapper(partial_func, func)
        return partial_func
    return hydra.utils.instantiate(config, *args, **kwargs)


def read_yaml(file_path: str, encoding: str = "utf-8") -> Mapping[str, Dict]:
    """Read yaml file and return a dictionary with file contents.

    If there are multiple sets of data in the yaml file,
    the key of the dictionary will be used as the identifier of multiple sets.
    """

    file_path = Path(file_path)
    with file_path.open("r", encoding=encoding) as f:
        dict_generator = yaml.load_all(f.read(), Loader=yaml.FullLoader)
    dict_list = list(dict_generator)
    yaml_dict = {}
    if len(dict_list) == 1:
        (yaml_dict,) = dict_list
    else:
        for idx, i in enumerate(dict_list):
            yaml_dict[idx] = i
    return yaml_dict


def write_yaml(file_path: str, *objs, encoding: str = "utf-8") -> None:
    """Write objects to a yaml file."""

    file_path = Path(file_path)
    with file_path.open("w", encoding=encoding) as f:
        yaml.dump_all(
            documents=objs, stream=f, allow_unicode=True, indent=4, sort_keys=False
        )


def read_fasta(file_path):
    sequences = {}
    header = None
    with open(file_path) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:]
                if not header:
                    raise ValueError("Error: empty sequence header.")
                if header in sequences:
                    raise ValueError(f"Error: duplicate sequence header '{header}'.")
                sequences[header] = []
            else:
                if header is None:
                    raise ValueError(
                        "Error: sequence data found before any sequence header."
                    )
                sequences[header].append(line)

        sequences = {header: "".join(seq) for header, seq in sequences.items()}
    return sequences


def seq2vec(seq_dict, k, seq_type, max_len=None):
    from functools import reduce

    import torch
    import torch.nn.functional as F

    assert k >= 1, " k must be an integer greater than zero. "
    assert max_len is None or (
        isinstance(max_len, int) and max_len > 0
    ), " max_len must be a positive integer or `None` . "

    alphabet = {
        "dna": list("ATCG"),
        "protein": list("ACDEFGHIKLMNPQRSTVWY"),
    }

    total_kmer = reduce(
        lambda x, y: [i + j for i in x for j in y], [alphabet[seq_type]] * k
    )
    kmer_map = dict(zip(total_kmer, range(len(total_kmer))))

    seq_lis = []
    label_lis = []
    kmer_lis = []
    LABEL_DICT = {"promoter": 1, "non_promoter": 0}
    for key, seq in seq_dict.items():
        if max_len is not None and len(seq) > max_len:
            seq = "".join(list(seq)[:max_len])
        seq_lis.append(seq)
        label_lis.append(LABEL_DICT[key.split("|")[2]])
        integer = [kmer_map[seq[i : i + k]] for i in range(len(seq) - k + 1)]
        kmer_lis.append(integer)

    kmer_lis = torch.tensor(kmer_lis)
    label_lis = torch.tensor(label_lis)

    if k == 1:
        kmer_lis = F.one_hot(kmer_lis).float()

    return kmer_lis, label_lis, seq_lis



def set_global_random_seed(seed):
    """Unify all random number seeds to ensure reproducibility as much as possible.

    Args:
        seed (int): Random seed number.
    """

    assert isinstance(seed, int) and 0 <= seed <= (
        2**32 - 1
    ), "Seed must be int and between 0 and 2**32 - 1!"
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True  # May slow down the speed of code running
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving.

    Attributes:
        mode (str, optional): The mode of metric optimization. Defaults to "min".
        patience (int, optional): The number of epochs to wait after the last time the metric improved. Defaults to 10.
        delta (float, optional): The minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.
    """

    def __init__(self, mode: str = "min", patience: int = 10, delta: float = 0) -> None:
        """Initialize EarlyStopping."""

        if mode not in ['min', "max"]:
            raise ValueError("mode must be 'min' or 'max'.")

        self.mode = mode
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False

    def __call__(self, metric: float) -> tuple[bool, float, int]:
        """Earlystop call"""

        improved = (self.mode == "min" and metric < self.best_score - self.delta) or (
            self.mode == "max" and metric > self.best_score + self.delta
        )

        if improved:
            self.best_score = metric
            self.counter = 0
            update = True
            return update, self.best_score, self.counter
        else:
            self.counter += 1
            update = False
            if self.counter >= self.patience:
                self.early_stop = True
            return update, self.best_score, self.counter


class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(columns=["epoch", "step", *keys])
        self._data.set_index(["epoch", "step"], inplace=True)

    def add(self, metric_dic, epoch, step):
        for key, value in metric_dic.items():
            self._data.loc[(epoch, step), key] = value

    def step_result(self):
        return dict(self._data.iloc[-1])

    def epoch_result(self):
        return dict(self.epoch_data.iloc[-1])

    def log_epoch_metrics(self, description="", logger=print, epoch=-1):
        logger(
            f"{description:>12s}: "
            f"{' '.join([f'[{key}: {value:.6f}]' for key, value in self.epoch_data.iloc[epoch, :].items()])}"
        )

    @property
    def step_data(self):
        return self._data.groupby("step").mean(numeric_only=False)

    @property
    def epoch_data(self):
        return self._data.groupby("epoch").mean(numeric_only=False)

    @property
    def data(self):
        return self._data
