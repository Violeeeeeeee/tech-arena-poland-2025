import torch

def kfold_split(dataset, k: int, seed=42):
    """
    k-fold split function

    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds (with shuffling by default).

    Inputs:
    k: number of folds
    seed=42: seed for random shuffle
    """
    n = len(dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) # shuffle index
    fold_size = n // k
    folds = []

    for i in range(k):
        val_idx = idx[i * fold_size : (i + 1) * fold_size]
        train_idx = torch.cat([idx[:i * fold_size], idx[(i + 1) * fold_size:]])
        folds.append((train_idx, val_idx))

    return folds


def data_split(dataset, train_ratio=0.9, seed=42):
    """
    Split dataset into train/test sets with shuffling.

    Inputs:
    - train_ratio: float (0..1) portion for training
    - seed: random seed
    """
    n = len(dataset)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))

    train_size = int(n * train_ratio)

    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    return train_idx, test_idx

