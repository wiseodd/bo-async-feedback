import torch
import torch.utils.data as data_utils
from collections import UserDict

from typing import *


class RewardDataset(data_utils.Dataset):
    """
    Parameters:
    -----------
    dataset: UserDict({'x_0': torch.Tensor, 'x_1': torch.Tensor, 'labels': torch.LongTensor})
    """
    def __init__(self, dataset):
        assert len(dataset['x_0']) == len(dataset['x_1']) == len(dataset['labels'])
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset['labels'])

    def __getitem__(self, idx):
        return (
            self.dataset['x_0'][idx],
            self.dataset['x_1'][idx],
            self.dataset['labels'][idx]
        )


def reward_data_collator(batch: List[tuple]) -> UserDict:
    """
    Transforming a list of tuples from `__getitem__` in `RewardDataset` to
    a `UserDict` where each `x_0`, `x_1`, `labels` are batches of data.
    """
    x_0s, x_1s, labels = [], [], []
    for x_0, x_1, label in batch:
        x_0s.append(x_0.unsqueeze(0))
        x_1s.append(x_1.unsqueeze(0))
        labels.append(label.unsqueeze(0))

    return UserDict({
        'x_0': torch.cat(x_0s, dim=0),
        'x_1': torch.cat(x_1s, dim=0),
        'labels': torch.cat(labels, dim=0)
    })
