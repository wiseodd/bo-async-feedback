import torch
import torch.distributions as dists
import itertools
import numpy as np
from collections import UserDict
import torchmetrics as tm
from botorch.models.transforms.outcome import Standardize

from typing import *


def y_transform(new_y, train_Y):
    trf = Standardize(1)
    trf.train()
    trf(train_Y)  # Fit
    trf.eval()
    return trf(new_y)[0]

def sample_x(num_points, bounds):
    """
    Sample uniformly from the input space with the bounds

    Parameters:
    -----------
    num_points: int

    bounds: torch.Tensor
        Shape (2, dim)

    Returns:
    --------
    samples: torch.Tensor
        Shape (num_points, dim)
    """
    samples = torch.cat([
        dists.Uniform(*bounds.T[i]).sample((num_points, 1))
        for i in range(bounds.shape[1])  # for each dimension
    ], dim=1)
    return samples


def sample_pair_idxs(source, num_samples):
    """
    Sample pairs (in the form of indices) from source.
    Without replacement.

    Parameters:
    -----------
    source: Iterable

    num_samples: int
        Must be < len(source) and > 0

    Returns:
    --------
    samples: np.array
        Shape (num_samples, 2)
    """
    idx_pairs = itertools.combinations(range(len(source)), 2)
    idx_pairs = list(idx_pairs)
    np.random.shuffle(idx_pairs)
    return idx_pairs[:num_samples]  # (num_samples, 2)


def sample_pref_data(source, pref_func, num_samples, exclude_indices=[], output_indices=False):
    """
    Sample preference data list (x_0, x_1, label) from source.
    The label is obtained by calling `pref_func(x_0, x_1)`.
    Without replacement.

    Parameters:
    -----------
    source: Iterable

    pref_func: Callable
        Takes two tensors, outputs 0 or 1

    num_samples: int
        Must be < len(source) and > 0

    exclude_indices: np.array of ints with shape (n, 2), default=[]
        If a sampled pair idxs is in this array, then skip.

    Returns:
    --------
    data_pref: UserDict
        Format:
        ```
        UserDict({
            'x_0': torch.FloatTensor(n, d),
            'x_1': torch.FloatTensor(n, d),
            'labels': torch.LongTensor(n,)
        })
        ```
    """
    idx_pairs = sample_pair_idxs(source, num_samples)
    # exclude_idxs_lst = exclude_indices.tolist()
    x_0s, x_1s, labels = [], [], []
    idx_pairs_excluded = []

    # print(len(idx_pairs))

    for idx_pair in idx_pairs:
        print(idx_pair in exclude_indices)
        if idx_pair in exclude_indices:
            continue

        x_0, x_1 = source[idx_pair[0]].unsqueeze(0), source[idx_pair[1]].unsqueeze(0)
        x_0s.append(x_0)
        x_1s.append(x_1)
        labels.append(torch.tensor(pref_func(x_0, x_1)).long().reshape(1,))
        idx_pairs_excluded.append(idx_pair)

    data_pref = UserDict({
        'x_0': torch.cat(x_0s, dim=0),
        'x_1': torch.cat(x_1s, dim=0),
        'labels': torch.cat(labels, dim=0)
    })

    print(len(x_0s)); input()

    return (data_pref, idx_pairs_excluded) if output_indices else data_pref


def subset_pref_data(pref_data: UserDict, subset_idxs: Iterable[int] | torch.LongTensor):
    """
    Get preference data by indices.

    Parameters:
    -----------
    pref_data: UserDict({
        'x_0': torch.Tensor(n, d),
        'x_1': torch.Tensor(n, d),
        'labels': torch.Tensor(n, d)
    })

    subset_idxs: Iterable[int] | torch.LongTensor
        Shape (m,)

    Returns:
    --------
    subset_pref_data: UserDict({
        'x_0': torch.Tensor(m, d),
        'x_1': torch.Tensor(m, d),
        'labels': torch.Tensor(m, d)
    })
        ```
    """
    return UserDict({
        'x_0': pref_data['x_0'][subset_idxs],
        'x_1': pref_data['x_1'][subset_idxs],
        'labels': pref_data['labels'][subset_idxs]
    })


def is_pair_selected(x0: torch.Tensor, x1: torch.Tensor, pref_data: UserDict) -> bool:
    """
    Check if `(x0, x1)` is already in `pref_data`.

    Parameters:
    -----------
    x0: torch.Tensor
        Shape (d,)

    x1: torch.Tensor
        Shape (d,)

    pref_data: UserDict({
        'x_0': torch.Tensor(n, d),
        'x_1': torch.Tensor(n, d),
        'labels': torch.Tensor(n, d)
    })

    Returns:
    --------
    selected: bool
        True if `(x0, x1)` is already in `pref_data`.
    """
    for data_x0, data_x1 in zip(pref_data['x_0'], pref_data['x_1']):
        if torch.allclose(x0, data_x0) and torch.allclose(x1, data_x1):
            return True

        # Check also the opposite order
        if torch.allclose(x1, data_x0) and torch.allclose(x0, data_x1):
            return True

    return False



