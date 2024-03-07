import torch
import torch.distributions as dists
import itertools
import numpy as np
from collections import UserDict
import torchmetrics as tm
from botorch.models.transforms.outcome import Standardize


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
    return np.array(idx_pairs[:num_samples])  # (num_samples, 2)


def sample_pref_data(source, pref_func, num_samples):
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

    Returns:
    --------
    data_pref: List[UserDict]
        Length num_samples. Format:
        ```
        UserDict({
            'x_0': torch.FloatTensor,
            'x_1': torch.FloatTensor,
            'labels': torch.LongTensor
        })
        ```
    """
    idx_pairs = sample_pair_idxs(source, num_samples)
    data_pref = []
    for idx_pair in idx_pairs:
        x_0 = source[idx_pair[0]]
        x_1 = source[idx_pair[1]]
        label = torch.tensor(pref_func(x_0, x_1)).long()
        data_pref.append(UserDict({
            'x_0': x_0,
            'x_1': x_1,
            'labels': label
        }))
    return data_pref
