import torch
import torch.distributions as dists
import itertools
import numpy as np


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
    Sample pairs (in the form of indices) from the source list.
    Without replacement.

    Parameters:
    -----------
    source: List[Any]

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
