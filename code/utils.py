import torch
import torch.distributions as dists


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
