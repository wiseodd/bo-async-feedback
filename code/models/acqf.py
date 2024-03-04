import torch

from botorch.acquisition import AcquisitionFunction
from botorch.models.model import ModelList

from typing import Optional, List


class ScalarizedTSWithExpert(AcquisitionFunction):
    """
    Scalarized Thompson sampling acquisition function.

    Parameters:
    -----------
    model: botorch.models.model.Model

    posterior_transform: botorch.acquisition.objective.PosteriorTransform
        Optional

    maximize: bool, default = True
        Whether to maximize the acqf f_s or minimize it

    random_state: int, default = 123
        The random state of the sampling f_s ~ p(f | D). This is to ensure that for any given x,
        the sample from p(f(x) | D) comes from the same sample posterior sample f_s ~ p(f | D).
    """
    def __init__(
        self,
        model: ModelList,
        weights: Optional[torch.Tensor] = None,
        maximize: bool = True,
        random_state: int = 123
    )-> None:
        super().__init__(model)

        self.maximize = maximize
        self.random_state = random_state
        self.n_models = len(model.models)

        if weights is None:
            weights = torch.ones(self.n_models)
            weights /= self.n_models  # convex combination
        assert weights.sum() == 1
        self.weights = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        x: torch.Tensor
            Shape (n, 1, dim)

        Returns:
        --------
        f_sample: torch.Tensor
            Shape (n,)
        """
        posterior = self.model.posterior(x)
        # Each (n, 1, n_models)
        mean, std = posterior.mean, posterior.variance.sqrt()

        generator = torch.Generator(device=x.device).manual_seed(self.random_state)
        eps = torch.randn(*std.shape, device=x.device, generator=generator)
        f_samples = mean + std * eps

        # Scalarize to (n, 1, 1), then (n,)
        if self.weights.device != x.device:
            self.weights.to(x.device)

        f_sample = torch.einsum('nom,m->no', f_samples, self.weights)
        f_sample = f_sample.squeeze()

        # BoTorch assumes acqf to be maximization
        return f_sample if self.maximize else -f_sample

