import torch
import numpy as np

from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model, ModelList

from typing import Optional, List


class TSWithExpertPref(AnalyticAcquisitionFunction):
    """
    Scalarized Thompson sampling acquisition function with expert preferences.

    Parameters:
    -----------
    model: botorch.models.model.Model

    model_pref: botorch.models.model.Model

    gamma: float > 0, default = 0
        Contribution strength of model_pref.

    maximize: bool, default = True
        Whether to maximize the acqf

    random_state: int, default = 123
        The random state of the sampling f_s ~ p(f | D). This is to ensure that for any given x,
        the sample from p(f(x) | D) comes from the same sample posterior sample f_s ~ p(f | D).
    """
    def __init__(
        self,
        model: Model,
        model_pref: Model,
        gamma: float = 0,
        maximize: bool = True,
        random_state: int = 123
    )-> None:
        super().__init__(model)
        self.model_pref = model_pref
        self.gamma = gamma
        self.maximize = maximize
        self.random_state = random_state

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
        mean, std = self._mean_and_sigma(x)

        # Thompson sample; deterministic via the random state
        generator = torch.Generator(device=x.device).manual_seed(self.random_state)
        eps = torch.randn(*std.shape, device=x.device, generator=generator)
        f_sample = mean + std * eps
        f_sample *= 1 if self.maximize else -1

        # Thompson sample for the expert preference model
        posterior_pref = self.model_pref.posterior(x)
        mean_pref = posterior_pref.mean.view(mean.shape)
        std_pref = posterior_pref.variance.sqrt().view(std.shape)
        pref_sample = mean_pref + std_pref * eps  # Always maximization

        return f_sample + self.gamma * pref_sample

