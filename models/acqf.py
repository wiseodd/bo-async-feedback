import torch
import numpy as np

from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model, ModelList

from collections import UserDict
from typing import Optional, List


class ThompsonSamplingWithExpertPref(AnalyticAcquisitionFunction):
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
        random_state: int = 123,
    ) -> None:
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

        if len(mean.shape) == 0:
            mean = mean.unsqueeze(0)
        if len(std.shape) == 0:
            std = std.unsqueeze(0)

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


class ThompsonSamplingRewardDiff(AnalyticAcquisitionFunction):
    """
    Thompson sampling acquisition function for reward modeling.
    Given a pair `x0` and `x1`, compute the distance `|r_s(x0) - r_s(x1)|^2`
    under a Thompson sample `r_s` of `p(r | D)`. The intuition is that
    if `r_s(x0)` and `r_s(x1)` are very different, then they provide a lot of
    signal about the preference.

    Parameters:
    -----------
    model_pref: botorch.models.model.Model

    random_state: int, default = 123
        The random state of the sampling r_s ~ p(r | D). This is to ensure that for any given x,
        the sample from p(r(x) | D) comes from the same sample posterior sample r_s ~ p(r | D).
    """

    def __init__(self, model_pref: Model, random_state: int = 123) -> None:
        super().__init__(model_pref)
        self.random_state = random_state

    def forward(self, data: UserDict) -> torch.Tensor:
        """
        Parameters:
        -----------
        data: UserDict
            Format: UserDict({
                'x_0': torch.Tensor(n, d),
                'x_1': torch.Tensor(n, d),
                'labels': torch.LongTensor(n,)
            })

        Returns:
        --------
        acqf_val: torch.Tensor
            Shape (n,)
        """
        x0, x1 = data["x_0"], data["x_1"]
        x0.to(self.model.device)
        x1.to(self.model.device)

        mean0, std0 = self._mean_and_sigma(x0)
        if len(mean0.shape) == 0:
            mean0 = mean0.unsqueeze(0)
        if len(std0.shape) == 0:
            std0 = std0.unsqueeze(0)

        mean1, std1 = self._mean_and_sigma(x1)
        if len(mean1.shape) == 0:
            mean1 = mean1.unsqueeze(0)
        if len(std1.shape) == 0:
            std1 = std1.unsqueeze(0)

        # Thompson sample; deterministic via the random state
        generator = torch.Generator(device=x0.device).manual_seed(self.random_state)
        eps = torch.randn(*std0.shape, device=x0.device, generator=generator)
        # Each (n,)
        r_sample0 = mean0 + std0 * eps
        r_sample1 = mean1 + std1 * eps

        return torch.abs(r_sample0 - r_sample1) ** 2


class BALDForRewardModel(AnalyticAcquisitionFunction):
    """
    Bayesian active learning by disagreement for preference learning reward model.
    Houlsby et al., 2011.

    Parameters:
    -----------
    model_pref: botorch.models.model.Model
    """

    def __init__(
        self,
        model_pref: Model,
    ) -> None:
        super().__init__(model_pref)

    def forward(self, data: UserDict) -> torch.Tensor:
        """
        Parameters:
        -----------
        data: UserDict
            Format: UserDict({
                'x_0': torch.Tensor(n, d),
                'x_1': torch.Tensor(n, d),
                'labels': torch.LongTensor(n,)
            })

        Returns:
        --------
        acqf_val: torch.Tensor
            Shape (n,)
        """
        x0, x1 = data["x_0"], data["x_1"]
        x0.to(self.model.device)
        x1.to(self.model.device)

        mean0, std0 = self._mean_and_sigma(x0)
        if len(mean0.shape) == 0:
            mean0 = mean0.unsqueeze(0)
        if len(std0.shape) == 0:
            std0 = std0.unsqueeze(0)

        mean1, std1 = self._mean_and_sigma(x1)
        if len(mean1.shape) == 0:
            mean1 = mean1.unsqueeze(0)
        if len(std1.shape) == 0:
            std1 = std1.unsqueeze(0)

        means = torch.stack([mean0, mean1]).T
        variances = (torch.stack([std0, std1]).T) ** 2

        # Probit approx. to get the marginalized softmax
        probs = _multiclass_probit_approx(means, variances)

        # Entropy H(y | x) of the pred. dist. p(y | x) where f is integrated out
        entropy = _entropy_cat(probs)

        # Expected entropy E_f[H(y | x, f)]
        n_samples = 100
        avg_entropy = 0
        for _ in range(n_samples):  # MC integration
            f0_sample = _sample_gaussian(mean0, std0)
            f1_sample = _sample_gaussian(mean1, std1)
            probs_sample = torch.softmax(torch.stack([f0_sample, f1_sample]).T, dim=-1)
            avg_entropy += 1 / n_samples * _entropy_cat(probs_sample)

        return entropy - avg_entropy


def _sample_gaussian(mean, std):
    eps = torch.randn(*std.shape, device=std.device)
    return mean + std * eps


def _entropy_cat(probs):
    return -torch.sum(probs * torch.log(probs), dim=-1)


def _multiclass_probit_approx(mean, var):
    if len(var.shape) == 3:  # Batch of covariance matrices
        var = torch.diagonal(var, dim1=-2, dim2=-1)

    return torch.softmax(mean / torch.sqrt(1 + torch.pi / 8 * var), dim=-1)
