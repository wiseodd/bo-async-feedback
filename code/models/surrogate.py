from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')

from torch import nn
import torch.nn.functional as F

import torch
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import Kernel

from botorch.models.model import Model
from botorch.models.gp_regression import SingleTaskGP

from typing import List


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim=1, normalize_output=False):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, out_dim)

        self.normalize_output = normalize_output
        self.register_buffer('f_mean', torch.tensor(0., dtype=torch.float))
        self.register_buffer('f_std', torch.tensor(1., dtype=torch.float))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)

        if self.normalize_output:
            out = out - getattr(self, 'f_mean')
            out = out / getattr(self, 'f_std')

        return out


class MLLGP(SingleTaskGP):

    def __init__(
            self,
            train_X: torch.Tensor,
            train_Y: torch.Tensor,
            kernel: Kernel | None = None,  # Default to Matern
            likelihood: Likelihood | None = None,
            lr: float = 0.01,
            n_epochs: int = 500,
    ):
        self.orig_train_X = train_X
        self.orig_train_Y = train_Y
        super().__init__(
            train_X=train_X, train_Y=train_Y,
            likelihood=likelihood, covar_module=kernel
        )
        self.kernel = kernel
        self.lr = lr
        self.n_epochs = n_epochs

        self._train_model()

    def _train_model(self):
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.train()
        self.likelihood.train()
        mll.train()

        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            output = self(self.train_inputs[0])
            loss = (-mll(output, self.train_targets)).mean()
            loss.backward()
            optimizer.step()

        self.eval()
        self.likelihood.eval()

    def condition_on_observations(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        **kwargs
    ) -> MLLGP:
        train_X = torch.cat([self.train_inputs[0], X])
        train_Y = torch.cat([self.train_targets.unsqueeze(-1), Y])
        return MLLGP(
            train_X, train_Y, self.kernel,
            self.likelihood, self.lr, self.n_epochs
        )
