from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')
import torch
from torch import nn, optim
from gpytorch import distributions as gdists
import torch.utils.data as data_utils
import botorch.models.model as botorch_model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from laplace import Laplace
from laplace.curvature import BackPackGGN, CurvatureInterface, AsdlGGN
from laplace.marglik_training import marglik_training
from typing import *
import math
from collections import UserDict
import torchmetrics as tm

from utils.data import ListDataset


class PrefLaplaceBoTorch(botorch_model.Model):
    """
    BoTorch surrogate model to model preference with a Laplace-approximated
    Bayesian neural network. The Laplace class is defined in the library
    laplace-torch; install via: `pip install https://github.com/aleximmer/laplace.git`.

    Args:
    -----
    get_net: function None -> nn.Module
        Function that doesn't take any args and return a PyTorch model.
        Prefer torch.nn.Sequential model due to BackPACK dependency.
        Example usage: `get_net=lambda: nn.Sequential(...)`.

    train_data : List[Dict] or List[UserDict]
        Training inputs and labels.

    bnn : Laplace, optional, default=None
        When creating a new model from scratch, leave this at None.
        Use this only to update this model with a new observation during BO.

    likelihood : {'regression', 'classification'}
        Indicates whether the problem is regression or classification.

    noise_var : float | None, default=None.
        Output noise variance. If float, must be >= 0. If None,
        it is learned by marginal likelihood automatically.

    last_layer : bool, default False
        Whether to do last-layer Laplace. If True, then the model used is the
        so-called "neural linear" model.

    hess_factorization : {'full', 'diag', 'kron'}, default='kron'
        Which Hessian factorization to use to do Laplace. 'kron' provides the best
        tradeoff between speed and approximation error.

    posthoc_marglik_iters: int > 0, default=100
        Number of iterations of post-hoc marglik tuning.

    batch_size : int, default=10
        Batch size to use for the NN training and doing Laplace.

    n_epochs : int, default=1000
        Number of epochs for training the NN.

    lr : float, default=1e-1
        Learning rate to use for training the NN.

    wd : float, default=1e-3
        Weight decay for training the NN.

    device : {'cpu', 'cuda'}, default='cpu'
        Which device to run the experiment on.
    """
    def __init__(
        self,
        get_net: Callable[[], nn.Module],
        train_data: List[Dict] | List[UserDict],
        bnn: Laplace = None,
        noise_var: float | None =  None,
        last_layer: bool = False,
        hess_factorization: str = 'kron',
        posthoc_marglik_iters: int = 100,
        batch_size: int = 10,
        n_epochs: int = 1000,
        lr: float = 1e-1,
        wd: float = 1e-3,
        backend: CurvatureInterface = AsdlGGN,
        device: str ='cpu',
        enable_backprop: bool = True
    ):
        super().__init__()
        self.train_data = train_data
        self.batch_size = batch_size
        self.last_layer = last_layer
        self.subset_of_weights = 'last_layer' if last_layer else 'all'
        self.hess_factorization = hess_factorization
        self.posthoc_marglik_iters = posthoc_marglik_iters
        assert device in ['cpu', 'cuda']
        self.device = device
        self.n_epochs = n_epochs
        self.lr = lr
        self.wd = wd
        self.backend = backend
        self.enable_backprop = enable_backprop
        self.get_net = get_net
        self.bnn = bnn

        if type(noise_var) != float and noise_var is not None:
            raise ValueError('Noise variance must be float >= 0. or None')
        if type(noise_var) == float and noise_var < 0:
            raise ValueError('Noise variance must be >= 0.')
        self.noise_var = noise_var

        # Initialize Laplace
        if self.bnn is None:
            self._train_model(self._get_train_loader())


    def posterior(
        self,
        X: torch.Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        # Notation:
        # ---------
        # B is the batch size
        # Q is the num. of x's predicted jointly
        # D is the feature size
        # K is the output size, i.e. num of tasks

        # Transform to `(B*Q, D)`
        B, Q, D = X.shape
        X = X.reshape(B*Q, D)

        # Posterior predictive distribution
        # mean_y is (B*Q, K); cov_y is (B*Q*K, B*Q*K)
        mean_y, cov_y = self.get_prediction(X, use_test_loader=False, joint=True)

        # Mean must be `(B, Q*K)`
        K = self.num_outputs
        mean_y = mean_y.reshape(B, Q*K)

        # Cov must be `(B, Q*K, Q*K)`
        cov_y += self.bnn.sigma_noise**2 * torch.eye(B*Q*K, device=self.device)
        cov_y = cov_y.reshape(B, Q, K, B, Q, K)
        cov_y = torch.einsum('bqkbrl->bqkrl', cov_y)  # (B, Q, K, Q, K)
        cov_y = cov_y.reshape(B, Q*K, Q*K)

        dist = gdists.MultivariateNormal(mean_y, covariance_matrix=cov_y)
        post_pred = GPyTorchPosterior(dist)

        if posterior_transform is not None:
            return posterior_transform(post_pred)

        return post_pred


    def condition_on_observations(self, data: List[Dict] | List[UserDict], **kwargs: Any) -> PrefLaplaceBoTorch:
        # Append new observation to the current data
        self.train_data += data

        # Update Laplace with the updated data
        train_loader = self._get_train_loader()
        self._train_model(train_loader)

        return PrefLaplaceBoTorch(
            # Replace the dataset & retrained BNN
            get_net=self.get_net,
            train_data=self.train_data,  # Important!
            bnn=self.bnn,  # Important!
            noise_var=self.noise_var,
            last_layer=self.last_layer,
            hess_factorization=self.hess_factorization,
            posthoc_marglik_iters=self.posthoc_marglik_iters,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            lr=self.lr,
            wd=self.wd,
            device=self.device
        )


    def get_prediction(self, test_X: torch.Tensor, joint=True, use_test_loader=False):
        """
        Batched Laplace prediction.

        Args:
        -----
        test_X: torch.Tensor
            Array of size `(batch_shape, feature_dim)`.

        joint: bool, default=True
            Whether to do joint predictions (like in GP).

        use_test_loader: bool, default=False
            Set to True if your test_X is large.


        Returns:
        --------
        mean_y: torch.Tensor
            Tensor of size `(batch_shape, num_tasks)`.

        cov_y: torch.Tensor
            Tensor of size `(batch_shape*num_tasks, batch_shape*num_tasks)`
            if joint is True. Otherwise, `(batch_shape, num_tasks, num_tasks)`.
        """
        if self.bnn is None:
            raise Exception('Train your model first before making prediction!')

        if not use_test_loader:
            mean_y, cov_y = self.bnn(test_X.to(self.device), joint=joint)
        else:
            test_loader = data_utils.DataLoader(
                data_utils.TensorDataset(test_X, torch.zeros_like(test_X)),
                batch_size=256
            )

            mean_y, cov_y = [], []

            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                _mean_y, _cov_y = self.bnn(X_batch, joint=joint)
                mean_y.append(_mean_y)
                cov_y.append(_cov_y)

            mean_y = torch.cat(mean_y, dim=0).squeeze()
            cov_y = torch.cat(cov_y, dim=0).squeeze()

        return mean_y, cov_y


    @property
    def num_outputs(self) -> int:
        """The number of outputs of the model."""
        return 1  # Always 1 for reward modeling


    def _train_model(self, train_loader):
        del self.bnn
        self._posthoc_laplace(train_loader)

        # Override sigma_noise if self.noise_var is not None
        if self.noise_var is not None:
            self.bnn.sigma_noise = math.sqrt(self.noise_var)


    def _posthoc_laplace(self, train_loader):
        net = self.get_net()  # Ensure that the base net is re-initialized
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_epochs*len(train_loader))
        loss_func = nn.CrossEntropyLoss()

        for _ in range(self.n_epochs):
            for data in train_loader:
                optimizer.zero_grad()
                output = net(data)
                loss = loss_func(output, data['labels'])
                loss.backward()
                optimizer.step()
                scheduler.step()

        net.eval()

        # Reward model normalization
        with torch.no_grad():
            mean_metric = tm.MeanMetric()
            mean_sq_metric = tm.MeanMetric()

            for data in train_loader:
                out = net(data).flatten()  # (batch_size*2,)
                mean_metric(out)
                mean_sq_metric(out**2)

            mean = mean_metric.compute()
            sq_mean = mean**2  # E(f(x))^2
            mean_sq = mean_sq_metric.compute()  # E(f(x)^2)
            std = torch.sqrt(mean_sq - sq_mean)  # std(f(x))

            net.f_mean = mean
            net.f_std = std

        # Laplace
        self.bnn = Laplace(
            net, likelihood='reward_modeling',
            subset_of_weights=self.subset_of_weights,
            hessian_structure=self.hess_factorization,
            backend=self.backend,
            enable_backprop=self.enable_backprop
        )
        self.bnn.fit(train_loader)
        self.bnn.optimize_prior_precision(n_steps=self.posthoc_marglik_iters)


    def _get_train_loader(self):
        return data_utils.DataLoader(
            ListDataset(self.train_data),
            batch_size=self.batch_size, shuffle=True
        )
