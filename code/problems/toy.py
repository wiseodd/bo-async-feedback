import abc
import torch
from botorch.test_functions import Ackley, Hartmann, Rastrigin, Levy
from botorch.test_functions.base import BaseTestProblem
import numpy as np


class ToyProblem(abc.ABC):
    """
    Parameters:
    -----------
    dim: int

    is_maximize: bool
    """

    def __init__(self, dim: int, is_maximize: bool) -> None:
        self.dim = dim
        self.is_maximize = is_maximize
        self.bounds = torch.tensor(self.get_function()._bounds).T  # (2, dim)
        self.optimal_f = self.get_function().optimal_value
        self.optimal_x = self.get_function().optimizers[0]

    def get_function(self) -> BaseTestProblem:
        """
        Returns:
        --------
        func: botorch.test_functions.BaseTestProblem
        """
        raise NotImplementedError

    def get_preference(self, x_0: torch.Tensor, x_1: torch.Tensor) -> int:
        """
        Given a pair x_0 and x_1, return 0 if the first one is preferred,
        otherwise return 1.

        Parameters:
        -----------
        x_0: torch.Tensor
            Shape (n_dim,)

        x_1: torch.Tensor
            See above.

        Returns:
        --------
        label: int
            Either 0 or 1, depending which x's is preferred
        """
        return np.argmax([self._score(x_0), self._score(x_1)])

    def _score(self, x: torch.Tensor) -> float:
        raise NotImplementedError


class Ackley2(ToyProblem):
    def __init__(self):
        dim = 2
        is_maximize = False
        super().__init__(dim, is_maximize)

    def get_function(self):
        return Ackley(dim=self.dim)

    def _score(self, x):
        return -torch.linalg.norm(x - self.optimal_x) ** 2


class Ackley10(ToyProblem):
    def __init__(self):
        dim = 10
        is_maximize = False
        super().__init__(dim, is_maximize)

    def get_function(self):
        return Ackley(dim=self.dim)

    def _score(self, x):
        return -torch.linalg.norm(x - self.optimal_x) ** 2


class Hartmann6(ToyProblem):
    def __init__(self):
        dim = 6
        is_maximize = False
        super().__init__(dim, is_maximize)

    def get_function(self):
        return Hartmann(dim=self.dim)

    def _score(self, x):
        return -torch.linalg.norm(x - self.optimal_x) ** 2


class Levy10(ToyProblem):
    def __init__(self):
        dim = 10
        is_maximize = False
        super().__init__(dim, is_maximize)

    def get_function(self):
        return Levy(dim=self.dim)

    def _score(self, x):
        return -torch.linalg.norm(x - self.optimal_x) ** 2


class Rastrigin10(ToyProblem):
    def __init__(self):
        dim = 10
        is_maximize = False
        super().__init__(dim, is_maximize)

    def get_function(self):
        return Rastrigin(dim=self.dim)

    def _score(self, x):
        return -torch.linalg.norm(x - self.optimal_x) ** 2


class Ackley10Constrained(ToyProblem):
    def __init__(self):
        dim = 10
        is_maximize = False
        super().__init__(dim, is_maximize)
        self.preferred_norm = 100

    def get_function(self):
        return Ackley(dim=self.dim)

    def _score(self, x):
        return -abs(torch.linalg.norm(x) - self.preferred_norm) ** 2


class Levy10Constrained(ToyProblem):
    def __init__(self):
        dim = 10
        is_maximize = False
        super().__init__(dim, is_maximize)
        self.preferred_norm = 22

    def get_function(self):
        return Levy(dim=self.dim)

    def _score(self, x):
        return -abs(torch.linalg.norm(x) - self.preferred_norm) ** 2


PROBLEM_LIST = {
    'ackley2': Ackley2,
    'ackley10': Ackley10,
    'hartmann6': Hartmann6,
    'levy10': Levy10,
    'rastrigin10': Rastrigin10,
    'ackley10constrained': Ackley10Constrained,
    'levy10constrained': Levy10Constrained,
}


if __name__ == '__main__':
    problem = Levy10()
    print(problem.bounds.shape)
    print(problem.optimal_x, problem.optimal_f)

    for _ in range(10):
        x0, x1 = torch.randn(problem.dim), torch.randn(problem.dim)
        print(problem.get_preference(x0, x1))
