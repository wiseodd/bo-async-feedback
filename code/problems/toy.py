import torch
from botorch.test_functions import Ackley, Hartmann, Rastrigin, Levy
import numpy as np


class Problem:
    """
    Parameters:
    -----------
    dim: int

    bounds: torch.Tensor
        Shape (2, dim)

    is_maximize: bool
    """
    def __init__(self, dim, bounds, is_maximize):
        self.dim = dim
        self.bounds = bounds
        self.is_maximize = is_maximize
        self.optimal_value = self.get_function().optimal_value

    def get_function(self):
        """
        Returns:
        --------
        func: botorch.test_functions.BaseTestProblem
        """
        raise NotImplementedError

    def get_preference(self, x_0, x_1):
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
        def score(x):
            return -torch.linalg.norm(x - torch.tensor(self.preferred_x, device=x.device).float())

        return np.argmax([score(x_0), score(x_1)])

    @property
    def preferred_x(self):
        """
        Returns:
        --------
        preferred_x: List[float]
            Length equals self.dim
        """
        raise NotImplementedError


class Ackley2(Problem):
    def __init__(self):
        dim = 2
        bounds = torch.tensor(dim*[[-32.768, 32.768]]).T
        is_maximize = False
        super().__init__(dim, bounds, is_maximize)

    def get_function(self):
        return Ackley(dim=self.dim, bounds=self.bounds.T)

    @property
    def preferred_x(self):
        return self.dim*[10.]


class Ackley10(Problem):
    def __init__(self):
        dim = 10
        bounds = torch.tensor(dim*[[-32.768, 32.768]]).T
        is_maximize = False
        super().__init__(dim, bounds, is_maximize)

    def get_function(self):
        return Ackley(dim=self.dim, bounds=self.bounds.T)

    @property
    def preferred_x(self):
        return self.dim*[5.]


class Hartmann6(Problem):
    def __init__(self):
        dim = 6
        bounds = torch.tensor(dim*[[0., 1.]]).T
        is_maximize = False
        super().__init__(dim, bounds, is_maximize)

    def get_function(self):
        return Hartmann(dim=self.dim, bounds=self.bounds.T)

    @property
    def preferred_x(self):
        return self.dim*[0.]


class Levy10(Problem):
    def __init__(self):
        dim = 10
        bounds = torch.tensor(dim*[[-10., 10.]]).T
        is_maximize = False
        super().__init__(dim, bounds, is_maximize)

    def get_function(self):
        return Levy(dim=self.dim, bounds=self.bounds.T)

    @property
    def preferred_x(self):
        return self.dim*[0.]


class Rastrigin10(Problem):
    def __init__(self):
        dim = 10
        bounds = torch.tensor(dim*[[-5.12, 5.12]]).T
        is_maximize = False
        super().__init__(dim, bounds, is_maximize)

    def get_function(self):
        return Rastrigin(dim=self.dim, bounds=self.bounds.T)

    @property
    def preferred_x(self):
        return self.dim*[-3.]


PROBLEM_LIST = {
    'ackley2': Ackley2,
    'ackley10': Ackley10,
    'hartmann6': Hartmann6,
    'levy10': Levy10,
    'rastrigin10': Rastrigin10
}


if __name__ == '__main__':
    problem = Hartmann6()
    f = problem.get_function()
    print(f(torch.tensor(problem.dim*[0.]).float()), f.optimal_value)

    for _ in range(10):
        x0, x1 = torch.randn(problem.dim), torch.randn(problem.dim)
        print(problem.get_preference(x0, x1))
