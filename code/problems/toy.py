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

    def get_preference(xfx_0, xfx_1):
        """
        Given two pairs (x_0, f(x_0)) and (x_1, f(x_1)), return 0 if the first pair is preferred,
        otherwise return 1

        Parameters:
        -----------
        xfx_0: Tuple of torch.Tensor's
            Format: (x, f(x_0)), where x has shape (n_dim,) and f(x_0) has shape (), i.e. a scalar

        xfx_1: Tuple of torch.Tensor's
            See above

        Returns:
        --------
        label: int
            Either 0 or 1, depending which (x_i, f(x_i)) is preferred
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

    def get_preference(self, xfx_0, xfx_1):
        # 1. Prefer x that's closer to x_c = [5., ..., 5.]
        # Note that f(x_c) = 6.594, while the true global min is 0
        # 2. Prefer x whose f(x) is very close to the global min
        # Criteria 2 has the same weight as 1
        def score(x, fx):
            sub_score_1 = -torch.linalg.norm(x - torch.tensor(self.dim*[10.], device=x.device).float())
            sub_score_2 = -torch.linalg.norm(fx - self.optimal_value)
            return 0.7*sub_score_1 + 0.3*sub_score_2

        return np.argmax([score(*xfx_0), score(*xfx_1)])


class Ackley10(Problem):
    def __init__(self):
        dim = 10
        bounds = torch.tensor(dim*[[-32.768, 32.768]]).T
        is_maximize = False
        super().__init__(dim, bounds, is_maximize)

    def get_function(self):
        return Ackley(dim=self.dim, bounds=self.bounds.T)

    def get_preference(self, xfx_0, xfx_1):
        # 1. Prefer x that's closer to x_c = [5., ..., 5.]
        # Note that f(x_c) = 6.594, while the true global min is 0
        # 2. Prefer x whose f(x) is very close to the global min
        # Criteria 2 has the same weight as 1
        def score(x, fx):
            sub_score_1 = -torch.linalg.norm(x - torch.tensor(self.dim*[5.], device=x.device).float())
            sub_score_2 = -torch.linalg.norm(fx - self.optimal_value)
            return 0.5*sub_score_1 + 0.5*sub_score_2

        return np.argmax([score(*xfx_0), score(*xfx_1)])


class Hartmann6(Problem):
    def __init__(self):
        dim = 6
        bounds = torch.tensor(dim*[[0., 1.]]).T
        is_maximize = False
        super().__init__(dim, bounds, is_maximize)
        self.preferred_x = dim*[0.]

    def get_function(self):
        return Hartmann(dim=self.dim, bounds=self.bounds.T)

    def get_preference(self, x0, x1):
        def score(x):
            return -torch.linalg.norm(x - torch.tensor(self.preferred_x, device=x.device).float())

        return np.argmax([score(x0), score(x1)])


class Levy10(Problem):
    def __init__(self):
        dim = 10
        bounds = torch.tensor(dim*[[-10., 10.]]).T
        is_maximize = False
        super().__init__(dim, bounds, is_maximize)

    def get_function(self):
        return Levy(dim=self.dim, bounds=self.bounds.T)

    def get_preference(self, xfx_0, xfx_1):
        # 1. Prefer x that's closer to x_c = [0, ..., 0]
        # Note that f(x_c) = 1.442, while the true global min is 0
        # 2. Prefer x whose f(x) is very close to the global min
        # Criteria 0 has much more weight than 1
        def score(x, fx):
            sub_score_1 = -torch.linalg.norm(x - torch.tensor(self.dim*[0.], device=x.device).float())
            sub_score_2 = -torch.linalg.norm(fx - self.optimal_value)
            return 0.8*sub_score_1 + 0.2*sub_score_2

        return np.argmax([score(*xfx_0), score(*xfx_1)])


class Rastrigin10(Problem):
    def __init__(self):
        dim = 10
        bounds = torch.tensor(dim*[[-5.12, 5.12]]).T
        is_maximize = False
        super().__init__(dim, bounds, is_maximize)

    def get_function(self):
        return Rastrigin(dim=self.dim, bounds=self.bounds.T)

    def get_preference(self, xfx_0, xfx_1):
        # 1. Prefer x that's closer to x_c = [-3, ..., -3]
        # Note that f(x_c) = 90, while the true global min is 0
        # 2. Prefer x whose f(x) is very close to the global min
        # Criteria 1 has much more weight than 0
        def score(x, fx):
            sub_score_1 = -torch.linalg.norm(x - torch.tensor(self.dim*[-3.], device=x.device).float())
            sub_score_2 = -torch.linalg.norm(fx - self.optimal_value)
            return 0.2*sub_score_1 + 0.8*sub_score_2

        return np.argmax([score(*xfx_0), score(*xfx_1)])


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
