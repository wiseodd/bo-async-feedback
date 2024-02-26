import numpy as np
import torch
from torch import nn
from torch import distributions as dists
import tqdm

from botorch.test_functions import Ackley, Branin
from botorch.optim.optimize import optimize_acqf

from laplace_bayesopt.botorch import LaplaceBoTorch
from laplace_bayesopt.acqf import TSAcquisitionFunction

import toy_problems

import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--problem', default='ackley10', choices=['levy10', 'ackley10', 'hartmann6', 'rastrigin10'])
parser.add_argument('--exp_len', type=int, default=200)
parser.add_argument('--cuda', default=False, action='store_true')
parser.add_argument('--randseed', type=int, default=9999)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

problem = toy_problems.PROBLEM_LIST[args.problem]()
true_f = problem.get_function()
bounds = problem.bounds

# Initialize training data by uniform sampling within the bounds
def sample_x(num_points):
    samples = torch.cat([
        dists.Uniform(*bounds.T[i]).sample((num_points, 1))
        for i in range(bounds.shape[1])  # for each dimension
    ], dim=1)
    return samples

train_x = sample_x(20)
train_y = true_f(train_x).reshape(-1, 1)

def get_net():
    return torch.nn.Sequential(
        nn.Linear(problem.dim, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )

model = LaplaceBoTorch(get_net, train_x, train_y, noise_var=1e-4)

best_y = train_y.min().item()
trace_best_y = []
pbar = tqdm.trange(args.exp_len)
pbar.set_description(
    f'[Best f(x) = {best_y:.3f}]'
)

# BO Loop
for i in pbar:
    acqf = TSAcquisitionFunction(model, maximize=problem.is_maximize)
    new_x, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=10, raw_samples=20)

    if len(new_x.shape) == 1:
        new_x = new_x.unsqueeze(0)  # (1, dim)

    # Evaluate the objective on the proposed x
    new_y = true_f(new_x).unsqueeze(-1)  # (q, 1)

    # Update posterior
    model = model.condition_on_observations(new_x, new_y)

    # Update the current best y
    best_cand_y = new_y.max().item() if problem.is_maximize else new_y.min().item()
    if best_cand_y <= best_y:
        best_y = best_cand_y

    trace_best_y.append(best_y)
    pbar.set_description(
        f'[Best f(x) = {best_y:.3f}, curr f(x) = {best_cand_y:.3f}]'
    )

# Save results
path = f'results/toy/{args.problem}'
if not os.path.exists(path):
    os.makedirs(path)

np.save(f'{path}/trace_best_y_{args.randseed}.npy', trace_best_y)
