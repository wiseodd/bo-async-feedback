import numpy as np
import torch
from torch import nn
from torch import distributions as dists
import tqdm

from botorch.optim.optimize import optimize_acqf
from botorch.models.model import ModelList

from laplace.curvature import CurvlinopsGGN
from laplace_bayesopt.botorch import LaplaceBoTorch
from laplace_bayesopt.acqf import TSAcquisitionFunction

import problems.toy as toy_problems
from models.surrogate import MLLGP
from models.reward import ToyRewardModel
from models.acqf import ScalarizedTSWithExpert
from utils import helpers

import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--problem', default='ackley10', choices=['levy10', 'ackley10', 'hartmann6', 'rastrigin10'])
parser.add_argument('--method', default='la', choices=['la', 'gp'])
parser.add_argument('--with_expert', default=False, action='store_true')
parser.add_argument('--exp_len', type=int, default=500)
parser.add_argument('--cuda', default=False, action='store_true')
parser.add_argument('--randseed', type=int, default=9999)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

#################################################################################################
#                           Standard Bayesian Optimization Preparation                          #
#################################################################################################
problem = toy_problems.PROBLEM_LIST[args.problem]()
true_f = problem.get_function()
bounds = problem.bounds

# Initialize training data by uniform sampling within the bounds
train_x = helpers.sample_x(20, bounds)
train_y = true_f(train_x).reshape(-1, 1)

if args.method == 'la':
    def get_net():
        return torch.nn.Sequential(
            nn.Linear(problem.dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
    model = LaplaceBoTorch(get_net, train_x, train_y, noise_var=1e-4, batch_size=1024, backend=CurvlinopsGGN)
elif args.method == 'gp':
    # Hotfix for a numerical issue (not PSD error)
    n_epochs = (0 if args.problem == 'hartmann6' else 500)
    model = MLLGP(train_x, train_y, n_epochs=n_epochs)

#################################################################################################
#                                  Expert Feedback Modeling                                     #
#################################################################################################
if args.with_expert:
    # Use a pretrained reward model to simulate expert preferences
    # (x, f(x)) -> expert(x, f(x)) scalar
    expert = ToyRewardModel(problem.dim)
    expert.load_state_dict(torch.load(f'pretrained_models/reward_models/{args.problem}.pt'))
    expert.eval()

    @torch.no_grad()
    def ask_expert(xfx):
        """Useful so that we don't repeat torch.no_grad()"""
        return expert(xfx)

    # Gather an initial set of expert preferences
    train_x_pref = train_x.clone()
    train_y_pref = ask_expert(torch.cat([train_x_pref, train_y], dim=-1))  # (n_data, 1)

    # Scalarization (linear with weights)
    weights = torch.tensor([0.9, 0.1], dtype=torch.float32)
    # scalarized_y = torch.cat([train_y, train_y_pref], dim=-1) @ weights
    # best_scalarized_y = scalarized_y.max()

    # Create a surrogate to model expert preferences
    if args.method == 'la':
        def get_net_pref():
            return torch.nn.Sequential(
                nn.Linear(problem.dim, 50),  # input: (x, f(x))
                nn.ReLU(),
                nn.Linear(50, 50),
                nn.ReLU(),
                nn.Linear(50, 1)  # output: estimated scalar expert score on (x, f(x))
            )
        model_pref = LaplaceBoTorch(get_net_pref, train_x_pref, train_y_pref, noise_var=1e-4, batch_size=1024, backend=CurvlinopsGGN)
    elif args.method == 'gp':
        n_epochs = 500
        model_pref = MLLGP(train_x_pref, train_y_pref, n_epochs=n_epochs)

#################################################################################################
#                                 Bayesian Optimization Loop                                    #
#################################################################################################
best_y = train_y.min().item()
trace_best_y = []
pbar = tqdm.trange(args.exp_len)
pbar.set_description(
    f'[Best f(x) = {best_y:.3f}]'
)

# BO Loop
for i in pbar:
    if not args.with_expert:
        acqf = TSAcquisitionFunction(model, maximize=problem.is_maximize)
    else:
        acqf = ScalarizedTSWithExpert(
            ModelList(model, model_pref), maximize=problem.is_maximize,
            weights=weights
        )

    new_x, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=10, raw_samples=20)

    if len(new_x.shape) == 1:
        new_x = new_x.unsqueeze(0)  # (1, dim)

    # Evaluate the objective on the proposed x
    new_y = true_f(new_x).unsqueeze(-1)  # (q, 1)

    # Update posterior
    model = model.condition_on_observations(new_x, new_y)
    if args.with_expert:
        new_y_pref = ask_expert(torch.cat([new_x, new_y], dim=-1))
        model_pref.condition_on_observations(new_x, new_y_pref)

    # Update the current best y
    best_cand_y = new_y.max().item() if problem.is_maximize else new_y.min().item()
    if best_cand_y <= best_y:
        best_y = best_cand_y
    trace_best_y.append(best_y)

    if not args.with_expert:
        desc = f'[Best f(x) = {best_y:.3f}, curr f(x) = {best_cand_y:.3f}]'
    else:
        desc = f'[Best f(x) = {best_y:.3f}, curr f(x) = {best_cand_y:.3f}, curr expert(x, f(x)) = {new_y_pref.min():.3f}]'
    pbar.set_description(desc)

# # Save results
# path = f'results/toy/{args.problem}/{args.method}'
# if not os.path.exists(path):
#     os.makedirs(path)

# np.save(f'{path}/trace_best_y_{args.randseed}.npy', trace_best_y)
