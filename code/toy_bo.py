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
parser.add_argument('--randseed', type=int, default=1)
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

    # Scalarization weights (linear with weights)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float32)
    # BoTorch is maximization by default; so update accordingly (preference model is always max)
    weights[0] *= 1 if problem.is_maximize else -1

    def scalarize(y, y_pref):
        y_combined = torch.stack([y, y_pref])  # (2, n, dim)
        return torch.einsum('mno,m->no', y_combined, weights).squeeze()

    # Identify the current best
    train_y_scal = scalarize(train_y, train_y_pref)
    best_idx_scal = train_y_scal.argmax()
    best_y_scal = train_y_scal[best_idx_scal].item()
    best_y = train_y[best_idx_scal].item()
    best_y_pref = train_y_pref[best_idx_scal].item()

    # To store preference maximization results
    trace_best_y_pref = []
    trace_best_y_scal = []

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
best_y = train_y.min().item() if not args.with_expert else best_y_scal
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
            # We always maximize the expert preference model
            ModelList(model, model_pref), weights=weights
        )

    new_x, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=10, raw_samples=20)

    if len(new_x.shape) == 1:
        new_x = new_x.unsqueeze(0)  # (1, dim)

    # Evaluate the objective on the proposed x
    new_y = true_f(new_x).unsqueeze(-1)  # (q, 1)

    # Update vanilla BO posterior
    model = model.condition_on_observations(new_x, new_y)

    if not args.with_expert:
        new_y_val = new_y.item()
        truth = (new_y_val > best_y) if problem.is_maximize else (new_y_val < best_y)
        if truth:
            best_y = new_y.item()
        trace_best_y.append(best_y)
        desc = f'[Best f(x) = {best_y:.3f}, curr f(x) = {new_y_val:.3f}]'
    else:
        # Update the posterior for preference model
        new_y_pref = ask_expert(torch.cat([new_x, new_y], dim=-1))
        model_pref.condition_on_observations(new_x, new_y_pref)

        # Update when there is a Pareto improvement
        # (The max of scalarization is in the Pareto frontier)
        new_y_scal = scalarize(new_y, new_y_pref).item()
        if new_y_scal > best_y_scal:
            best_y_scal = new_y_scal
            best_y = new_y.item()
            best_y_pref = new_y_pref.item()

        trace_best_y_scal.append(best_y_scal)
        trace_best_y.append(best_y)
        trace_best_y_pref.append(best_y_pref)

        desc = f'[Best scal\'d obj = {best_y_scal:.3f}; Best f(x) = {best_y:.3f}, best expert(x, f(x)) = {best_y_pref:.3f}]'

    pbar.set_description(desc)

# Save results
problem_name = args.problem + ('-pref' if args.with_expert else '')
path = f'results/toy/{problem_name}/{args.method}'
if not os.path.exists(path):
    os.makedirs(path)

# Vanilla BO
np.save(f'{path}/trace_best_y_{args.randseed}.npy', trace_best_y)
# Expert preference maximization results
if args.with_expert:
    np.save(f'{path}/trace_best_y_scal_{args.randseed}.npy', trace_best_y_scal)
    np.save(f'{path}/trace_best_y_pref_{args.randseed}.npy', trace_best_y_pref)
