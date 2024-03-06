import numpy as np
import torch
from torch import nn
from torch import distributions as dists
import tqdm

from botorch.optim.optimize import optimize_acqf
from botorch.models.transforms.outcome import Standardize

from laplace.curvature import AsdlGGN
from laplace_bayesopt.botorch import LaplaceBoTorch
from laplace_bayesopt.acqf import TSAcquisitionFunction

import problems.toy as toy_problems
from models.surrogate import MLLGP, MLP
from models.surrogate_pref import PrefLaplaceBoTorch
from models.reward import ToyRewardModel
from models.acqf import TSWithExpertPref
from utils import helpers

from collections import UserDict
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--problem', default='ackley10', choices=['levy10', 'ackley2', 'ackley10', 'hartmann6', 'rastrigin10'])
parser.add_argument('--method', default='la', choices=['la', 'gp'])
parser.add_argument('--with_expert', default=False, action='store_true')
parser.add_argument('--exp_len', type=int, default=500)
parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
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
train_y = true_f(train_x).reshape(-1, 1).to(args.device)

if args.method == 'la':
    def get_net():
        return MLP(problem.dim, normalize_output=True)
    model = LaplaceBoTorch(
        get_net, train_x, train_y, noise_var=1e-4, batch_size=1024,
        backend=AsdlGGN, device=args.device
    )
elif args.method == 'gp':
    # Hotfix for a numerical issue (not PSD error)
    n_epochs = (0 if args.problem == 'hartmann6' else 500)
    model = MLLGP(train_x, train_y, n_epochs=n_epochs)


#################################################################################################
#                                  Expert Feedback Modeling                                     #
#################################################################################################
if args.with_expert:
    # Gather initial preference dataset
    train_pref = helpers.sample_pref_data(train_x, problem.get_preference, 20)

    # Surrogate to model expert preferences
    model_pref = PrefLaplaceBoTorch(
        lambda: ToyRewardModel(dim=problem.dim), train_pref,
        noise_var=1e-4, batch_size=1024, backend=AsdlGGN, device=args.device,
        enable_backprop=False
    )

    # To make sure f(x) has the same scale as the preference model's output
    def y_transform(y):
        trf = Standardize(1)
        trf.train()
        trf(model.orig_train_Y)  # Fit
        trf.eval()
        return trf(y)[0]

    # Scalarization to take into account both f(x) and expert preferences
    def scalarize(y, r):
        return 0.5 * (y if problem.is_maximize else -y) + 0.5* r


#################################################################################################
#                                 Bayesian Optimization Loop                                    #
#################################################################################################
best_x = train_x[train_y.argmin().item()].unsqueeze(0)
best_y = train_y.min().item()
trace_best_y = []

if args.with_expert:
    best_r = model_pref.posterior(best_x.unsqueeze(1)).mean.squeeze().item()
    best_scal_y = scalarize(y_transform(best_y).item(), best_r)
    trace_best_r = []
    trace_best_scal_y = []

pbar = tqdm.trange(args.exp_len)
pbar.set_description(
    f'[Best f(x) = {best_y:.3f}]'
)

# BO Loop
for i in pbar:
    if not args.with_expert:
        acqf = TSAcquisitionFunction(model, maximize=problem.is_maximize).to(args.device)
    else:
        acqf = TSWithExpertPref(
            model=model, model_pref=model_pref, maximize=False
        ).to(args.device)

    new_x, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=10, raw_samples=20)

    if len(new_x.shape) == 1:
        new_x = new_x.unsqueeze(0)  # (1, dim)

    # Evaluate the objective on the proposed x
    new_y = true_f(new_x).unsqueeze(-1)  # (q, 1)

    # Update vanilla BO posterior
    model = model.condition_on_observations(new_x, new_y)

    if not args.with_expert:
        truth = (new_y.item() > best_y) if problem.is_maximize else (new_y.item() < best_y)
        if truth:
            best_y = new_y.item()
        trace_best_y.append(best_y)

        desc = f'[Best f(x) = {best_y:.3f}, curr f(x) = {new_y.item():.3f}]'
    else:
        # TODO update reward model; with schedule
        new_train_pref = helpers.sample_pref_data(model.train_X, problem.get_preference, 1)
        model_pref = model_pref.condition_on_observations(new_train_pref)

        with torch.no_grad():
            # inputs = torch.cat([best_x, new_x], dim=0).unsqueeze(1)
            out = model_pref.posterior(new_x.unsqueeze(1)).mean.squeeze()  # (1,)
            new_r = out.item()

        # Use the same scale
        scal_y_new = scalarize(y_transform(new_y).item(), new_r)
        if scal_y_new > best_scal_y:
            best_x = new_x
            best_y = new_y.item()
            best_r = new_r
            best_scal_y = scal_y_new

        trace_best_y.append(best_y)
        trace_best_r.append(best_r)
        trace_best_scal_y.append(best_scal_y)

        desc = f'[f(x*): {best_y:.3f}, r(x*): {best_r:.3f}, scal_y*: {best_scal_y:.3f}, f(x): {new_y.item():.3f}, r(x): {new_r:.3f}]'

    pbar.set_description(desc)

# Save results
problem_name = args.problem + ('-pref' if args.with_expert else '')
path = f'results/toy/{problem_name}/{args.method}'
if not os.path.exists(path):
    os.makedirs(path)

np.save(f'{path}/trace_best_y_{args.randseed}.npy', trace_best_y)

if args.with_expert:
    np.save(f'{path}/trace_best_r_{args.randseed}.npy', trace_best_r)
    np.save(f'{path}/trace_best_scal_y_{args.randseed}.npy', trace_best_scal_y)
    np.save(f'{path}/best_x_{args.randseed}.npy', best_x.squeeze().numpy())
