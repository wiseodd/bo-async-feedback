import numpy as np
import torch
from torch import nn
from torch import distributions as dists
import tqdm

from botorch.optim.optimize import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.generation.gen import gen_candidates_torch

from laplace.curvature import AsdlGGN
from laplace_bayesopt.botorch import LaplaceBoTorch
from laplace_bayesopt.acqf import ThompsonSampling

import problems.toy as toy_problems
from models.surrogate import MLLGP, MLP
from models.surrogate_pref import PrefLaplaceBoTorch
from models.reward import ToyRewardModel
from models.acqf import ThompsonSamplingWithExpertPref
from utils import helpers

from collections import UserDict
import argparse, os

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser()
parser.add_argument('--problem', default='ackley10', choices=['levy10', 'ackley2', 'ackley10', 'hartmann6', 'rastrigin10'])
parser.add_argument('--method', default='la', choices=['la', 'gp'])
parser.add_argument('--exp_len', type=int, default=250)
parser.add_argument('--with_expert', default=False, action='store_true')
parser.add_argument('--expert_gamma', type=float, default=1.)
parser.add_argument('--expert_prob', type=float, default=0.25)
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
parser.add_argument('--randseed', type=int, default=1)
args = parser.parse_args()

assert 0 <= args.expert_prob <= 1

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
    train_pref = helpers.sample_pref_data(model.orig_train_X, problem.get_preference, 20)

    # Surrogate to model expert preferences
    model_pref = PrefLaplaceBoTorch(
        lambda: ToyRewardModel(dim=problem.dim), train_pref,
        noise_var=1e-2, batch_size=1024, backend=AsdlGGN, device=args.device,
        enable_backprop=True
    )

    # Scalarization to take into account both f(x) and expert preferences
    def scalarize(y, r):
        return (y if problem.is_maximize else -y) + args.expert_gamma * r


#################################################################################################
#                                 Bayesian Optimization Loop                                    #
#################################################################################################
best_x = train_x[train_y.argmin().item()].unsqueeze(0)
best_y = train_y.min().item()
trace_best_y = []

if args.with_expert:
    best_r = model_pref.posterior(best_x).mean.squeeze().item()
    best_scal_y = scalarize(helpers.y_transform(best_y, model.orig_train_Y).item(), best_r)
    trace_best_r = []
    trace_best_scal_y = []

print()
print(f'Problem: {args.problem}, method: {args.method}, with expert: {args.with_expert}, gamma: {args.expert_gamma}, prob: {args.expert_prob}, randseed: {args.randseed}')
print('--------------------------------------------------------------------------------')

pbar = tqdm.trange(args.exp_len)
pbar.set_description(
    f'[Best f(x) = {best_y:.3f}]'
)

# BO Loop
for i in pbar:
    if not args.with_expert:
        acqf = ThompsonSampling(model, maximize=problem.is_maximize).to(args.device)
    else:
        acqf = ThompsonSamplingWithExpertPref(
            model=model, model_pref=model_pref, maximize=False, gamma=args.expert_gamma
        ).to(args.device)

    new_x, _ = optimize_acqf(
        acqf, bounds=bounds, q=1, num_restarts=10, raw_samples=20,
    )

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
        # expert_random_prob of the time, sample preference data
        # and update the expert surrogate
        if np.random.rand() <= args.expert_prob:
            # TODO more intelligent sampling, e.g. via active learning
            new_train_pref = helpers.sample_pref_data(model.orig_train_X, problem.get_preference, 3)
            model_pref = model_pref.condition_on_observations(new_train_pref)

        with torch.no_grad():
            ins = torch.cat([best_x, new_x], dim=0)  # (2, dim)
            out = model_pref.posterior(ins).mean.squeeze()  # (2,)
            _best_r, new_r = out[0].item(), out[1].item()

        # Use the same scale
        scal_y_old = scalarize(helpers.y_transform(best_y, model.orig_train_Y).item(), _best_r)
        scal_y_new = scalarize(helpers.y_transform(new_y, model.orig_train_Y).item(), new_r)
        if scal_y_new > scal_y_old:
            best_x = new_x
            best_y = new_y.item()
            best_r = new_r
            best_scal_y = scal_y_new

            if args.verbose:
                print('New x!', best_x.squeeze().numpy())
                print()

        trace_best_y.append(best_y)
        trace_best_r.append(best_r)
        trace_best_scal_y.append(best_scal_y)

        desc = f'[f(x*): {best_y:.2f}, r(x*): {best_r:.2f}, scal_y*: {best_scal_y:.2f}, f(x): {new_y.item():.2f}, r(x): {new_r:.2f}, scal_y: {scal_y_new:.2f}]'

    pbar.set_description(desc)

# Save results
problem_name = args.problem + ('-pref' if args.with_expert else '')
path = f'results/toy/{problem_name}/{args.method}'
if not os.path.exists(path):
    os.makedirs(path)

if not args.with_expert:
    np.save(f'{path}/trace-best-y_{args.randseed}.npy', trace_best_y)
else:
    print(best_x.squeeze().numpy())
    np.save(f'{path}/trace_best-y_gamma{args.expert_gamma}_prob{args.expert_prob}_rs{args.randseed}.npy', trace_best_y)
    np.save(f'{path}/trace_best-r_gamma{args.expert_gamma}_prob{args.expert_prob}_rs{args.randseed}.npy', trace_best_r)
    np.save(f'{path}/trace_best-scal-y_gamma{args.expert_gamma}_prob{args.expert_prob}_rs{args.randseed}.npy', trace_best_scal_y)
    np.save(f'{path}/best-x_gamma{args.expert_gamma}_prob{args.expert_prob}_rs{args.randseed}.npy', best_x.squeeze().numpy())
