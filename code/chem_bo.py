import numpy as np
import torch
import tqdm

from laplace.curvature import CurvlinopsGGN, AsdlGGN
from laplace_bayesopt.botorch import LaplaceBoTorch
from laplace_bayesopt.acqf import ThompsonSampling

from botorch.acquisition.analytic import ExpectedImprovement
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from gpytorch.kernels import ScaleKernel

import problems.chem as chem_probs
from models.surrogate import MLLGP, MLP
from models.surrogate_pref import PrefLaplaceBoTorch
from models.reward import RewardModel
from models.acqf import (
    ThompsonSamplingRewardDiff,
    ThompsonSamplingWithExpertPref,
    BALDForRewardModel,
)
from utils import helpers

import argparse
import os
import sys

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--problem",
    default="kinase",
    choices=["kinase", "ampc", "d4"],
)
parser.add_argument("--feats", default="fingerprints", choices=["fingerprints"])
parser.add_argument("--method", default="la", choices=["la", "gp"])
parser.add_argument("--exp-len", type=int, default=250)
parser.add_argument("--acqf", default="ts", choices=["ts", "ei"])
parser.add_argument(
    "--acqf_pref",
    default="random",
    choices=["random", "active_bald", "active_large_diff", "active_small_diff"],
)
parser.add_argument("--with-expert", default=False, action="store_true")
parser.add_argument("--expert-gamma", type=float, default=1.0)
parser.add_argument("--expert-prob", type=float, default=0.1)
parser.add_argument("--verbose", default=False, action="store_true")
parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
parser.add_argument("--randseed", type=int, default=1)
args = parser.parse_args()

if args.with_expert and args.acqf == "ei":
    print("Thompson sampling is a must for expert feedback")
    sys.exit(1)

assert 0 <= args.expert_prob <= 1

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)

#######################################################################################
#                         Standard Bayesian Optimization Preparation                  #
#######################################################################################
problem = chem_probs.PROBLEM_LIST[args.problem](args.feats)

# Initialize training data by uniform sampling
train_smiles, train_X, train_Y = [], [], []

while len(train_smiles) < 10:
    rand_idx = np.random.randint(len(problem))
    smiles, feats, obj = problem.pop_candidate(rand_idx)
    train_smiles.append(smiles)
    train_X.append(feats)
    train_Y.append(obj)

train_X = torch.stack(train_X)
train_Y = torch.stack(train_Y)

# Initialize surrogate models
if args.method == "la":

    def get_net():
        return MLP(problem.dim, normalize_output=True)

    model = LaplaceBoTorch(
        get_net,
        train_X,
        train_Y,
        noise_var=1e-4,
        batch_size=1024,
        backend=CurvlinopsGGN,
        device=args.device,
    )
elif args.method == "gp":
    model = MLLGP(
        train_X,
        train_Y,
        kernel=ScaleKernel(TanimotoKernel()),
        lr=1e-3,
        n_epochs=500,
    )


#######################################################################################
#                              Expert Feedback Modeling                               #
#######################################################################################
if args.with_expert:
    # Gather initial preference dataset
    train_pref, train_pair_idxs = helpers.sample_pref_data(
        train_X,
        problem.get_preference,
        10,
        output_indices=True,
        source_smiles=train_smiles,
    )
    train_pair_idxs = list(map(tuple, train_pair_idxs))  # list of 2-tuples

    # Surrogate to model expert preferences
    model_pref = PrefLaplaceBoTorch(
        lambda: RewardModel(dim=problem.dim),
        train_pref,
        noise_var=1e-2,
        batch_size=1024,
        backend=AsdlGGN,
        device=args.device,
        enable_backprop=True,
    )

    # Scalarization to take into account both f(x) and expert preferences
    def scalarize(y, r):
        return (y if problem.is_maximize else -y) + args.expert_gamma * r


#######################################################################################
#                               Bayesian Optimization Loop                            #
#######################################################################################
best_idx = train_Y.argmax().item() if problem.is_maximize else train_Y.argmin().item()
best_x = train_X[best_idx].unsqueeze(0)
best_y = train_Y[best_idx].item()
trace_best_y = []
trace_best_x = []

if args.with_expert:
    best_r = model_pref.posterior(best_x).mean.squeeze().item()
    best_scal_y = scalarize(
        helpers.y_transform(best_y, model.orig_train_Y).item(), best_r
    )
    trace_best_r = []
    trace_best_scal_y = []

print()
print(
    f"Problem: {args.problem}, method: {args.method}, with expert: {args.with_expert}, "
    + f"gamma: {args.expert_gamma}, prob: {args.expert_prob}, "
    + f"acqf_pref: {args.acqf_pref}, randseed: {args.randseed}"
)
print(
    "----------------------------------------------------------------------------------"
)

pbar = tqdm.trange(args.exp_len)
pbar.set_description(f"[Best f(x) = {best_y:.3f}]")

# BO Loop
for i in pbar:
    if not args.with_expert:
        if args.acqf == "ts":
            acqf = ThompsonSampling(model, maximize=problem.is_maximize).to(args.device)
        else:
            acqf = ExpectedImprovement(
                model, best_f=best_y, maximize=problem.is_maximize
            ).to(args.device)
    else:
        acqf = ThompsonSamplingWithExpertPref(
            model=model, model_pref=model_pref, maximize=False, gamma=args.expert_gamma
        ).to(args.device)

    acq_vals = torch.cat([acqf(x) for x, _ in problem.get_dataloader()], dim=0)
    best_acq_val, best_acq_idx = acq_vals.max(dim=0)

    new_smiles, new_x, new_y = problem.pop_candidate(best_acq_idx)
    train_smiles.append(new_smiles)

    if len(new_x.shape) == 1:
        new_x = new_x.unsqueeze(0)  # (1, dim_x)
    if len(new_y.shape) == 1:
        new_y = new_y.unsqueeze(0)  # (1, dim_y)

    # Update vanilla BO posterior
    model = model.condition_on_observations(new_x, new_y)

    if not args.with_expert:
        truth = (
            (new_y.item() > best_y) if problem.is_maximize else (new_y.item() < best_y)
        )
        if truth:
            best_y = new_y.item()
            best_x = new_x
        trace_best_y.append(best_y)

        desc = f"[Best f(x) = {best_y:.3f}, curr f(x) = {new_y.item():.3f}]"
    else:
        # `expert_random_prob` of the time, sample preference data
        # and update the expert surrogate
        if np.random.rand() <= args.expert_prob:
            N_NEW_PREF_DATA = 3
            if args.acqf_pref == "random":
                new_train_pref = helpers.sample_pref_data(
                    model.orig_train_X,
                    problem.get_preference,
                    N_NEW_PREF_DATA,
                    source_smiles=train_smiles,
                )
            elif "active" in args.acqf_pref:
                # Randomly sample pairs to make the cost constant wrt. the size of X.
                # Because, the num. of ordered pairs of X with size N is quadratic.
                rand_train_pref, rand_idxs = helpers.sample_pref_data(
                    model.orig_train_X,
                    problem.get_preference,
                    2000,
                    exclude_indices=train_pair_idxs,
                    output_indices=True,
                    source_smiles=train_smiles,
                )

                # Get the active learning acqf. vals.
                with torch.no_grad():
                    if args.acqf_pref == "active_bald":
                        acqf = BALDForRewardModel(model_pref)
                    elif (
                        args.acqf_pref == "active_large_diff"
                        or args.acqf_pref == "active_small_diff"
                    ):
                        acqf = ThompsonSamplingRewardDiff(model_pref)

                    acq_vals = acqf(rand_train_pref)
                    acq_vals *= -1 if args.acqf_pref == "active_small_diff" else 1

                # Get the top N_NEW_PREF_DATA
                topk_idxs = torch.topk(acq_vals, k=N_NEW_PREF_DATA)[1]
                new_train_pref = helpers.subset_pref_data(rand_train_pref, topk_idxs)

                # Track the pair indices of the preference training data
                train_pair_idxs += list(map(tuple, np.array(rand_idxs)[topk_idxs]))

            model_pref = model_pref.condition_on_observations(new_train_pref)

        # Compute the rewards of the previous best x and the proposed x using
        # the updated pref model
        with torch.no_grad():
            ins = torch.cat([best_x, new_x], dim=0)  # (2, dim)
            out = model_pref.posterior(ins).mean.squeeze()  # (2,)
            _best_r, new_r = out[0].item(), out[1].item()

        # Use the same scale
        scal_y_old = scalarize(
            helpers.y_transform(best_y, model.orig_train_Y).item(), _best_r
        )
        scal_y_new = scalarize(
            helpers.y_transform(new_y, model.orig_train_Y).item(), new_r
        )
        if scal_y_new > scal_y_old:
            best_x = new_x
            best_y = new_y.item()
            best_r = new_r
            best_scal_y = scal_y_new

            if args.verbose:
                print("New x!", best_x.squeeze().numpy())
                print()

        trace_best_y.append(best_y)
        trace_best_r.append(best_r)
        trace_best_scal_y.append(best_scal_y)

        desc = (
            f"[f(x*): {best_y:.2f}, r(x*): {best_r:.2f}, scal_y*: {best_scal_y:.2f},"
            + f" f(x): {new_y.item():.2f}, r(x): {new_r:.2f}, scal_y: {scal_y_new:.2f}]"
        )

    pbar.set_description(desc)

    # Also save the current best x
    trace_best_x.append(best_x)

# Save results
problem_name = args.problem + ("-pref" if args.with_expert else "")
path = f"results/chem/{problem_name}/{args.acqf_pref}/{args.method}"
if not os.path.exists(path):
    os.makedirs(path)

if not args.with_expert:
    np.save(f"{path}/trace_best-x_rs{args.randseed}.npy", trace_best_x)

    if args.acqf == "ts":
        np.save(f"{path}/trace_best-y_rs{args.randseed}.npy", trace_best_y)
    elif args.acqf == "ei":
        np.save(f"{path}/trace_best-y_ei_rs{args.randseed}.npy", trace_best_y)
else:

    def args_to_name():
        name = f"gamma{args.expert_gamma}_prob{args.expert_prob}"
        return name

    np.save(f"{path}/trace_best-x_{args_to_name()}_rs{args.randseed}.npy", trace_best_x)
    np.save(f"{path}/trace_best-y_{args_to_name()}_rs{args.randseed}.npy", trace_best_y)
    np.save(f"{path}/trace_best-r_{args_to_name()}_rs{args.randseed}.npy", trace_best_r)
    np.save(
        f"{path}/trace_best-scal-y_{args_to_name()}_rs{args.randseed}.npy",
        trace_best_scal_y,
    )
    np.save(
        f"{path}/best-x_{args_to_name()}_rs{args.randseed}.npy",
        best_x.squeeze().numpy(),
    )
