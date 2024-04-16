import torch
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

plt.style.use("bmh")
import utils.plot as plot_utils
import argparse
import os
import seaborn as sns

# sns.set_palette('colorblind')
# sns.set_style('whitegrid')
import pprint

import problems.toy as toy_problems

parser = argparse.ArgumentParser()
parser.add_argument("--expert_prob", type=float, default=0.1)
parser.add_argument("--layout", default="aabi", choices=["aabi", "neurips"])
parser.add_argument("--no_title", default=False, action="store_true")
parser.add_argument("--no_xaxis", default=False, action="store_true")
parser.add_argument("--no_legend", default=False, action="store_true")
args = parser.parse_args()

if args.layout == "aabi":
    args.layout = "jmlr"

PROBLEMS = ["ackley10constrained", "levy10constrained"]
PROBLEM2TITLE = {
    "ackley10constrained": r"Ackley $d = 10$ With Conflicting Pref.",
    "levy10constrained": r"Levy $d = 10$ With Conflicting Pref.",
}
METHODS_BASE = [
    # 'gp',
    "la"
]
METHODS_PREF = [
    # 'gp',
    "la"
]
METHOD2LABEL = {
    "gp": "GP",
    "la": "LA",
    "gp-pref": "GP-Pref",
    "la-pref": "LA-Pref",
}
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
METHOD2COLOR = {k: v for k, v in zip(METHOD2LABEL.keys(), colors)}
RANDSEEDS = [1, 2, 3, 4, 5]


FIG_WIDTH = 1
FIG_HEIGHT = 0.15
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(
    FIG_WIDTH, FIG_HEIGHT, single_col=False, layout=args.layout
)
plt.rcParams.update(rc_params)

fig, axs = plt.subplots(1, len(PROBLEMS), sharex=True, constrained_layout=True)
fig.set_size_inches(fig_width, fig_height)

for i, (problem, ax) in enumerate(zip(PROBLEMS, axs.flatten())):
    # if problem not in ['ackley10constrained']:
    #     continue

    problem_obj = toy_problems.PROBLEM_LIST[problem]()

    # Plot optimal val for f(x)
    best_val = problem_obj.get_function().optimal_value
    ax.axhline(best_val, c="k", ls="dashed", zorder=1000)

    # Plot base BO methods
    for method in METHODS_BASE:
        path = f'results/toy/{problem.replace("constrained", "")}/{method}'

        trace_best_y = np.stack(
            [np.load(f"{path}/trace_best-y_rs{rs}.npy") for rs in RANDSEEDS]
        )

        mean = np.mean(trace_best_y, axis=0)  # Over randseeds
        sem = st.sem(trace_best_y, axis=0)  # Over randseeds
        T = np.arange(len(mean)) + 1

        ax.plot(
            T,
            mean,
            color=METHOD2COLOR[method],
            label=f"{METHOD2LABEL[method]}",
            linestyle="dashed",
        )
        ax.fill_between(
            T, mean - sem, mean + sem, color=METHOD2COLOR[method], alpha=0.25
        )

    # Plot BO methods with preferences
    for method_pref in METHODS_PREF:
        path = f"results/toy/{problem}-pref/random/{method_pref}"

        trace_best_y = np.stack(
            [
                np.load(
                    f"{path}/trace_best-y_gamma1.0_prob{args.expert_prob}_rs{rs}.npy"
                )
                for rs in RANDSEEDS
            ]
        )
        trace_best_y_pref = np.stack(
            [
                np.load(
                    f"{path}/trace_best-r_gamma1.0_prob{args.expert_prob}_rs{rs}.npy"
                )
                for rs in RANDSEEDS
            ]
        )
        trace_best_y_scal = np.stack(
            [
                np.load(
                    f"{path}/trace_best-scal-y_gamma1.0_prob{args.expert_prob}_rs{rs}.npy"
                )
                for rs in RANDSEEDS
            ]
        )
        trace_norm_x = np.stack(
            [
                np.load(
                    f"{path}/trace_best-x_gamma1.0_prob{args.expert_prob}_rs{rs}.npy"
                )
                for rs in RANDSEEDS
            ]
        )
        trace_norm_x = np.linalg.norm(trace_norm_x.squeeze(), axis=-1)

        # f(x)
        mean = np.mean(trace_best_y, axis=0)  # Over randseeds
        sem = st.sem(trace_best_y, axis=0)  # Over randseeds
        T = np.arange(len(mean)) + 1
        ax.plot(
            T,
            mean,
            color=METHOD2COLOR[f"{method_pref}"],
            label=METHOD2LABEL[f"{method_pref}-pref"],
            linestyle="solid",
        )
        ax.fill_between(
            T, mean - sem, mean + sem, color=METHOD2COLOR[f"{method_pref}"], alpha=0.25
        )

        # norm(x)
        mean = np.mean(trace_norm_x, axis=0)  # Over randseeds
        sem = st.sem(trace_norm_x, axis=0)  # Over randseeds
        T = np.arange(len(mean)) + 1
        ax.plot(T, mean, label=r"$\Vert x_* \Vert_2$", linestyle="dotted", color="gray")
        ax.fill_between(T, mean - sem, mean + sem, color="gray", alpha=0.25)

    title = f"{PROBLEM2TITLE[problem]}"

    if not args.no_title:
        ax.set_title(title)

    if not args.no_xaxis:
        ax.set_xlabel(r"$t$")

    # if args.no_xaxis:
    #     ax.set_xticks([])

    if i == 0:  # Only the left-most
        ax.set_ylabel(r"$f(x_*)$")

    ax.set_xlim(0, 250)

    # handles, labels = ax.get_legend_handles_labels()
    if i == 0 and not args.no_legend:
        ax.legend(loc="best", title="Methods")

# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0, 1.065, 1, 0.005), mode='expand', ncols=8)

# Save results
if args.layout == "neurips":
    path = f"../paper/figs"
else:
    path = f"../paper/aabi/figs_aabi"

if not os.path.exists(path):
    os.makedirs(path)

fname = f"toy_bo_constrained_prob{args.expert_prob}"
plt.savefig(f"{path}/{fname}.pdf")
