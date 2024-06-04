import utils.plot as plot_utils
import argparse
import os
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import problems.chem as chem_problems


plt.style.use("bmh")

parser = argparse.ArgumentParser()
parser.add_argument("--expert-prob", type=float, default=0.1)
parser.add_argument("--layout", default="aabi", choices=["aabi", "neurips"])
parser.add_argument("--no-title", default=False, action="store_true")
parser.add_argument("--no-xaxis", default=False, action="store_true")
parser.add_argument("--no-legend", default=False, action="store_true")
args = parser.parse_args()

if args.layout == "aabi":
    args.layout = "jmlr"

PROBLEMS = ["kinase", "ampc", "d4"]
PROBLEM2TITLE = {
    "kinase": r"Kinase",
    "ampc": r"AmpC",
    "d4": r"D4",
}
PROBLEM2YLIM = {
    "kinase": (-9.95, -8.75),
    "ampc": (-85, -40),
    "d4": (-67, -35),
}
METHODS_BASE = [
    "gp",
    # "la",
]
METHODS_PREF = [
    "gp",
    # "la",
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
FIG_HEIGHT = 0.175
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(
    FIG_WIDTH, FIG_HEIGHT, single_col=False, layout=args.layout
)
plt.rcParams.update(rc_params)

fig, axs = plt.subplots(1, len(PROBLEMS), sharex=True, constrained_layout=True)
fig.set_size_inches(fig_width, fig_height)

for i, (problem, ax) in enumerate(zip(PROBLEMS, axs.flatten())):
    # Get optimal val for f(x)
    problem_obj = chem_problems.PROBLEM_LIST[problem](feature_type="fingerprints")
    best_val = (
        max(problem_obj.cand_objs)
        if problem_obj.is_maximize
        else min(problem_obj.cand_objs)
    )

    # Plot optimal val for f(x)
    ax.axhline(best_val, c="k", ls="dashed", zorder=1000)

    # Plot base BO methods
    for method in METHODS_BASE:
        path = f"results/chem/{problem}/random/{method}"

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
        path = f"results/chem/{problem}-pref/random/{method_pref}"

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

    title = f"{PROBLEM2TITLE[problem]}"

    if not args.no_title:
        ax.set_title(title)

    if not args.no_xaxis:
        ax.set_xlabel(r"$t$")

    if i == 0:  # Only the left-most
        ax.set_ylabel(r"Docking Score $(\downarrow)$")

    ax.set_xlim(0, 250)
    ax.set_ylim(*PROBLEM2YLIM[problem])

    if i == 0 and not args.no_legend:
        ax.legend(loc="best", title="Methods")

# Save results
if args.layout == "neurips":
    path = "../paper/figs"
else:
    path = "../paper/figs_aabi"

if not os.path.exists(path):
    os.makedirs(path)

fname = f"chem_bo_prob{args.expert_prob}"
plt.savefig(f"{path}/{fname}.pdf")
