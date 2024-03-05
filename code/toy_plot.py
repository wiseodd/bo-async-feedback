import torch
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
plt.style.use('bmh')
import utils.plot as plot_utils
import argparse
import os
import seaborn as sns
# sns.set_palette('colorblind')
# sns.set_style('whitegrid')
import pprint

import problems.toy as toy_problems

parser = argparse.ArgumentParser()
args = parser.parse_args()

PROBLEMS = ['ackley10', 'hartmann6', 'levy10', 'rastrigin10']
PROBLEM2TITLE = {
    'ackley10': r'Ackley-10',
    'hartmann6': r'Hartmann-6',
    'levy10': r'Lévy-10',
    'rastrigin10': r'Rastrigin-10',
}
METHODS_BASE = ['gp', 'la']
METHODS_PREF = ['la']
METHOD2LABEL = {
    'gp': 'GP',
    'la': 'LA',
    'la-pref': 'LA-Pref'
}
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
METHOD2COLOR = {k: v for k, v in zip(METHOD2LABEL.keys(), colors)}
RANDSEEDS = [1, 2, 3, 4, 5]


FIG_WIDTH = 1
FIG_HEIGHT = 0.2
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(
    FIG_WIDTH, FIG_HEIGHT,
    single_col=False
)
plt.rcParams.update(rc_params)

fig, axs = plt.subplots(1, len(PROBLEMS), sharex=True, constrained_layout=True)
fig.set_size_inches(fig_width, fig_height)

for i, (problem, ax) in enumerate(zip(PROBLEMS, axs.flatten())):
    problem_obj = toy_problems.PROBLEM_LIST[problem]()

    # Plot optimal val for f(x)
    best_val = problem_obj.get_function().optimal_value
    ax.axhline(best_val, c='k', ls='dashed', zorder=1000)

    # Plot base BO methods
    for method in METHODS_BASE:
        path = f'results/toy/{problem}/{method}'

        try:
            trace_best_y = np.stack([np.load(f'{path}/trace_best_y_{rs}.npy') for rs in RANDSEEDS])
        except FileNotFoundError:
            continue

        mean = np.mean(trace_best_y, axis=0)  # Over randseeds
        sem = st.sem(trace_best_y, axis=0)  # Over randseeds
        T = np.arange(len(mean)) + 1

        ax.plot(
            T, mean, color=METHOD2COLOR[method],
            label=f'{METHOD2LABEL[method]}'
        )
        ax.fill_between(
            T, mean-sem, mean+sem,
            color=METHOD2COLOR[method], alpha=0.25
        )

    # Plot BO methods with preferences
    for method_pref in METHODS_PREF:
        path = f'results/toy/{problem}-pref/{method}'

        try:
            trace_best_y = np.stack([np.load(f'{path}/trace_best_y_{rs}.npy') for rs in RANDSEEDS])
            trace_best_y_pref = np.stack([np.load(f'{path}/trace_best_y_pref_{rs}.npy') for rs in RANDSEEDS])
            trace_best_y_scal = np.stack([np.load(f'{path}/trace_best_y_scal_{rs}.npy') for rs in RANDSEEDS])
        except FileNotFoundError:
            continue

        # f(x)
        mean = np.mean(trace_best_y, axis=0)  # Over randseeds
        sem = st.sem(trace_best_y, axis=0)  # Over randseeds
        T = np.arange(len(mean)) + 1
        ax.plot(
            T, mean, color=METHOD2COLOR[f'{method}-pref'],
            label=METHOD2LABEL[f'{method}-pref']
        )
        ax.fill_between(
            T, mean-sem, mean+sem,
            color=METHOD2COLOR[f'{method}-pref'], alpha=0.25
        )

        # expert(x, f(x))
        mean = np.mean(trace_best_y_pref, axis=0)
        sem = st.sem(trace_best_y_pref, axis=0)
        ax.plot(
            T, mean, color=METHOD2COLOR[f'{method}-pref'],
            label=METHOD2LABEL[f'{method}-pref'], ls='dashed'
        )
        ax.fill_between(
            T, mean-sem, mean+sem,
            color=METHOD2COLOR[f'{method}-pref'], alpha=0.25
        )

    title = f'{PROBLEM2TITLE[problem]}'
    ax.set_title(title)
    ax.set_xlabel(r'$t$')

    if i == 0:  # Only the left-most
        ax.set_ylabel(r'$f(x_*)$')

    ax.set_xlim(0, len(mean))

    # handles, labels = ax.get_legend_handles_labels()
    if i == 0:
        ax.legend(
            loc='best', title='Methods'
        )

# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0, 1.065, 1, 0.005), mode='expand', ncols=8)

# Save results
path = f'../paper/figs'
if not os.path.exists(path):
    os.makedirs(path)

fname = f'toy_bo'
plt.savefig(f'{path}/{fname}.pdf')