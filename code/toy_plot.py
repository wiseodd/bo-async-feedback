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
    'levy10': r'Levy-10',
    'rastrigin10': r'Rastrigin-10',
}
METHOD2LABEL = {
    # 'gp': 'GP',
    'la': 'LA'
}
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
    if problem is None:
        continue

    problem_obj = toy_problems.PROBLEM_LIST[problem]()

    # Plot optimal val
    best_val = problem_obj.get_function().optimal_value
    ax.axhline(best_val, c='k', ls='dashed', zorder=1000)

    # Plot methods
    for method in METHOD2LABEL.keys():
        path = f'results/toy/{problem}/{method}'
        trace_best_y = np.stack([np.load(f'{path}/trace_best_y_{rs}.npy') for rs in RANDSEEDS])
        mean = np.mean(trace_best_y, axis=0)  # Over randseeds
        sem = st.sem(trace_best_y, axis=0)  # Over randseeds
        T = np.arange(len(mean)) + 1

        ax.plot(T, mean, label=f'{METHOD2LABEL[method]}')
        ax.fill_between(T, mean-sem, mean+sem, alpha=0.25)

    title = f'{PROBLEM2TITLE[problem]}'
    ax.set_title(title)
    ax.set_xlabel(r'$t$')

    if i == 0:  # Only the left-most
        ax.set_ylabel(r'$f(x_*)$')

    ax.set_xlim(0, len(mean))

    # handles, labels = ax.get_legend_handles_labels()
    if i == 0:
        ax.legend(
            loc='upper right', ncols=1, frameon=True,
            framealpha=1, fancybox=False, facecolor='white',
            edgecolor='black', title='Methods'
        )

# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0, 1.065, 1, 0.005), mode='expand', ncols=8)

# Save results
path = f'../paper/figs'
if not os.path.exists(path):
    os.makedirs(path)

fname = f'toy_bo'
plt.savefig(f'{path}/{fname}.pdf')
