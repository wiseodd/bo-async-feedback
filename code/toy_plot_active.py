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
parser.add_argument('--expert_prob', type=float, default=0.1)
parser.add_argument('--layout', default='aabi', choices=['aabi', 'neurips'])
parser.add_argument('--al_acqf', default='active_bald', choices=['active_bald', 'active_large_diff', 'active_small_diff'])
parser.add_argument('--no_legend', default=False, action='store_true')
args = parser.parse_args()

if args.layout == 'aabi':
    args.layout = 'jmlr'

PROBLEMS = [
    'ackley10',
    'levy10',
    'rastrigin10',
    'hartmann6'
]
PROBLEM2TITLE = {
    'ackley10': r'Ackley $d = 10$ ($\downarrow$)',
    'hartmann6': r'Hartmann $d = 6$ ($\downarrow$)',
    'levy10': r'Levy $d = 10$ ($\downarrow$)',
    'rastrigin10': r'Rastrigin $d = 10$ ($\downarrow$)',
}
METHODS_PREF = [
    'gp',
    'la'
]
AL_ACQFS = [
    'random',
    args.al_acqf
]
ACQF2NAME = {
    'active_bald': 'BALD',
    'active_large_diff': 'LDiff',
    'active_small_diff': 'SDiff'
}
METHOD2LABEL = {
    'gp-random': 'GP-Rand',
    'la-random': 'LA-Rand',
    f'gp-{args.al_acqf}': f'GP-{ACQF2NAME[args.al_acqf]}',
    f'la-{args.al_acqf}': f'LA-{ACQF2NAME[args.al_acqf]}',
}
ALACQF2LINESTYLE = {
    'random': 'solid',
    'active_bald': 'solid',
    'active_large_diff': 'solid',
    'active_small_diff': 'solid',
}
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
METHOD2COLOR = {k: v for k, v in zip(METHOD2LABEL.keys(), colors)}
RANDSEEDS = [1, 2, 3, 4, 5]


FIG_WIDTH = 1
FIG_HEIGHT = 0.2
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(
    FIG_WIDTH, FIG_HEIGHT,
    single_col=False,
    layout=args.layout
)
plt.rcParams.update(rc_params)

fig, axs = plt.subplots(1, len(PROBLEMS), sharex=True, constrained_layout=True)
fig.set_size_inches(fig_width, fig_height)

for i, (problem, ax) in enumerate(zip(PROBLEMS, axs.flatten())):
    problem_obj = toy_problems.PROBLEM_LIST[problem]()

    # Plot optimal val for f(x)
    best_val = problem_obj.get_function().optimal_value
    ax.axhline(best_val, c='k', ls='dashed', zorder=1000)

    # Plot BO methods with preferences
    if problem not in [
        'ackley10',
        'levy10',
        'rastrigin10',
        'hartmann6'
    ]:
        continue

    for method_pref in METHODS_PREF:
        for al_acqf in AL_ACQFS:
            path = f'results/toy/{problem}-pref/{al_acqf}/{method_pref}'

            trace_best_y = np.stack([np.load(f'{path}/trace_best-y_gamma1.0_prob{args.expert_prob}_rs{rs}.npy') for rs in RANDSEEDS])
            trace_best_y_pref = np.stack([np.load(f'{path}/trace_best-r_gamma1.0_prob{args.expert_prob}_rs{rs}.npy') for rs in RANDSEEDS])
            trace_best_y_scal = np.stack([np.load(f'{path}/trace_best-scal-y_gamma1.0_prob{args.expert_prob}_rs{rs}.npy') for rs in RANDSEEDS])

            # f(x)
            mean = np.mean(trace_best_y, axis=0)  # Over randseeds
            sem = st.sem(trace_best_y, axis=0)  # Over randseeds
            T = np.arange(len(mean)) + 1
            ax.plot(
                T, mean, color=METHOD2COLOR[f'{method_pref}-{al_acqf}'],
                label=METHOD2LABEL[f'{method_pref}-{al_acqf}'],
                linestyle=ALACQF2LINESTYLE[al_acqf]
            )
            ax.fill_between(
                T, mean-sem, mean+sem,
                color=METHOD2COLOR[f'{method_pref}-{al_acqf}'], alpha=0.2
            )

    title = f'{PROBLEM2TITLE[problem]}'
    ax.set_title(title)
    ax.set_xlabel(r'$t$')

    if i == 0:  # Only the left-most
        ax.set_ylabel(r'$f(x_*)$')

    ax.set_xlim(0, 250)

    # handles, labels = ax.get_legend_handles_labels()
    if i == 0 and not args.no_legend:
        ax.legend(
            loc='best', title='Methods'
        )

# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0, 1.065, 1, 0.005), mode='expand', ncols=8)

# Save results
if args.layout == 'neurips':
    path = f'../paper/figs'
else:
    path = f'../paper/aabi/figs_aabi'

if not os.path.exists(path):
    os.makedirs(path)

fname = f'toy_bo_prob{args.expert_prob}_{args.al_acqf}'
plt.savefig(f'{path}/{fname}.pdf')
