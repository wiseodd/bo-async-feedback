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
from models.reward_models import ToyRewardModel


problem = toy_problems.Ackley2()
problem_func = problem.get_function()

reward_model = ToyRewardModel(problem.dim)
reward_model.load_state_dict(torch.load('pretrained_models/reward_models/ackley2.pt'))
reward_model.eval()

FIG_WIDTH = 1
FIG_HEIGHT = 0.15
rc_params, fig_width, fig_height = plot_utils.get_mpl_rcParams(
    FIG_WIDTH, FIG_HEIGHT
)
plt.rcParams.update(rc_params)

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, constrained_layout=True)
fig.set_size_inches(fig_width, fig_height)

# Plot f(x) landscape
n = 500
x = np.linspace(problem.bounds[0, 0], problem.bounds[1, 0], n)
y = np.linspace(problem.bounds[0, 1], problem.bounds[1, 1], n)
X, Y = np.meshgrid(x, y)
X, Y = torch.tensor(X).float(), torch.tensor(Y).float()
XY = torch.cat([X.reshape(n**2, 1), Y.reshape(n**2, 1)], dim=1)  # (n^2, 2)
Z = problem_func(XY)  # (n^2,)
axs[0].contourf(X, Y, Z.reshape(n, n))

# Plot r(x, f(x)) landscape
with torch.no_grad():
    R = reward_model(torch.cat([XY, Z.unsqueeze(1)], dim=1)).squeeze()
    axs[1].contourf(X, Y, -R.reshape(n, n))  # Negative reward since f is min problem

# Save results
path = f'../paper/figs'
if not os.path.exists(path):
    os.makedirs(path)

fname = f'fig2'
plt.savefig(f'{path}/{fname}.pdf')

