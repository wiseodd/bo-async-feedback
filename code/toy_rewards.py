import numpy as np
import torch
from torch import nn
from torch import distributions as dists
import torch.utils.data as data_utils
import tqdm
import itertools
from collections import UserDict
from transformers import get_scheduler
import torchmetrics as tm

import problems.toy as toy_problems
import utils
from models.reward_models import ToyRewardModel

import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--problem', default='ackley10', choices=['levy10', 'ackley10', 'hartmann6', 'rastrigin10'])
parser.add_argument('--train_size', type=int, default=5000)
parser.add_argument('--val_size', type=int, default=1000)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--randseed', type=int, default=9999)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)


problem = toy_problems.PROBLEM_LIST[args.problem]()
f_true = problem.get_function()

x_samples = utils.sample_x(args.train_size + args.val_size, problem.bounds)
fx_samples = f_true(x_samples)

# Randomly sample pairs
idx_pairs = itertools.combinations(range(len(x_samples)), 2)
idx_pairs = list(idx_pairs)
np.random.shuffle(idx_pairs)
idx_pairs = np.array(idx_pairs[:args.train_size+args.val_size])  # (train_size+val_size, 2)

# Get List[((x_0, f(x_0)), (x_1, f(x_1)))]
dataset = []
for idx_pair in idx_pairs:
    x_0, x_1 = x_samples[idx_pair[0]], x_samples[idx_pair[1]]
    fx_0, fx_1 = fx_samples[idx_pair[0]], fx_samples[idx_pair[0]]
    label = torch.tensor(problem.get_preference((x_0, fx_0), (x_1, fx_1))).long()
    dataset.append(UserDict({  # UserDict is required by Laplace
        'x_0': x_0, 'x_1': x_1, 'fx_0': fx_0, 'fx_1': fx_1, 'labels': label
    }))

# As dataloader
class PreferenceDataset(data_utils.Dataset):
    def __init__(self, split='train'):
        assert split in ['train', 'val']
        self.split = split
        self.dataset = {
            'train': dataset[:args.train_size],
            'val': dataset[args.train_size:]
        }

    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, idx):
        return self.dataset[self.split][idx]

train_loader = data_utils.DataLoader(PreferenceDataset(split='train'), batch_size=64)
val_loader = data_utils.DataLoader(PreferenceDataset(split='val'), batch_size=64)

model = ToyRewardModel(dim=problem.dim)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
scd = get_scheduler(
    name='cosine',
    optimizer=opt,
    num_warmup_steps=0,
    num_training_steps=args.n_epochs*len(train_loader),
)
pbar = tqdm.trange(args.n_epochs)

for it in pbar:
    for data in train_loader:
        model.train()
        opt.zero_grad()
        out = model(data)
        loss = loss_fn(out, data['labels'])
        loss.backward()
        opt.step()
        scd.step()

    # Validate
    with torch.no_grad():
        model.eval()

        train_acc_metric = tm.Accuracy(task='multiclass', num_classes=2)
        train_loss = 0.
        for data in train_loader:
            out = model(data)
            train_acc_metric(torch.softmax(out, dim=-1), data['labels'])
            train_loss += out.shape[0]*loss_fn(out, data['labels'])

        val_acc_metric = tm.Accuracy(task='multiclass', num_classes=2)
        val_loss = 0.
        for data in val_loader:
            out = model(data)
            val_acc_metric(torch.softmax(out, dim=-1), data['labels'])
            val_loss += out.shape[0]*loss_fn(out, data['labels'])

        train_acc = train_acc_metric.compute()
        val_acc = val_acc_metric.compute()
        pbar.set_description(f'[Train_loss: {train_loss/args.train_size:.3f}; train_acc: {train_acc:.3f}; val loss: {val_loss/args.val_size:.3f}; val_acc: {val_acc:.3f}]')

# Reward model normalization
with torch.no_grad():
    mean_metric = tm.MeanMetric()
    mean_sq_metric = tm.MeanMetric()

    for data in train_loader:
        out = model(data).flatten()  # (batch_size*2,)
        mean_metric(out)
        mean_sq_metric(out**2)

    mean = mean_metric.compute()
    sq_mean = mean**2  # E(f(x))^2
    mean_sq = mean_sq_metric.compute()  # E(f(x)^2)
    std = torch.sqrt(mean_sq - sq_mean)  # std(f(x))

    model.register_buffer('f_mean', mean)
    model.register_buffer('f_std', std)

# Test Laplace
from laplace import Laplace

la = Laplace(model, likelihood='reward_modeling', hessian_structure='kron', subset_of_weights='all')
la.fit(train_loader)
la.optimize_prior_precision()
f_mean, f_var = la(torch.randn(3, problem.dim+1))
print('Laplace A-OK')

# Save model
path = f'pretrained_models/reward_models'
if not os.path.exists(path):
    os.makedirs(path)

torch.save(model.state_dict(), f'{path}/{args.problem}.pt')
