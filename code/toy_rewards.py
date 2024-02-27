import numpy as np
import torch
from torch import nn
from torch import distributions as dists
import torch.utils.data as data_utils
import tqdm
import itertools
from transformers import get_scheduler
from torchmetrics import Accuracy

import toy_problems
import utils

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

class RewardModel(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim+1, 50),  # +1 for the f(x) input
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, data):
        """
        data: torch.Tensor or dict
            If training then dict
            Else torch.Tensor of shape (batch_size, dim)
        """
        if isinstance(data, dict):
            x_0, x_1 = data['x_0'], data['x_1']
            fx_0, fx_1 = data['fx_0'], data['fx_1']

            # (batch_size, dim+1)
            input_0 = torch.cat([x_0, fx_0.unsqueeze(-1)], dim=-1)
            input_1 = torch.cat([x_1, fx_1.unsqueeze(-1)], dim=-1)
            inputs = torch.stack([input_0, input_1], dim=1)  # (batch_size, 2, dim+1)

            flat_inputs = inputs.reshape(-1, inputs.shape[-1])  # (batch_size*2, dim+1)
            flat_logits = self.net(flat_inputs)  # (batch_size*2, 1)
            return flat_logits.reshape(inputs.shape[0], 2)  # (batch_size, 2)
        else:  # data is torch.Tensor
            return self.net(data)  # (batch_size, 1)

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
    dataset.append({
        'x_0': x_0, 'x_1': x_1, 'fx_0': fx_0, 'fx_1': fx_1, 'labels': label
    })

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

model = RewardModel(dim=problem.dim)
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

        train_acc_metric = Accuracy(task='multiclass', num_classes=2)
        train_loss = 0.
        for data in train_loader:
            out = model(data)
            train_acc_metric(torch.softmax(out, dim=-1), data['labels'])
            train_loss += out.shape[0]*loss_fn(out, data['labels'])

        val_acc_metric = Accuracy(task='multiclass', num_classes=2)
        val_loss = 0.
        for data in val_loader:
            out = model(data)
            val_acc_metric(torch.softmax(out, dim=-1), data['labels'])
            val_loss += out.shape[0]*loss_fn(out, data['labels'])

        train_acc = train_acc_metric.compute()
        val_acc = val_acc_metric.compute()
        pbar.set_description(f'[Train_loss: {train_loss/args.train_size:.3f}; train_acc: {train_acc:.3f}; val loss: {val_loss/args.val_size:.3f}; val_acc: {val_acc:.3f}]')

# TODO reward model normalization

# Save model
path = f'pretrained_models/reward_models'
if not os.path.exists(path):
    os.makedirs(path)

torch.save(model.state_dict(), f'{path}/{args.problem}.pt')
