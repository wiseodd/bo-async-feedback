from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
