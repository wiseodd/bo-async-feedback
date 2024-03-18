import torch
from torch import nn
from collections import UserDict


class ToyRewardModel(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 50),  # +1 for the f(x) input
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        self.register_buffer('f_mean', torch.tensor(0., dtype=torch.float))
        self.register_buffer('f_std', torch.tensor(1., dtype=torch.float))

    def forward(self, data):
        """
        data: torch.Tensor or dict
            If training then dict
            Else torch.Tensor of shape (batch_size, dim)
        """
        if isinstance(data, UserDict):
            device = next(self.parameters()).device
            x_0, x_1 = data['x_0'], data['x_1']
            x_0.to(device)
            x_1.to(device)

            # (batch_size, dim+1)
            inputs = torch.stack([x_0, x_1], dim=1)  # (batch_size, 2, dim+1)
            flat_inputs = inputs.reshape(-1, inputs.shape[-1])  # (batch_size*2, dim+1)
            flat_logits = self.net(flat_inputs)  # (batch_size*2, 1)
            return flat_logits.reshape(inputs.shape[0], 2)  # (batch_size, 2)
        else:  # data is torch.Tensor
            out = self.net(data)  # (batch_size, 1)

            # Normalize
            out = out - getattr(self, 'f_mean')
            out = out / getattr(self, 'f_std')

            return out


class ToyRewardModelWithOutput(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim+1, 50),  # +1 for the f(x) input
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        self.register_buffer('f_mean', torch.tensor(0., dtype=torch.float))
        self.register_buffer('f_std', torch.tensor(1., dtype=torch.float))

    def forward(self, data):
        """
        data: torch.Tensor or dict
            If training then dict
            Else torch.Tensor of shape (batch_size, dim)
        """
        if isinstance(data, UserDict):
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
            out = self.net(data)  # (batch_size, 1)

            # Normalize
            out = out - getattr(self, 'f_mean')
            out = out / getattr(self, 'f_std')

            return out
