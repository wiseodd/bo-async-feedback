import torch
from torch import nn
from collections import UserDict


class ToyRewardModel(nn.Module):

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
            if hasattr(self, 'f_mean'):
                out = out - getattr(self, 'f_mean')
            if hasattr(self, 'f_std'):
                out = out / getattr(self, 'f_std')

            return out