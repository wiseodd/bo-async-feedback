import torch
from botorch.test_functions import Ackley, Hartmann, Rastrigin, Levy

DIMS = {'hartmann6': 6, 'ackley10': 10, 'rastrigin10': 10, 'levy10': 10}
BOUNDS = {
    'hartmann6': torch.tensor(DIMS['hartmann6']*[[0., 1.]]).T,
    'ackley10': torch.tensor(DIMS['ackley10']*[[-32.768, 32.768]]).T,
    'rastrigin10': torch.tensor(DIMS['rastrigin10']*[[-5.12, 5.12]]).T,
    'levy10': torch.tensor(DIMS['levy10']*[[-10., 10.]]).T,
}

for name, bounds in BOUNDS.items():
    assert bounds.shape == (2, DIMS[name])

CLASSES = {'hartmann6': Hartmann, 'ackley10': Ackley, 'rastrigin10': Rastrigin, 'levy10': Levy}
FUNCS = {k: v(dim=DIMS[k], bounds=BOUNDS[k].T) for k, v in CLASSES.items()}

IS_MAXIMIZE = {
    'hartmann6': False, 'ackley10': False, 'rastrigin10': False, 'levy10': False
}
