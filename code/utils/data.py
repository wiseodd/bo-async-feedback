import torch.utils.data as data_utils


class ListDataset(data_utils.Dataset):
    """
    PyTorch dataset where the source comes from a list.

    Parameters:
    -----------
    dataset: List[Any]
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
