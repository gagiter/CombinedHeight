
from torch.utils.data import Dataset


class Vaihgen(Dataset):
    def __init__(self):
        self.samples = [1, 2, 3]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]