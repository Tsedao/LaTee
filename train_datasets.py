import numpy as np


from torch.utils.data import Dataset



class SyntheticEventdata(Dataset):

    def __init__(self, history) -> None:
        super().__init__()

        self.train_x, self.train_y = history

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        y = self.train_y[idx]
        X = self.train_x[idx]
        event_time, event_type = X[:, 0], X[:, 1] 
        return event_time, event_type,  y
    