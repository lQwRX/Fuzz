import torch
from torch.utils.data import Dataset

class MessageDataset(Dataset):
    def __init__(self, data):
        """
        data: numpy array of shape (N, max_len)
        """
        self.data = torch.LongTensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]