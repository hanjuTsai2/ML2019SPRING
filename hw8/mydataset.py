import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, imgs, labels=None, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        if not np.any(self.labels):
            img = self.imgs[index]
            if self.transform is not None:
                img = self.transform(img)
            return img
        else:
            img = self.imgs[index]
            label = self.labels[index]
            if self.transform is not None:
                img = self.transform(img)
            return img, label

    def __len__(self):
        return len(self.imgs)
