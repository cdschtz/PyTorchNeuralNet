import torch
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset


class MnistDataset(Dataset):

    def __init__(self, train=True, transform=None):

        if train:
            train_in_file = 'data/mnist_small_train_in.txt'
            train_out_file = 'data/mnist_small_train_out.txt'
            self.data = pd.read_csv(train_in_file, dtype=np.float64)
            self.targets = pd.read_csv(train_out_file, dtype=np.float64)
        else:
            test_in_file = 'data/mnist_small_test_in.txt'
            test_out_file = 'data/mnist_small_test_out.txt'
            self.data = pd.read_csv(test_in_file, dtype=np.float64)
            self.targets = pd.read_csv(test_out_file, dtype=np.float64)

        self.classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven',
                        '8 - eight', '9 - nine']

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img, label = self.data[idx], int(self.targets[idx])
        return img, label

    def transform(self):
        self.data = torch.from_numpy(self.data.values).type('torch.FloatTensor')
        self.targets = torch.from_numpy(self.targets.values).type('torch.FloatTensor')

    def show_image(self, image):
        pass
