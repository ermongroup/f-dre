import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from .looping import LoopingDataset
from .vision import VisionDataset
# from torchvision.datasets import Omniglot
import torchvision.transforms as T


class Omniglot(VisionDataset):
    def __init__(self, root, 
                 config,
                 split='train',
                 target_type='attr',
                 transform=None, target_transform=None, load_in_mem=False,
                 download=True, **kwargs):
        super(Omniglot, self).__init__(root)

        self.config = config
        self.split = split
        self.root = root
        self.perc = config.data.perc
        # print('Instantiating x-space {} dataset with perc={}'.format(
            # self.split, self.perc))

        data = np.load('/atlas/u/kechoi/DAGAN/datasets/omniglot_data.npy')
        if self.split == 'train':
            data = data[:, 0:10, :, :, :]
            labels = np.repeat(np.arange(len(data)), 10)
        elif self.split == 'val':
            data = data[:, 10:15, :, :, :]
            labels = np.repeat(np.arange(len(data)), 5)
        else:
            data = data[:, 15:, :, :, :]
            labels = np.repeat(np.arange(len(data)), 5)
        data = data.reshape(-1, 28, 28, 1)
        self.dataset = torch.from_numpy(data).permute((0, 3, 1, 2)).float()
        self.labels = torch.from_numpy(labels).float()

    def __getitem__(self, index):
        """
        Make sure dataset doesn't go out of bounds
        """
        item = self.dataset[index]
        label = self.labels[index]

        return item, label

    def __len__(self):
        return len(self.dataset)