import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from .looping import LoopingDataset
from .vision import VisionDataset
import torchvision.transforms as T


class OmniglotMixture(VisionDataset):
    def __init__(self, root, 
                 config,
                 split='train',
                 target_type='attr',
                 transform=None, target_transform=None, load_in_mem=False,
                 download=True, synthetic=False, **kwargs):
        super(OmniglotMixture, self).__init__(root)

        self.config = config
        self.split = split
        self.root = root
        self.perc = config.data.perc
        self.synthetic = synthetic
        # print('Instantiating x-space {} dataset with perc={}'.format(
            # self.split, self.perc))

        # paths to real/synthetic data
        ref_data = np.load('/atlas/u/kechoi/DAGAN/datasets/omniglot_data.npy')
        bias_data = np.load('/atlas/u/kechoi/DAGAN/datasets/gen_omniglot_data.npy')
        # TODO: will have to experiment with train/val/test splits
        # TODO: maybe a "perc" argument can be used here
        
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
        # TODO
        item = self.dataset[index]
        label = self.labels[index]

        # return item, label
        return ref_item, bias_item

    def __len__(self):
        # TODO
        return len(self.dataset)


class Omniglot(VisionDataset):
    def __init__(self, root, 
                 config,
                 split='train',
                 target_type='attr',
                 transform=None, target_transform=None, load_in_mem=False,
                 download=True, synthetic=False, **kwargs):
        super(Omniglot, self).__init__(root)

        self.config = config
        self.split = split
        self.root = root
        self.perc = config.data.perc
        self.synthetic = synthetic
        # print('Instantiating x-space {} dataset with perc={}'.format(
            # self.split, self.perc))

        if not self.synthetic:
            data = np.load('/atlas/u/kechoi/DAGAN/datasets/omniglot_data.npy')
        else:
            data = np.load('/atlas/u/kechoi/DAGAN/datasets/gen_omniglot_data.npy')
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