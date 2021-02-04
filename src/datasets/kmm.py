import os
import numpy as np
import torch
import torch.distributions as dist
from torch.distributions import Normal
from torch.utils.data import Dataset, TensorDataset
from .looping import LoopingDataset


class KMM(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split

        self.perc = config.data.perc
        self.input_size = config.data.input_size
        self.label_size = 1

        if self.split == 'train':
            source_record = np.load('/atlas/u/kechoi/multi-fairgen/data/kmm/source.npz')
            target_record = np.load('/atlas/u/kechoi/multi-fairgen/data/kmm/target.npz')
            data = np.vstack([source_record['x'], target_record['x']])
            labels = np.hstack([source_record['y'], target_record['y']])
        else:
            record = np.load('/atlas/u/kechoi/multi-fairgen/data/kmm/target_test.npz')
            data = record['x']
            labels = record['y']
        self.dataset = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        label = self.labels[i]

        return item, label