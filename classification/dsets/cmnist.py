import os
import torch
from torchvision import datasets
from torch.utils.data import Dataset

import numpy as np
from .looping import LoopingDataset
from .vision import VisionDataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


class ourMNIST(VisionDataset):
    """
    original MNIST with dequantization
    """
    def __init__(self,
                config,
                split='train',
                transform=None, target_transform=None, load_in_mem=False,
                download=True, **kwargs):
        super(ourMNIST, self).__init__(config.data_dir)

        self.config = config
        self.split = split
        self.perc = config.data.perc
        self.lam = 1e-6
        self.root = os.path.join(config.training.data_dir, 'mnist/')
        mnist = datasets.MNIST(self.root, train=True if self.split == 'train' else False, download=True)  # don't apply transformations yet

        if split == 'train':
            self.data = mnist.train_data
            self.labels = mnist.train_labels
        else:
            self.data = mnist.test_data
            self.labels = mnist.test_labels

    def _data_transform(self, x):
        # performs dequantization, rescaling, then logit transform
        x = (x + torch.rand(x.size())) / 256.
        x = logit_transform(x, self.lam)
        return x

    def __getitem__(self, index):

        # get anchor data points
        item = self.data[index]
        label = self.labels[index]

        # dequantize input
        # (TODO: maybe this won't work out of the box without rng)
        item = self._data_transform(item)
        item = item.view((-1, 784))

        return item, label

    def __len__(self):
        return len(self.data)


class FlippedMNIST(VisionDataset):
    '''
    MNIST with samples recolored with blue background and (1-alpha) yellow.
    '''
    def __init__(self,
                config,
                split='train',
                transform=None, target_transform=None, load_in_mem=False,
                download=True, **kwconfig):
        super(FlippedMNIST, self).__init__(config.data_dir)

        self.config = config
        self.split = split
        self.perc = config.data.perc
        self.lam = 1e-6
        self.root = os.path.join(config.training.data_dir, 'mnist/')
        mnist = datasets.MNIST(self.root, train=True if self.split == 'train' else False, download=True)  # don't apply transformations yet

        if split == 'train':
            data = mnist.train_data
            labels = mnist.train_labels
        else:
            data = mnist.test_data
            labels = mnist.test_labels

        self.data, self.labels = self.initialize_data_splits(data, labels)

    def initialize_data_splits(self, data, labels):
        """
        set aside a balanced number of classes for specified perc
        """
        '''
        Randomly select alpha % of digits to be recolored blue and the rest yellow.
        '''
        n_examples = int(len(data) * self.perc)
        unique = torch.unique(labels)
        n_classes = len(unique)

        new_dset = []
        new_labels = []
        for class_label in unique:
            num_samples = n_examples // n_classes
            sub_y = labels[labels==class_label][0:num_samples]
            sub_x = data[labels==class_label][0:num_samples]

            # add examples
            new_labels.append(sub_y)
            new_dset.append(sub_x)
        new_labels = torch.cat(new_labels)
        new_dset = torch.cat(new_dset)

        # apply reverse black/white background
        new_dset = (255 - new_dset)

        return new_dset, new_labels

    def _data_transform(self, x):
        # performs dequantization, rescaling, then logit transform
        x = (x + torch.rand(x.size())) / 256.
        x = logit_transform(x, self.lam)
        return x

    def __getitem__(self, index):

        # get anchor data points
        item = self.data[index]
        label = self.labels[index]

        # dequantize input
        # (TODO: maybe this won't work out of the box without rng)
        item = self._data_transform(item)
        item = item.view((-1, 784))

        return item, label

    def __len__(self):
        return len(self.data)