import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from .looping import LoopingDataset
from .vision import VisionDataset
import torchvision.transforms as T


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


class OmniglotMixture(VisionDataset):
    """
    Dataset class exclusively for training the DRE classifier (will need a different dataset class for actual downstream classification)
    """
    def __init__(self, root, 
                 config,
                 split='train',
                 target_type='attr',
                 transform=None, target_transform=None, load_in_mem=False,
                 download=True, **kwargs):
        super(OmniglotMixture, self).__init__(root)

        self.config = config
        self.split = split
        self.root = root
        self.perc = config.data.perc
        self.flow = True if config.model.name == 'maf' else False
        # print('Instantiating x-space {} dataset with perc={}'.format(
            # self.split, self.perc))

        # paths to real/synthetic data
        ref_data = np.load('/atlas/u/kechoi/DAGAN/datasets/omniglot_data.npy')
        # tensorflow samples
        bias_data = np.load('/atlas/u/kechoi/DAGAN/datasets/generated_omniglot/generated_omniglot.npy').reshape(1622, 100, 28, 28, 1)
        # bias_data = bias_data[0:1200, :, :, :, :]
        # TODO: commented this out so that I'm augmenting across all classes
        bias_data = bias_data / 255.
        # TODO: will have to experiment with train/val/test splits
        # TODO: maybe a "perc" argument can be used here

        self.ref_dataset, self.bias_dataset = self.initialize_data_splits(ref_data, bias_data)

    def initialize_data_splits(self, ref_data, bias_data):
        if self.split == 'train':
            ref = ref_data[:, 0:15, :, :, :]
            bias = bias_data[:, 0:15, :, :, :]
            # bias = bias_data[:, 0:5, :, :, :]
        elif self.split == 'val':
            ref = ref_data[:, 15:18, :, :, :]
            bias = bias_data[:, 15:18, :, :, :]
        else:
            ref = ref_data[:, 18:, :, :, :]
            bias = bias_data[:, 18:, :, :, :]

        # reshape bc numpy 
        bias = bias.reshape(-1, 28, 28, 1)
        ref = ref.reshape(-1, 28, 28, 1)

        # tensorize
        bias = torch.from_numpy(bias).permute((0, 3, 1, 2)).float()
        ref = torch.from_numpy(ref).permute((0, 3, 1, 2)).float()

        if self.flow:
            print('applying flow transforms in advance...')
            # apply the data transformations in advance :p
            bias = self._data_transform(bias)
            ref = self._data_transform(ref)

        # pseudolabels
        bias_y = torch.zeros(len(bias))
        ref_y = torch.ones(len(ref))

        # construct dataloaders (data, biased/ref dataset)
        # NOTE: not saving actual data labels for now
        ref_dataset = torch.utils.data.TensorDataset(ref, ref_y)
        bias_dataset = torch.utils.data.TensorDataset(bias, bias_y)

        return ref_dataset, bias_dataset

    def __getitem__(self, index):
        """
        Make sure dataset doesn't go out of bounds
        """
        ref_item, _ = self.ref_dataset[index]
        bias_item, _ = self.bias_dataset[index]

        return ref_item, bias_item

    def __len__(self):
        # iterate through both at the same time
        return len(self.ref_dataset)

    def _data_transform(self, x):
        # data is originally between [0,1], so change it back
        x = (x * 255).byte()
        # performs dequantization, rescaling, then logit transform
        x = (x + torch.rand(x.size())) / 256.
        x = logit_transform(x)
        
        return x


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
        self.augment = config.data.augment
        if self.augment:
            print('augmenting real data with synthetic data...')

        if not self.synthetic:
            true_data = np.load('/atlas/u/kechoi/DAGAN/datasets/omniglot_data.npy')
            true_data = true_data[0:1200, :, :, :, :]
            data = true_data
        else:
            # tensorflow shenanigans
            print('loading synthetic data for training...')
            data = np.load('/atlas/u/kechoi/DAGAN/datasets/generated_omniglot/generated_omniglot.npy').reshape(1622, 100, 28, 28, 1)
            data = data[0:1200, :, :, :, :]
            data = data / 255.
            true_data = np.load('/atlas/u/kechoi/DAGAN/datasets/omniglot_data.npy')[0:1200, :, :, :, :]
        if self.split == 'train':
            # data = data[:, 0:15, :, :, :]
            data = data[:, 0:5, :, :, :]
            n_labels = 5
            if self.augment:
                real = true_data[:, 0:15, :, :, :]
                data = np.hstack([data, real])
                n_labels = 20
            labels = np.repeat(np.arange(len(data)), n_labels)
        elif self.split == 'val':
            data = true_data[:, 15:18, :, :, :]
            labels = np.repeat(np.arange(len(data)), 3)
        else:
            data = true_data[:, 18:, :, :, :]
            labels = np.repeat(np.arange(len(data)), 2)
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