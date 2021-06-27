import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .vision import VisionDataset

DATA_ROOT = '/atlas/u/kechoi/f-dre/data/datasets/omniglot/'


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


class OmniglotMixture(VisionDataset):
    """
    Dataset class exclusively for training the DRE classifier
    (will need a different dataset class for actual downstream classification)

    NOTE: assumes that DAGAN has already been used to generate samples!
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

        # paths to real omniglot data
        ref_data = np.load(os.path.join(DATA_ROOT, 'omniglot_data.npy'))
        ref_data = ref_data[:1200, :, :, :, :]
        # path to samples generated
        bias_data = np.load(os.path.join(DATA_ROOT, 'generated_omniglot.npy')).reshape(1200, 100, 28, 28, 1)
        bias_data = bias_data / 255.

        self.ref_dataset, self.bias_dataset = self.initialize_data_splits(ref_data, bias_data)

    def initialize_data_splits(self, ref_data, bias_data):
        if self.split == 'train':
            ref = ref_data[:, 0:10, :, :, :]
            bias = bias_data[:, 0:50, :, :, :]
        elif self.split == 'val':
            ref = ref_data[:, 10:15, :, :, :]
            bias = bias_data[:, 50:55, :, :, :]
        else:
            ref = ref_data[:, 15:, :, :, :]
            bias = bias_data[:, 55:60, :, :, :]

        # reshape bc numpy 
        bias = bias.reshape(-1, 28, 28, 1)
        ref = ref.reshape(-1, 28, 28, 1)

        # tensorize
        bias = torch.from_numpy(bias).permute((0, 3, 1, 2)).float()
        ref = torch.from_numpy(ref).permute((0, 3, 1, 2)).float()

        if self.flow:
            print('applying flow transforms in advance...')
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
        bias_item, _ = self.bias_dataset[index]
        if index >= len(self.ref_dataset):
            index = np.random.choice(len(self.ref_dataset))
        ref_item, _ = self.ref_dataset[index]

        return ref_item, bias_item

    def __len__(self):
        # iterate through both at the same time
        return len(self.bias_dataset)

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
            true_data = np.load(os.path.join(DATA_ROOT, 'omniglot_data.npy'))
            true_data = true_data[0:1200, :, :, :, :]
            data = true_data
        else:
            # tensorflow shenanigans
            print('loading synthetic data for training...')
            data = np.load(os.path.join(DATA_ROOT,
                                        'generated_omniglot.npy')).reshape(1200, 100, 28, 28, 1)
            data = data[0:1200, :, :, :, :]
            data = data / 255.
            true_data = np.load(os.path.join(DATA_ROOT, 'omniglot_data.npy'))[0:1200, :, :, :, :]
        if self.split == 'train':
            if self.synthetic:
                data = data[:, 0:50, :, :, :]
                n_labels = 50
            else:
                data = data[:, 0:10, :, :, :]
                n_labels = 10
            aux_labels = np.ones(1200*n_labels)  # fake is y=1
            if self.augment:
                real = true_data[:, 0:10, :, :, :]
                data = np.hstack([data, real])
                n_labels += 10
                aux_labels = np.hstack([aux_labels, np.zeros(1200*10)])  # real is y = 0
            labels = np.repeat(np.arange(len(data)), n_labels)
        elif self.split == 'val':
            data = true_data[:, 10:15, :, :, :]
            labels = np.repeat(np.arange(len(data)), 5)
            aux_labels = np.zeros(1200*5)
        else:
            data = true_data[:, 15:, :, :, :]
            labels = np.repeat(np.arange(len(data)), 5)
            aux_labels = np.zeros(1200*5)
        data = data.reshape(-1, 28, 28, 1)
        labels = np.vstack([labels, aux_labels])
        self.dataset = torch.from_numpy(data).permute((0, 3, 1, 2)).float()
        self.labels = torch.from_numpy(labels).float().permute(1,0)  # (2, n_data)

    def __getitem__(self, index):
        """
        Make sure dataset doesn't go out of bounds
        """
        item = self.dataset[index]
        label = self.labels[index]

        return item, label

    def __len__(self):
        return len(self.dataset)