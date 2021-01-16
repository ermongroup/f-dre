import os
import torch
from torchvision import datasets
from torch.utils.data import Dataset

import numpy as np
from .looping import LoopingDataset
from .vision import VisionDataset
from .dataset_utils import download_file_from_google_drive, check_integrity
from .mnist import MNIST


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


class ourMNIST(VisionDataset):
    """
    original MNIST with dequantization
    """
    def __init__(self,
                args,
                split='train',
                transform=None, target_transform=None, load_in_mem=False,
                download=True, **kwargs):
        super(ourMNIST, self).__init__(args.data_dir)

        self.split = split
        self.perc = args.perc
        self.lam = 1e-6
        self.root = os.path.join(args.data_dir, 'mnist/')
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
                args,
                split='train',
                transform=None, target_transform=None, load_in_mem=False,
                download=True, **kwargs):
        super(FlippedMNIST, self).__init__(args.data_dir)

        self.split = split
        self.perc = args.perc
        self.lam = 1e-6
        self.root = os.path.join(args.data_dir, 'mnist/')
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


class CMNISTalpha(VisionDataset):
    '''
    MNIST with alpha frac of samples recolored with blue background and (1-alpha) yellow.
    '''
    def __init__(self,
                args,
                split='train',
                transform=None, target_transform=None, load_in_mem=False,
                download=True, **kwargs):
        super(CMNIST, self).__init__(args.data_dir)

        self.split = split
        self.alpha = args.alpha
        mnist = MNIST()


        if split == 'train':
            og_mnist = mnist.trn
        elif split == 'val':
            og_mnist = mnist.val
        else:
            og_mnist = mnist.tst

        self.og_data = og_mnist.x
        self.og_y = og_mnist.y

        try:
            print('loading data')
            self.data = torch.load(os.path.join(args.data_dir, f'cmnist_{split}_{self.alpha}_data.pt'))
            self.labels = torch.load(os.path.join(args.data_dir, f'cmnist_{split}_{self.alpha}_labels.pt'))
        except:
            # split data into [0, 1] labels
            print(f'generating CMNIST {split} data')
            self.data, self.labels = self.initialize_data_splits()

            # if self.split == 'train':
            #     self.data = self.colorize_images(self.data)
            # else:
            #     self.data = self.colorize_images_test(self.data)

            torch.save(self.data, os.path.join(args.data_dir, f'cmnist_{split}_{self.alpha}_data.pt'))
            torch.save(self.labels, os.path.join(args.data_dir, f'cmnist_{split}_{self.alpha}_labels.pt'))

        # self.pos_data = self.data[self.labels==1]
        # self.neg_data = self.data[self.labels==0]

    def initialize_data_splits(self):
        '''
        Randomly select alpha % of digits to be recolored blue and the rest yellow.
        '''
        dset = []
        labels = []

        num_samples = len(self.og_data)
        num_blue = int(self.alpha * num_samples)

        blue_idx = np.random.choice(np.arange(num_samples), size=num_blue, replace=False)
        yellow_idx = np.setdiff1d(np.arange(num_samples), blue_idx)


        dset.append(torch.Tensor(self.og_data[blue_idx]))
        labels.append(torch.Tensor(self.og_y[blue_idx]))

        dset.append(torch.Tensor(self.og_data[yellow_idx]))
        labels.append(torch.Tensor(self.og_y[yellow_idx]))

        dset = torch.cat(dset)
        dset = torch.reshape(dset, (-1, 28, 28))  
        dset = torch.stack([dset, dset, dset], dim=1) # (n, 3, 28, 28)

        labels = torch.cat(labels)

        # recolor
        dset = (255 - dset)
        dset[:num_blue, 0, :, :] = 12
        dset[num_blue:, 2, :, :] = 12
        
        # this is the shape of the MAF implementation of MNIST
        dset = torch.reshape(dset, (-1, 3, 784))  # (n, 3, 784)

        return dset, labels

    def colorize_images(self, dset):
        # first flip the background and foreground color
        dset = (255 - dset)

        # first color alpha% of y = 0 in yellow
        n_zeros = len(self.labels[self.labels == 0])
        n_ones = len(self.labels[self.labels == 1])

        # color in yellow
        n_yellow = int(n_zeros * self.alpha)
        zero_idx = np.where(self.labels.numpy() == 0)[0]
        yellow_perm = np.random.permutation(zero_idx)
        to_yellow = yellow_perm[0:n_yellow]
        # to_blue = ~np.in1d(zero_idx, to_yellow)
        to_blue = yellow_perm[n_yellow:]

        # do the coloring
        dset[to_yellow, 2, :, :] = 12
        dset[to_blue, 0, :, :] = 12

        # then color alpha% of y = 1 in blue
        n_blue = int(n_ones * self.alpha)
        ones_idx = np.where(self.labels.numpy() == 1)[0]
        blue_perm = np.random.permutation(ones_idx)
        to_blue = blue_perm[0:n_blue]
        # to_yellow = ~np.in1d(ones_idx, to_blue)
        to_yellow = blue_perm[n_blue:]

        # do the coloring
        dset[to_blue, 0, :, :] = 12
        dset[to_yellow, 2, :, :] = 12

        return dset

    def colorize_images_test(self, dset):
        # first flip the background and foreground color
        dset = (255 - dset)

        # color all test images for y = 0 in yellow
        to_yellow = torch.where(self.labels == 0)[0]
        dset[to_yellow, 2, :, :] = 12

        # color all test images for y = 1 in blue
        to_blue = torch.where(self.labels == 1)[0]
        dset[to_blue, 0, :, :] = 12

        return dset

    def __getitem__(self, index):

        # get anchor data points
        item = self.data[index]
        label = self.labels[index]

        rand_idx = torch.randint(0, len(self.pos_data), (1, ))
        x_y1 = self.pos_data[rand_idx]

        rand_idx = torch.randint(0, len(self.neg_data), (1, ))
        x_y0 = self.neg_data[rand_idx]

        if label == 0:
            x_pos = x_y0
            x_neg = x_y1
        else:
            x_pos = x_y1
            x_neg = x_y0

        return item, x_pos, x_neg, label

    def __len__(self):
        return len(self.data)


class CMNIST_ERM(VisionDataset):
    def __init__(self, root, 
                config, 
                split='train',
                transform=None, target_transform=None, load_in_mem=False,
                download=True, **kwargs):
        super(CMNIST_ERM, self).__init__(root)

        self.split = split
        self.root = os.path.join(root, 'mnist/')
        self.alpha = config.data.alpha
        self.data = torch.load('/atlas/u/kechoi/contrastive/src/datasets/c_mnist_{}_{}_data.pt'.format(self.split, self.alpha))
        self.labels = torch.load('/atlas/u/kechoi/contrastive/src/datasets/c_mnist_{}_{}_labels.pt'.format(self.split, self.alpha))

    def __getitem__(self, index):

        # get data points
        item = self.data[index]
        label = self.labels[index]

        return item, label

    def __len__(self):
        return len(self.data)