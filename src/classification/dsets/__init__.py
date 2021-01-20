import os
import numpy as np

import torch
import torchvision.transforms as transforms
from .flipped_mnist import (
    SplitEncodedMNIST,
    SplitMNIST,
    SplitMNISTSubset
)
from .dataset_utils import *


def get_dataset(args, config):
    if config.data.random_flip is False:
        train_transform = test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])

    if config.data.dataset == 'MNIST':
        if config.data.x_space:
            print('using x-space for density ratio estimation')
            dataset = SplitMNIST(config, split='train')
            val_dataset = SplitMNIST(config, split='val')
            test_dataset = SplitMNIST(config, split='test')
        else:
            print('using z-space for density ratio estimation')
            dataset = SplitEncodedMNIST(config, split='train')
            val_dataset = SplitEncodedMNIST(config, split='val')
            test_dataset = SplitEncodedMNIST(config, split='test')
    elif config.data.dataset == 'SplitMNIST':
        dataset = SplitMNIST(config, split='train')
        val_dataset = SplitMNIST(config, split='val')
        test_dataset = SplitMNIST(config, split='test')
    elif config.data.dataset == 'SplitMNISTSubset':
        dataset = SplitMNISTSubset(config, split='train')
        val_dataset = SplitMNISTSubset(config, split='val')
        test_dataset = SplitMNISTSubset(config, split='test')
    elif config.data.dataset == 'CIFAR10':
        raise NotImplementedError
        # dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10'), train=True, download=True,
        #                   transform=train_transform)
        # test_dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10_test'), train=False, download=True,
        #                        transform=test_transform)

    elif config.data.dataset == 'CELEBA':
        raise NotImplementedError
        # if config.data.random_flip:
        #     dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba'), split='train',
        #                      transform=transforms.Compose([
        #                          transforms.CenterCrop(140),
        #                          transforms.Resize(config.data.image_size),
        #                          transforms.RandomHorizontalFlip(),
        #                          transforms.ToTensor(),
        #                      ]), download=True)
        # else:
        #     dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba'), split='train',
        #                      transform=transforms.Compose([
        #                          transforms.CenterCrop(140),
        #                          transforms.Resize(config.data.image_size),
        #                          transforms.ToTensor(),
        #                      ]), download=True)

        # test_dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba_test'), split='test',
        #                       transform=transforms.Compose([
        #                           transforms.CenterCrop(140),
        #                           transforms.Resize(config.data.image_size),
        #                           transforms.ToTensor(),
        #                       ]), download=True)
        # processing
        # num_items = len(dataset)
        # indices = list(range(num_items))
        # random_state = np.random.get_state()
        # np.random.seed(777)
        # np.random.shuffle(indices)
        # np.random.set_state(random_state)
        # train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        # test_dataset = Subset(dataset, test_indices)
        # dataset = Subset(dataset, train_indices)

    return dataset, val_dataset, test_dataset
    # return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)