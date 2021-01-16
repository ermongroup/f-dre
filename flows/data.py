from functools import partial
import numpy as np

import torch
from torch.utils.data.dataset import ConcatDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

import datasets

# --------------------
# Helper functions
# --------------------

def logit(x, eps=1e-5):
    x.clamp_(eps, 1 - eps)
    return x.log() - (1 - x).log()

def one_hot(x, label_size):
    out = torch.zeros(len(x), label_size).to(x.device)
    out[torch.arange(len(x)), x] = 1
    return out

def load_dataset(name):
    exec('from datasets.{} import {}'.format(name.lower(), name))
    return locals()[name]


# --------------------
# Dataloaders
# --------------------

def fetch_dataloaders(dataset_name, batch_size, device, args, flip_toy_var_order=False, toy_train_size=25000, toy_test_size=5000):

    # grab datasets
    if dataset_name in ['GAS', 'POWER', 'HEPMASS', 'MINIBOONE', 'BSDS300']:  # use the constructors by MAF authors
        dataset = load_dataset(dataset_name)()

        # join train and val data again
        train_data = np.concatenate((dataset.trn.x, dataset.val.x), axis=0)

        # construct datasets
        train_dataset = TensorDataset(torch.from_numpy(train_data.astype(np.float32)))
        test_dataset  = TensorDataset(torch.from_numpy(dataset.tst.x.astype(np.float32)))

        input_dims = dataset.n_dims
        label_size = None
        lam = None

    elif dataset_name in ['MNIST']:
        dataset = load_dataset(dataset_name)()

        # join train and val data again
        train_x = np.concatenate((dataset.trn.x, dataset.val.x), axis=0).astype(np.float32)
        train_y = np.concatenate((dataset.trn.y, dataset.val.y), axis=0).astype(np.float32)

        # construct datasets
        train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        test_dataset  = TensorDataset(torch.from_numpy(dataset.tst.x.astype(np.float32)),
                                      torch.from_numpy(dataset.tst.y.astype(np.float32)))

        input_dims = dataset.n_dims
        label_size = 10
        lam = dataset.alpha

    elif dataset_name in ['MNIST_combined']:
        mnist = load_dataset('MNIST')(rgb=True)
        
        # join train and val data again
        mnist_train_x = np.concatenate((mnist.trn.x, mnist.val.x), axis=0).astype(np.float32)
        mnist_train_y = np.concatenate((mnist.trn.y, mnist.val.y), axis=0).astype(np.float32)

        # construct datasets
        mnist_train = TensorDataset(torch.from_numpy(mnist_train_x), torch.from_numpy(mnist_train_y))
        mnist_test  = TensorDataset(torch.from_numpy(mnist.tst.x.astype(np.float32)),
                                      torch.from_numpy(mnist.tst.y.astype(np.float32)))

        #TODO: CMNIST load_dataset
        train_cmnist = load_dataset('CMNIST')(args, split='train')
        val_cmnist = load_dataset('CMNIST')(args, split='val')
        test_cmnist = load_dataset('CMNIST')(args, split='test')

        # join train and val data again
        cmnist_train_x = np.concatenate((train_cmnist.data, val_cmnist.data), axis=0).astype(np.float32)
        cmnist_train_y = np.concatenate((train_cmnist.labels, val_cmnist.labels), axis=0).astype(np.float32)

        # construct datasets
        cmnist_train = TensorDataset(torch.from_numpy(cmnist_train_x), torch.from_numpy(cmnist_train_y))
        cmnist_test = TensorDataset(test_cmnist.data, test_cmnist.labels)

        train_dataset = ConcatDataset([mnist_train, cmnist_train])
        test_dataset = ConcatDataset([mnist_test, cmnist_test])
        
        input_dims = mnist.n_dims
        label_size = 10
        lam = mnist.alpha

    elif dataset_name in ['TOY', 'MOONS']:  # use own constructors
        train_dataset = load_dataset(dataset_name)(toy_train_size, flip_toy_var_order)
        test_dataset = load_dataset(dataset_name)(toy_test_size, flip_toy_var_order)

        input_dims = train_dataset.input_size
        label_size = train_dataset.label_size
        lam = None

    # imaging dataset pulled from torchvision
    elif dataset_name in ['CIFAR10']:
        label_size = 10

        # MAF logit trainform parameter (cf. MAF paper 4.3
        lam = 1e-6 if dataset_name == 'mnist' else 5e-2

        # MAF paper converts image data to logit space via transform described in section 4.3
        image_transforms = T.Compose([T.ToTensor(),
                                      T.Lambda(lambda x: x + torch.rand(*x.shape) / 256.),    # dequantize (cf MAF paper)
                                      T.Lambda(lambda x: logit(lam + (1 - 2 * lam) * x))])    # to logit space (cf MAF paper)
        target_transforms = T.Lambda(lambda x: partial(one_hot, label_size=label_size)(x))

        train_dataset = load_dataset(dataset_name)(root=datasets.root, train=True, transform=image_transforms, target_transform=target_transforms)
        test_dataset =  load_dataset(dataset_name)(root=datasets.root, train=True, transform=image_transforms, target_transform=target_transforms)

        input_dims = train_dataset[0][0].shape

    else:
        raise ValueError('Unrecognized dataset.')


    # keep input dims, input size and label size
    train_dataset.input_dims = input_dims
    train_dataset.input_size = int(np.prod(input_dims))
    train_dataset.label_size = label_size
    train_dataset.lam = lam

    test_dataset.input_dims = input_dims
    test_dataset.input_size = int(np.prod(input_dims))
    test_dataset.label_size = label_size
    test_dataset.lam = lam

    # construct dataloaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if device.type is 'cuda' else {}

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader
