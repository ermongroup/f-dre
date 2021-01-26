from functools import partial
import numpy as np

import torch
from torch.utils.data.dataset import ConcatDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

import datasets
from .cmnist import (
    ourMNIST,
    MNISTSubset,
    SplitEncodedMNIST
)
from .toy import (
    Gaussian, 
    GaussianMixtures, 
    EncodedGaussianMixtures
)
from .mi_gaussians import (
    GaussiansForMI,
    MIGaussians,
    EncodedMIGaussians
)

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

def fetch_dataloaders(dataset_name, batch_size, device, args, config, flip_toy_var_order=False, toy_train_size=25000, toy_test_size=5000):
    val_dataset = None
    # grab datasets
    if dataset_name in ['MNIST']:
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

    elif dataset_name in ['ourMNIST']:
        input_dims = 784
        label_size = 10
        lam = 1e-6

        train_dataset = ourMNIST(args, config, split='train')
        val_dataset = ourMNIST(args, config, split='val')
        test_dataset = ourMNIST(args, config, split='test')
        
    elif dataset_name in ['BackgroundMNIST']:
        '''
        MNIST with black and white backgrounds
        '''
        input_dims = 784
        label_size = 10
        lam = 1e-6

        # black background
        train_mnist = ourMNIST(args, config, split='train')
        val_mnist = ourMNIST(args, config, split='val')
        test_mnist = ourMNIST(args, config, split='test')

        # white background
        train_flipped = ourMNIST(args, config, split='train', flipped=True)
        val_flipped = ourMNIST(args, config, split='val', flipped=True)
        test_flipped = ourMNIST(args, config, split='test', flipped=True)
        
        if args.encode_z:
            # keep MNIST and flipped MNIST separate for encoding
            for dataset in (train_mnist, train_flipped):
                dataset.input_dims = input_dims
                dataset.input_size = int(np.prod(input_dims))
                dataset.label_size = label_size
            
            kwargs = {'num_workers': 1, 'pin_memory': True} if device.type is 'cuda' else {}

            train_loader = DataLoader(train_mnist, batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_mnist, batch_size, shuffle=False, **kwargs)
            test_loader = DataLoader(test_mnist, batch_size, shuffle=False, **kwargs)
            train_loader2 = DataLoader(train_flipped, batch_size, shuffle=True, **kwargs)
            val_loader2 = DataLoader(val_flipped, batch_size, shuffle=False, **kwargs)
            test_loader2 = DataLoader(test_flipped, batch_size, shuffle=False, **kwargs)

            return [train_loader, train_loader2], [val_loader, val_loader2], [test_loader, test_loader2]
        else:
            # combine both
            train_dataset = ConcatDataset([train_mnist, train_flipped])
            val_dataset = ConcatDataset([val_mnist, val_flipped])
            test_dataset = ConcatDataset([test_mnist, test_flipped])

    elif dataset_name in ['BackgroundMNISTSubset']:
        '''
        Subset of MNIST digits with black and white backgrounds; same digits
        '''
        input_dims = 784
        label_size = 10
        lam = 1e-6

        ref_perc = config.data.perc
        config.data.digits = config.data.biased_digits
        config.data.digit_percs = config.data.biased_digit_percs
        config.data.perc = 1.0

        train_mnist = MNISTSubset(args, config, split='train')
        val_mnist = MNISTSubset(args, config, split='val')
        test_mnist = MNISTSubset(args, config, split='test')

        config.data.digits = config.data.ref_digits
        config.data.digit_percs = config.data.ref_digit_percs
        config.data.perc = ref_perc

        train_flipped = MNISTSubset(args, config, split='train', flipped=True)
        val_flipped = MNISTSubset(args, config, split='val', flipped=True)
        test_flipped = MNISTSubset(args, config, split='test', flipped=True)
        
        if args.encode_z:
            # keep regular and flipped MNIST separate for encoding
            for dataset in (train_mnist, train_flipped):
                dataset.input_dims = input_dims
                dataset.input_size = int(np.prod(input_dims))
                dataset.label_size = label_size
            
            kwargs = {'num_workers': 1, 'pin_memory': True} if device.type is 'cuda' else {}

            train_loader = DataLoader(train_mnist, batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_mnist, batch_size, shuffle=False, **kwargs)
            test_loader = DataLoader(test_mnist, batch_size, shuffle=False, **kwargs)
            train_loader2 = DataLoader(train_flipped, batch_size, shuffle=True, **kwargs)
            val_loader2 = DataLoader(val_flipped, batch_size, shuffle=False, **kwargs)
            test_loader2 = DataLoader(test_flipped, batch_size, shuffle=False, **kwargs)

            return [train_loader, train_loader2], [val_loader, val_loader2], [test_loader, test_loader2]
        else:
            train_dataset = ConcatDataset([train_mnist, train_flipped])
            val_dataset = ConcatDataset([val_mnist, val_flipped])
            test_dataset = ConcatDataset([test_mnist, test_flipped])


    elif dataset_name in ['DigitMNISTSubset']:
        '''
        Biased and reference sets with different digit subsets (but same background color)
        '''
        input_dims = 784
        label_size = 10
        lam = 1e-6

        ref_perc = config.data.perc
        config.data.digits = config.data.biased_digits
        config.data.digit_percs = config.data.biased_digit_percs
        config.data.perc = 1.0

        train_biased = MNISTSubset(args, config, split='train')
        val_biased = MNISTSubset(args, config, split='val')
        test_biased = MNISTSubset(args, config, split='test')

        config.data.digits = config.data.ref_digits
        config.data.digit_percs = config.data.ref_digit_percs
        config.data.perc = ref_perc

        train_ref = MNISTSubset(args, config, split='train', is_ref=True)
        val_ref = MNISTSubset(args, config, split='val', is_ref=True)
        test_ref = MNISTSubset(args, config, split='test', is_ref=True)
        if args.encode_z:
            # keep each dataset separate for encoding
            for dataset in (train_biased, train_ref):
                dataset.input_dims = input_dims
                dataset.input_size = int(np.prod(input_dims))
                dataset.label_size = label_size
            
            kwargs = {'num_workers': 1, 'pin_memory': True} if device.type is 'cuda' else {}

            train_loader_biased = DataLoader(train_biased, batch_size, shuffle=True, **kwargs)
            val_loader_biased = DataLoader(val_biased, batch_size, shuffle=False, **kwargs)
            test_loader_biased = DataLoader(test_biased, batch_size, shuffle=False, **kwargs)
            train_loader_ref = DataLoader(train_ref, batch_size, shuffle=True, **kwargs)
            val_loader_ref = DataLoader(val_ref, batch_size, shuffle=False, **kwargs)
            test_loader_ref = DataLoader(test_ref, batch_size, shuffle=False, **kwargs)

            return [train_loader_biased, train_loader_ref], [val_loader_biased, val_loader_ref], [test_loader_biased, test_loader_ref]

        train_dataset = ConcatDataset([train_biased, train_ref])
        val_dataset = ConcatDataset([val_biased, val_ref])
        test_dataset = ConcatDataset([test_biased, test_ref])
    elif dataset_name in ['SplitEncodedMNIST']:
        input_dims = 784
        label_size = 1
        lam = 1e-6

        train_dataset = SplitEncodedMNIST(config, split='train')
        val_dataset = SplitEncodedMNIST(config, split='val')
        test_dataset = SplitEncodedMNIST(config, split='test')

    elif dataset_name in ['GMM_flow']:
        input_dims = 2
        label_size = 1
        lam = 1e-6

        train_biased = Gaussian(args, config, 'bias', split='train')
        val_biased = Gaussian(args, config, 'bias', split='val')
        test_biased = Gaussian(args, config, 'bias', split='test')

        train_ref = Gaussian(args, config, 'ref', split='train')
        val_ref = Gaussian(args, config, 'ref', split='val')
        test_ref = Gaussian(args, config, 'ref', split='test')

        if args.encode_z:
            # keep each dataset separate for encoding
            for dataset in (train_biased, train_ref):
                dataset.input_dims = input_dims
                dataset.input_size = int(np.prod(input_dims))
                dataset.label_size = label_size
            
            kwargs = {'num_workers': 1, 'pin_memory': True} if device.type is 'cuda' else {}

            train_loader_biased = DataLoader(train_biased, batch_size, shuffle=True, **kwargs)
            val_loader_biased = DataLoader(val_biased, batch_size, shuffle=False, **kwargs)
            test_loader_biased = DataLoader(test_biased, batch_size, shuffle=False, **kwargs)
            train_loader_ref = DataLoader(train_ref, batch_size, shuffle=True, **kwargs)
            val_loader_ref = DataLoader(val_ref, batch_size, shuffle=False, **kwargs)
            test_loader_ref = DataLoader(test_ref, batch_size, shuffle=False, **kwargs)

            return [train_loader_biased, train_loader_ref], [val_loader_biased, val_loader_ref], [test_loader_biased, test_loader_ref]

        train_dataset = ConcatDataset([train_biased, train_ref])
        val_dataset = ConcatDataset([val_biased, val_ref])
        test_dataset = ConcatDataset([test_biased, test_ref])

    elif dataset_name in ['GMM']:
        input_dims = 2
        label_size = 1
        lam = 1e-6

        if config.data.x_space:  # x-space
            train_dataset = GaussianMixtures(args, config, split='train')
            val_dataset = GaussianMixtures(args, config, split='val')
            test_dataset = GaussianMixtures(args, config, split='test')
        else:
            # z-space
            print('using encodings in z-space...')
            train_dataset = EncodedGaussianMixtures(
                args, config, split='train')
            val_dataset = EncodedGaussianMixtures(args, config, split='val')
            test_dataset = EncodedGaussianMixtures(args, config, split='test')

    elif dataset_name in ['MI']:
        input_dims = config.data.input_size
        label_size = 1
        lam = 1e-6

        if config.data.x_space:  # x-space
            train_dataset = GaussiansForMI(config, split='train')
            val_dataset = GaussiansForMI(config, split='val')
            test_dataset = GaussiansForMI(config, split='test')
        else:
            # z-space
            print('using encodings in z-space...')
            train_dataset = EncodedMIGaussians(config, split='train')
            val_dataset = EncodedMIGaussians(config, split='val')
            test_dataset = EncodedMIGaussians(config, split='test')

    elif dataset_name == 'MI_flow':
        input_dims = config.data.input_size
        label_size = 1
        lam = 1e-6

        train_biased = MIGaussians(config, 'bias', split='train')
        val_biased = MIGaussians(config, 'bias', split='val')
        test_biased = MIGaussians(config, 'bias', split='test')

        train_ref = MIGaussians(config, 'ref', split='train')
        val_ref = MIGaussians(config, 'ref', split='val')
        test_ref = MIGaussians(config, 'ref', split='test')

        if args.encode_z:
            raise NotImplementedError

        train_dataset = ConcatDataset([train_biased, train_ref])
        val_dataset = ConcatDataset([val_biased, val_ref])
        test_dataset = ConcatDataset([test_biased, test_ref])

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

    if val_dataset is not None:
        val_dataset.input_dims = input_dims
        val_dataset.input_size = int(np.prod(input_dims))
        val_dataset.label_size = label_size
        val_dataset.lam = lam

    # construct dataloaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if device.type is 'cuda' else {}

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, **kwargs) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader
