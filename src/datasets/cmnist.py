import os
import torch
from torchvision import datasets
from torch.utils.data import Dataset, TensorDataset
import numpy as np
from .vision import VisionDataset
from .looping import LoopingDataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


class ourMNIST(VisionDataset):
    """
    original MNIST with dequantization
    """
    def __init__(self,
                args,
                config,
                split='train',
                flipped=False,
                transform=None, target_transform=None, load_in_mem=False,
                download=True, **kwargs):
        super(ourMNIST, self).__init__(config.training.data_dir)

        self.args = args
        self.config = config
        self.split = split
        self.flipped = flipped
        self.perc = self.config.data.perc
        self.lam = 1e-6
        self.root = os.path.join(config.training.data_dir, 'mnist/')
        mnist = datasets.MNIST(self.root, train=True if self.split in ['train', 'val'] else False, download=True)  # don't apply transformations

        if split in ['train', 'val']:
            num_train = int(0.8 * len(mnist.train_data))
            train_idxs = np.random.choice(np.arange(len(mnist.train_data)), size=num_train, replace=False)
            val_idxs = np.setdiff1d(np.arange(len(mnist.train_data)), train_idxs)

            data_idxs = train_idxs if split == 'train' else val_idxs
            self.data = mnist.train_data[data_idxs]
            self.labels = mnist.train_labels[data_idxs]
        else:
            self.data = mnist.test_data
            self.labels = mnist.test_labels

        if flipped:
            self.data, self.labels  = self.initialize_data_splits(self.data, self.labels)
    
    def initialize_data_splits(self, data, labels):
        """
        set aside a balanced number of classes for specified perc
        """
        # construct even split for given perc
        if self.args.attr is None:
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
        else:
            # for attr classifier, include all data (ignore splits/percs)
            new_dset = data
            new_labels = labels
        
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
        
        if not self.args.classify:
            # dequantize input
            label = self.labels[index]
            item = self._data_transform(item)
        else:
            # for attr classification: label = 1 if flipped (reference), 0 else
            label = int(self.flipped)
        
        item = item.view((-1, 784))

        return item, label

    def __len__(self):
        return len(self.data)

class MNISTSubset(VisionDataset):
    '''
    MNIST with only subset of the digits
    '''
    def __init__(self,
                args,
                config,
                split='train',
                flipped=False,
                is_ref=False,
                transform=None, target_transform=None, load_in_mem=False,
                download=True, **kwargs):
        super(MNISTSubset, self).__init__(config.training.data_dir)
        # super(MNISTSubset, self).__init__(
        #         args, 
        #         config,
        #         split=split,
        #         flipped=flipped,
        #         transform=transform, 
        #         target_transform=target_transform, 
        #         load_in_mem=load_in_mem,
        #         download=download)

        self.args = args
        self.config = config
        self.split = split
        self.flipped = flipped
        self.is_ref = is_ref
        self.perc = self.config.data.perc
        self.lam = 1e-6
        self.root = os.path.join(config.training.data_dir, 'mnist/')
        # list of digits to include
        self.digits = torch.Tensor(self.config.data.digits)
        # digit_percs[i] = what % of the dataset digits[i] should make up
        self.digit_percs = torch.Tensor(self.config.data.digit_percs)
        
        mnist = datasets.MNIST(self.root, train=True if (self.split != 'test')  else False, download=True)  # don't apply transformations yet

        # get correct data split
        if split != 'test':
            data = mnist.train_data
            labels = mnist.train_labels
        else:
            data = mnist.test_data
            labels = mnist.test_labels
        
        self.data, self.labels = self.initialize_data_splits(data, labels, flipped, split)
    
    def initialize_data_splits(self, data, labels, flipped, split):
        # select datapoints with desired digits
        digit_idxs = [] 
        for digit in self.digits:
            digit_idxs.extend(torch.where(labels == digit)[0])
        data = data[digit_idxs]
        labels = labels[digit_idxs]

        # divide into train and val sets
        if split == 'train' or split == 'val':
            num_train = int(0.8 * len(data))
            train_idxs = np.random.choice(np.arange(len(data)), size=num_train, replace=False)
            val_idxs = np.setdiff1d(np.arange(len(data)), train_idxs)
            data_idxs = train_idxs if split == 'train' else val_idxs
            data = data[data_idxs]
            labels = labels[data_idxs]
        
        # cut down dataset size and construct splits
        max_perc_idx = torch.argmax(self.digit_percs)
        if self.args.attr is not None: # ignore perc
            n_samples_needed = sum(labels == self.digits[max_perc_idx]) // self.digit_percs[max_perc_idx]
        else:
            total_samples_available = len(labels)
            n_samples_needed = min(int(float(self.perc) * total_samples_available), sum(labels == self.digits[max_perc_idx]).item() // self.digit_percs[max_perc_idx])
            
        subset_idxs = []
        for digit, perc in zip(self.digits, self.digit_percs):
            digit_idxs = torch.where(labels == digit)[0]
            if self.args.attr is not None:
                # include all samples for attr classification
                n_digit_samples = len(digit_idxs)
            else:
                # balanced digit split for test/val set; split by digit_percs for train
                n_digit_samples = int(perc * n_samples_needed) if split == 'train' else int(n_samples_needed // len(self.digits))
            digit_idxs = digit_idxs[:n_digit_samples]
            subset_idxs.extend(digit_idxs)
        
        if flipped:
            data = (255 - data)

        return data[subset_idxs], labels[subset_idxs] 
    
    def _data_transform(self, x):
        # performs dequantization, rescaling, then logit transform
        x = (x + torch.rand(x.size())) / 256.
        x = logit_transform(x, self.lam)
        return x

    def __getitem__(self, index):

        # get anchor data points
        item = self.data[index]
        
        if not self.args.classify:
            # dequantize input
            label = self.labels[index]
            item = self._data_transform(item)
        else:
            # for attr classification
            if self.args.attr == 'background':
                label = int(self.flipped) # y_black_bkgd = 0, y_white_bkgd = 1
            elif self.args.attr == 'digit':
                label = int(self.is_ref) # y = 1 if ref, 0 if biased
                # label = self.labels[index]
            else:
                raise NotImplementedError('This attribute is not defined.')
        
        item = item.view((-1, 784))

        return item, label
    def __len__(self):
        return len(self.data)

class SplitEncodedMNIST(Dataset):
    '''
    Dataset that returns (ref_z, biased_z) when iterated through via dataloader
    (need to specify targets upon dataloading: y_ref = 1, y_biased = 0)
    '''
    def __init__(self, config, split='train'):
        self.config = config
        # name of flow model used for encoding:
        self.encoding_model_name = config.data.encoding_model 
        self.perc = config.data.perc
        self.ref_dset = self.load_dataset(split, config.data.encoded_dataset, 'ref')
        self.biased_dset = self.load_dataset(split, config.data.encoded_dataset, 'biased')

    def load_dataset(self, split, dataset, variant):
        fpath = os.path.join(
            self.config.training.data_dir, 
            'encodings', dataset, 
            f'{self.encoding_model_name}_{split}_{variant}_z_perc{self.perc}.npz'
        )
        
        record = np.load(fpath)
        print('loading dataset from {}'.format(fpath))
        zs = torch.from_numpy(record['z']).float()
        ys = record['y']
        d_ys = torch.from_numpy(record['d_y']).float()

        # Truncate biased test/val set to be same size as reference val/test sets
        if (split == 'test' or split == 'val') and variant == 'biased':
            # len(self.ref_dset) is always <= len(self.biased_dset)
            zs = zs[:len(self.ref_dset)]
            d_ys = d_ys[:len(self.ref_dset)]
        dataset = TensorDataset(zs, d_ys)
        dataset = LoopingDataset(dataset)
        return dataset
    
    def __getitem__(self, index):
        ref_z, _ = self.ref_dset[index]
        biased_z, _ = self.biased_dset[index]

        return (ref_z, biased_z)
    
    def __len__(self):
        return len(self.ref_dset) + len(self.biased_dset)
