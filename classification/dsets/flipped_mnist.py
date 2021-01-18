import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from .looping import LoopingDataset
from .cmnist import ourMNIST, FlippedMNIST


class SplitMNIST(Dataset):
    """
    same dataset class as SplitEncodedMNIST, except operating in x-space (mostly as a sanity check)
    """
    def __init__(self, config, split='train'):

        self.config = config
        self.perc = config.data.perc
        self.biased_dset = LoopingDataset(
            ourMNIST(config, split=split))
        self.ref_dset = LoopingDataset(
            FlippedMNIST(config, split=split))
    
    def __getitem__(self, index):
        ref_z, _ = self.ref_dset[index]
        biased_z, _ = self.biased_dset[index]

        ref_z = ref_z.float() / 255.
        biased_z = biased_z.float() / 255.

        #TODO: eventually also return attr label in addition to ref/bias label?
        return (ref_z, biased_z)
    
    def __len__(self):
        return len(self.ref_dset) + len(self.biased_dset)


class SplitEncodedMNIST(Dataset):
    """ 
    dataset that returns (ref_z, biased_z) when iterated through via dataloader
    (need to specify targets upon dataloading)
    """
    def __init__(self, config, split='train'):

        self.config = config
        self.perc = config.data.perc
        self.ref_dset = self.load_dataset(split, 'cmnist')
        self.biased_dset = self.load_dataset(split, 'mnist')

    def load_dataset(self, split, variant='mnist'):
        fpath = os.path.join(self.config.training.data_dir, 'maf_{}_{}_z.npz'.format(
            split, variant))
        record = np.load(fpath)
        print('loading dataset from {}'.format(fpath))
        # record = np.load(os.path.join(self.args.data_dir, 'maf_{}_{}_z_perc1.npz'.format(split, variant)))
        zs = torch.from_numpy(record['z']).float()
        ys = record['y']
        d_ys = torch.from_numpy(record['d_y']).float()

        # Truncate biased test/val set to be same size as reference val/test sets
        if (split == 'test' or split == 'val') and variant == 'mnist':
            # len(self.ref_dset) is always <= len(self.biased_dset)
            zs = zs[:len(self.ref_dset)]
            d_ys = d_ys[:len(self.ref_dset)]
        dataset = TensorDataset(zs, d_ys)
        dataset = LoopingDataset(dataset)
        return dataset
    
    def __getitem__(self, index):
        ref_z, _ = self.ref_dset[index]
        biased_z, _ = self.biased_dset[index]

        #TODO: eventually also return attr label in addition to ref/bias label?
        return (ref_z, biased_z)
    
    def __len__(self):
        return len(self.ref_dset) + len(self.biased_dset)