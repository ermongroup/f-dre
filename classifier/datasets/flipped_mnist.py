import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from .looping import LoopingDataset


class SplitEncodedMNIST(Dataset):
    """ 
    dataset that returns (ref_z, biased_z) when iterated through via dataloader
    (need to specify targets upon dataloading)
    """
    def __init__(self, args, split='train'):

        self.args = args
        self.perc = args.perc
        self.biased_dset = self.load_dataset(split, 'mnist')
        self.ref_dset = self.load_dataset(split, 'cmnist')

    def load_dataset(self, split, variant='mnist'):
        record = np.load(os.path.join(self.args.data_dir, 'maf_{}_{}_z.npz'.format(split, variant)))
        zs = torch.from_numpy(record['z']).float()
        ys = record['y']
        d_ys = torch.from_numpy(record['d_y']).float()

        # HACK for calibration
        if split == 'test' and variant == 'mnist':
            zs = zs[0:5000]
            d_ys = d_ys[0:5000]
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