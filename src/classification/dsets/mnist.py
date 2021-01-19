import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from .looping import LoopingDataset


class EncodedCMNIST(Dataset):
    """
    encodings of entire CMNIST dataset
    """
    def __init__(self, args, split='train'):
        self.args = args
        # assumes data has been cached by split
        print('loading in zs and labels from numpy...(split={})'.format(split))
        record = np.load(os.path.join(args.data_dir, '{}_z.npz'.format(split)))
        self.zs = record['z']
        self.labels = record['y']
        
        print('loaded!')

    def __getitem__(self, index):
        return (self.zs[index], self.labels[index])
    
    def __len__(self):
        return len(self.labels)


class SplitEncodedCMNIST(Dataset):
    """ 
    dataset that returns (ref_z, biased_z) when iterated through via dataloader
    (need to specify targets upon dataloading)
    """
    def __init__(self, args, split='train'):

        self.ref_split = args.ref_split
        self.biased_split = args.biased_split
        self.perc = args.perc
        # the class we're using to split the data (e.g., male/female):
        self.class_idx = args.class_idx

        self.encoded_data = EncodedCelebA(args, split=split)
        self.labels = self.encoded_data.labels # attr labels

        # create ref/biased dataset based on given splits/perc
        self.biased_dset = self._get_split_dataset(self.biased_split, is_biased=True, perc=self.perc)
        self.ref_dset = self._get_split_dataset(self.ref_split, is_biased=False)

    def  _get_split_dataset(self, split, is_biased=True, perc=1.0):
        """
        Returns LoopingDataset of data according to split/perc
        """
        class1_indices = np.flatnonzero(self.labels[:, self.class_idx]==-1)
        class2_indices = np.flatnonzero(self.labels[:, self.class_idx]==1)
        n_class1_total = len(class1_indices)
        n_class2_total = len(class2_indices)
        n_samples_total = len(self.encoded_data)
        print('n_class1_total: ', n_class1_total)
        print('n_class2_total: ', n_class2_total)
        print('n_samples_total: ', n_samples_total)

        # Determine the  number of each class needed; unless we have a
        # desired split that is exactly the same as the original dataset, 
        # we will need to leave out some samples
        if is_biased:
            if int(split * n_samples_total) >= n_class1_total:
                n_class1 = n_class1_total
                n_class2 = int(n_class1 // split) - n_class1
            else:
                n_class2 = n_class2_total
                n_class1 = int(n_class2 // split) - n_class2
        else:
            n_samples_needed = int(perc * len(self.biased_dset))
            if int(split * n_samples_needed) >= int(perc * n_class1_total):
                n_class1 = int(perc * n_class1_total)
                n_class2 = int(n_class1 // split) - n_class1
            else:
                n_class2 = int(perc * n_class2_total)
                n_class1 = int(n_class2 // split) - n_class2


        indices = np.append(class1_indices[:n_class1],class2_indices[:n_class2])

        # Subset creates a subset of encoded_data with specified indices
        # LoopingDataset handles situation where len(self.ref_dset) != len(self.biased_dset)
        return LoopingDataset(Subset(self.encoded_data, indices))
    
    def __getitem__(self, index):
        ref_z, _ = self.ref_dset[index]
        biased_z, _ = self.biased_dset[index]

        #TODO: eventually also return attr label in addition to ref/bias label?
        return (ref_z, biased_z)
    
    def __len__(self):
        return len(self.ref_dset) + len(self.biased_dset)