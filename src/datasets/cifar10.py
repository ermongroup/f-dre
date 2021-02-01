import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from .looping import LoopingDataset
from .vision import VisionDataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as T


# domain adaptation experiments 
class DASplitCIFAR10(VisionDataset):
    def __init__(self, root, 
                 config,
                 split='train',
                 target_type='attr',
                 transform=None, target_transform=None, load_in_mem=False,
                 download=True, **kwargs):
        super(DASplitCIFAR10, self).__init__(root)

        self.config = config
        self.split = split
        self.root = root
        self.perc = config.data.perc
        print('Instantiating x-space {} dataset with perc={}'.format(
            self.split, self.perc))

        data = CIFAR10(os.path.join(self.root, 'datasets', 'cifar10'), train=True if self.split != 'test' else False, download=True, transform=transform)
        labels = torch.tensor(data.targets)
        data = torch.stack([x[0] for x in data])

        if self.split == 'train':
            data = data[:-10000]
            labels = labels[:-10000]
        elif self.split == 'val':
            data = data[-10000:]
            labels = labels[-10000:]
        else:
            pass

        # datasets
        self.dataset = self.initialize_splits(data, labels, self.perc)

    def initialize_splits(self, data, labels, perc):
        print('classifying animals vs. vehicles...')

        # get appropriate attribute splits
        animal_labels = [2, 3, 4, 5, 6, 7]
        vehicle_labels = [0, 1, 8, 9]
        n_classes = len(animal_labels)

        animal_mask = torch.from_numpy(
            np.in1d(labels.numpy(), np.ravel(animal_labels)))
        vehicle_mask = torch.from_numpy(
            np.in1d(labels.numpy(), np.ravel(vehicle_labels)))

        vehicle = torch.where(vehicle_mask)[0]
        animal = torch.where(animal_mask)[0]

        # TODO
        # animal_split = int((len(animal) // 6) * 5)
        # vehicle_split = len(vehicle) // 4
        animal_split = int((len(animal) // 10) * 9)
        vehicle_split = len(vehicle) // 10

        bias_animal = animal[:animal_split]
        ref_animal = animal[animal_split:]

        bias_vehicle = vehicle[:vehicle_split]
        ref_vehicle = vehicle[vehicle_split:]

        bias = torch.cat([data[bias_animal], data[bias_vehicle]])
        ref = torch.cat([data[ref_animal], data[ref_vehicle]])

        # pseudolabels
        bias_y = torch.cat([
            torch.zeros(len(bias_animal)),  # animal = 0
            torch.ones(len(bias_vehicle))   # vehicle = 1
        ])
        ref_y = torch.cat([
            torch.zeros(len(ref_animal)),  # animal = 0
            torch.ones(len(ref_vehicle))   # vehicle = 1
        ])

        # datalabels
        data_bias_y = torch.cat([labels[bias_animal], labels[bias_vehicle]])
        data_ref_y = torch.cat([labels[ref_animal], labels[ref_vehicle]])

        # construct dataloaders (data, attribute, biased/ref dataset)
        ref_dataset = torch.utils.data.TensorDataset(
            ref, ref_y, data_ref_y)
        bias_dataset = torch.utils.data.TensorDataset(
            bias, bias_y, data_bias_y)

        # return the right split
        if self.split == 'test':
            return ref_dataset
        else:
            return bias_dataset

    def __getitem__(self, index):
        """
        Make sure dataset doesn't go out of bounds
        """
        item, label, _ = self.dataset[index]

        return item, label

    def __len__(self):
        return len(self.dataset)


#(DRE)
class DADRESplitCIFAR10(VisionDataset):
    def __init__(self, root, 
                 config,
                 split='train',
                 target_type='attr',
                 transform=None, target_transform=None, load_in_mem=False,
                 download=True, **kwargs):
        super(DADRESplitCIFAR10, self).__init__(root)

        self.config = config
        self.split = split
        self.root = root
        self.perc = config.data.perc
        print('Instantiating x-space {} dataset with perc={}'.format(
            self.split, self.perc))

        data = CIFAR10(os.path.join(self.root, 'datasets', 'cifar10'), train=True if self.split != 'test' else False, download=True, transform=transform)
        labels = torch.tensor(data.targets)
        data = torch.stack([x[0] for x in data])
        if self.split == 'train':
            data = data[:-10000]
            labels = labels[:-10000]
        elif self.split == 'val':
            data = data[-10000:]
            labels = labels[-10000:]
        else:
            pass

        # datasets
        dset_splits = self.initialize_splits(data, labels, self.perc)
        self.ref = dset_splits[0]
        self.bias = dset_splits[1]

        # if validation set, make sure ref and bias are balanced!
        # if self.split == 'val':
            # assert len(self.ref) == len(self.bias)
        print(self.split, len(self.bias), len(self.ref))

    def initialize_splits(self, data, labels, perc):
        print('classifying animals vs. vehicles...')

        # get appropriate attribute splits
        animal_labels = [2, 3, 4, 5, 6, 7]
        vehicle_labels = [0, 1, 8, 9]
        n_classes = len(animal_labels)

        animal_mask = torch.from_numpy(
            np.in1d(labels.numpy(), np.ravel(animal_labels)))
        vehicle_mask = torch.from_numpy(
            np.in1d(labels.numpy(), np.ravel(vehicle_labels)))

        vehicle = torch.where(vehicle_mask)[0]
        animal = torch.where(animal_mask)[0]

        # TODO
        # animal_split = int((len(animal) // 6) * 5)
        # vehicle_split = len(vehicle) // 4
        animal_split = int((len(animal) // 10) * 9)
        vehicle_split = len(vehicle) // 10

        bias_animal = animal[:animal_split]
        ref_animal = animal[animal_split:]

        bias_vehicle = vehicle[:vehicle_split]
        ref_vehicle = vehicle[vehicle_split:]

        bias = torch.cat([data[bias_animal], data[bias_vehicle]])
        ref = torch.cat([data[ref_animal], data[ref_vehicle]])

        # TODO
        # if self.split == 'val':
        #     # cut both datasets to be the same size
        #     n_ref = len(vehicle)
        #     animal = animal[0:n_ref]

        # pseudolabels
        ref_y = torch.ones(len(ref))  # y = 1 for ref
        bias_y = torch.zeros(len(bias))  # y = 0 for bias
        # datalabels
        data_bias_y = torch.cat([labels[bias_animal], labels[bias_vehicle]])
        data_ref_y = torch.cat([labels[ref_animal], labels[ref_vehicle]])

        # construct dataloaders (data, attribute, biased/ref dataset)
        ref_dataset = torch.utils.data.TensorDataset(
            ref, ref_y, data_ref_y)
        bias_dataset = torch.utils.data.TensorDataset(
            bias, bias_y, data_bias_y)

        return ref_dataset, bias_dataset

    def __getitem__(self, index):
        """
        Make sure dataset doesn't go out of bounds
        """
        min_len = min(len(self.ref), len(self.bias))
        if index >= min_len:
            index = np.random.choice(len(self.ref))

        # get items from both ref and bias dataset 
        ref_item, _, _ = self.ref[index]
        bias_item, _, _ = self.bias[index]

        return ref_item, bias_item

    def __len__(self):
        return len(self.bias)  # iterate through both at the same time



# -------
# targeted generation experiments
class AttrCIFAR10(VisionDataset):
    def __init__(self, root, 
                 config,
                 split='train',
                 target_type='attr',
                 transform=None, target_transform=None, load_in_mem=False,
                 download=True, **kwargs):
        super(AttrCIFAR10, self).__init__(root)

        self.config = config
        self.split = split
        self.root = root
        self.perc = config.data.perc
        print('Instantiating x-space {} dataset with perc={}'.format(
            self.split, self.perc))

        if self.split == 'val':
            transform = T.Compose([
            T.Resize(config.data.image_size),
            T.ToTensor()
        ])
        data = CIFAR10(os.path.join(self.root, 'datasets', 'cifar10'), train=True if self.split != 'test' else False, download=True, transform=transform)
        labels = torch.tensor(data.targets)
        data = torch.stack([x[0] for x in data])
        if self.split == 'train':
            data = data[:-10000]
            labels = labels[:-10000]
        elif self.split == 'val':
            data = data[-10000:]
            labels = labels[-10000:]
        else:
            pass
        self.data, self.labels = self.initialize_splits(data, labels, self.perc)
        print(self.split, len(self.data))

    def initialize_splits(self, data, labels, perc):
        print('classifying animals vs. vehicles...')

        # get appropriate attribute splits
        animal_labels = [2, 3, 4, 5, 6, 7]
        vehicle_labels = [0, 1, 8, 9]

        animal_mask = torch.from_numpy(
            np.in1d(labels.numpy(), np.ravel(animal_labels)))
        vehicle_mask = torch.from_numpy(
            np.in1d(labels.numpy(), np.ravel(vehicle_labels)))

        vehicle = torch.where(vehicle_mask)[0]
        animal = torch.where(animal_mask)[0]

        bias = data[animal]  # ref
        ref = data[vehicle]  # bias
        new_data = torch.cat([ref, bias])

        ref_y = torch.ones(len(ref))  # y = 1 for ref
        bias_y = torch.zeros(len(bias))  # y = 0 for bias
        new_labels = torch.cat([
            ref_y,
            bias_y
        ])
        return new_data, new_labels


    def __getitem__(self, index):
        item = self.data[index]
        label = self.labels[index]

        return item, label

    def __len__(self):
        return len(self.data)


class SplitCIFAR10(VisionDataset):
    def __init__(self, root, 
                 config,
                 split='train',
                 target_type='attr',
                 transform=None, target_transform=None, load_in_mem=False,
                 download=True, **kwargs):
        super(SplitCIFAR10, self).__init__(root)

        self.config = config
        self.split = split
        self.root = root
        self.perc = config.data.perc
        print('Instantiating x-space {} dataset with perc={}'.format(
            self.split, self.perc))

        data = CIFAR10(os.path.join(self.root, 'datasets', 'cifar10'), train=True if self.split != 'test' else False, download=True, transform=transform)
        labels = torch.tensor(data.targets)
        data = torch.stack([x[0] for x in data])
        if self.split == 'train':
            data = data[:-10000]
            labels = labels[:-10000]
        elif self.split == 'val':
            data = data[-10000:]
            labels = labels[-10000:]
        else:
            pass

        # datasets
        dset_splits = self.initialize_splits(data, labels, self.perc)
        self.ref = dset_splits[0]
        self.bias = dset_splits[1]

        # if validation set, make sure ref and bias are balanced!
        if self.split == 'val':
            assert len(self.ref) == len(self.bias)
        print(self.split, len(self.bias), len(self.ref))

    def initialize_splits(self, data, labels, perc):
        print('classifying animals vs. vehicles...')

        # get appropriate attribute splits
        animal_labels = [2, 3, 4, 5, 6, 7]
        vehicle_labels = [0, 1, 8, 9]
        n_classes = len(animal_labels)

        animal_mask = torch.from_numpy(
            np.in1d(labels.numpy(), np.ravel(animal_labels)))
        vehicle_mask = torch.from_numpy(
            np.in1d(labels.numpy(), np.ravel(vehicle_labels)))

        vehicle = torch.where(vehicle_mask)[0]
        animal = torch.where(animal_mask)[0]

        if self.split == 'val':
            # cut both datasets to be the same size
            n_ref = len(vehicle)
            animal = animal[0:n_ref]

        bias = data[animal]  # bias
        ref = data[vehicle]  # ref

        # pseudolabels
        ref_y = torch.ones(len(ref))  # y = 1 for ref
        bias_y = torch.zeros(len(bias))  # y = 0 for bias
        # datalabels
        data_bias_y = labels[animal]
        data_ref_y = labels[vehicle]

        # construct dataloaders (data, attribute, biased/ref dataset)
        ref_dataset = torch.utils.data.TensorDataset(
            ref, ref_y, data_ref_y)
        bias_dataset = torch.utils.data.TensorDataset(
            bias, bias_y, data_bias_y)

        return ref_dataset, bias_dataset

    def __getitem__(self, index):
        """
        Make sure dataset doesn't go out of bounds
        """
        min_len = min(len(self.ref), len(self.bias))
        if index >= min_len:
            index = np.random.choice(len(self.ref))

        # get items from both ref and bias dataset 
        ref_item, _, _ = self.ref[index]
        bias_item, _, _ = self.bias[index]

        return ref_item, bias_item

    def __len__(self):
        return len(self.ref)  # iterate through both at the same time



class ZCIFAR10(VisionDataset):
    def __init__(self, root, 
                 config,
                 split='train',
                 target_type='attr',
                 transform=None, target_transform=None, load_in_mem=False,
                 download=True, **kwargs):
        super(ZCIFAR10, self).__init__(root)

        self.config = config
        self.split = split

        # TODO: correct for proper path, without hardcoding
        self.root = '/atlas/u/kechoi/Glow-PyTorch/results/'
        # self.root = '/atlas/u/kechoi/ddim_private/results/logs/cifar10/'
        # self.root = '/atlas/u/kechoi/ddim_private/jan28_cifar10_z_redo/logs/jan28_cifar10_z_redo/'
        self.perc = config.data.perc
        print('Instantiating z-space {} dataset with perc={}'.format(
            self.split, self.perc))

        # load data
        self.data = torch.load(os.path.join(self.root, '{}_encoded_z.pt'.format(self.split)))
        self.labels = torch.load(os.path.join(self.root, '{}_encoded_labels.pt'.format(self.split)))
        print('loaded data from {}'.format(self.root))

        # datasets
        self.ref, self.bias = self.initialize_splits(self.perc)

        # if validation set, make sure ref and bias are balanced!
        if self.split == 'val':
            assert len(self.ref) == len(self.bias)
        print(self.split, len(self.bias), len(self.ref))

    def initialize_splits(self, perc):
        print('classifying animals vs. vehicles...')

        # get appropriate attribute splits
        animal_labels = [2, 3, 4, 5, 6, 7]
        vehicle_labels = [0, 1, 8, 9]
        n_classes = len(animal_labels)

        animal_mask = torch.from_numpy(
            np.in1d(self.labels.numpy(), np.ravel(animal_labels)))
        vehicle_mask = torch.from_numpy(
            np.in1d(self.labels.numpy(), np.ravel(vehicle_labels)))

        vehicle = torch.where(vehicle_mask)[0]
        animal = torch.where(animal_mask)[0]

        if self.split == 'val':
            # cut both datasets to be the same size
            n_ref = len(vehicle)
            animal = animal[0:n_ref]

        bias = self.data[animal]  # bias
        ref = self.data[vehicle]  # ref

        # pseudolabels
        ref_y = torch.ones(len(ref))  # y = 1 for ref
        bias_y = torch.zeros(len(bias))  # y = 0 for bias
        # datalabels
        data_bias_y = self.labels[animal]
        data_ref_y = self.labels[vehicle]

        # construct dataloaders (data, attribute, biased/ref dataset)
        ref_dataset = torch.utils.data.TensorDataset(
            ref, ref_y, data_ref_y)
        bias_dataset = torch.utils.data.TensorDataset(
            bias, bias_y, data_bias_y)

        return ref_dataset, bias_dataset

    def __getitem__(self, index):
        """
        Make sure dataset doesn't go out of bounds
        """
        min_len = min(len(self.ref), len(self.bias))
        if index >= min_len:
            index = np.random.choice(len(self.ref))

        # get items from both ref and bias dataset 
        ref_item, _, _ = self.ref[index]
        bias_item, _, _ = self.bias[index]

        # TODO: in case you need attribute labels later downstream
        # item = torch.stack([ref_item[0], bias_item[0]])
        # attr_label = torch.stack([ref_item[1], bias_item[1]])
        # label = torch.stack([ref_item[2], bias_item[2]])

        # return item, attr_label, label

        return ref_item, bias_item

    def __len__(self):
        # return len(self.ref) + len(self.bias)
        return len(self.ref)  # iterate through both at the same time