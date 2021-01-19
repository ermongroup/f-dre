import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from .looping import LoopingDataset

class EncodedCelebA(Dataset):
    """
    encodings of entire celebA dataset
    """
    def __init__(self, args, split='train'):
        self.args = args
        zs = torch.load(os.path.join(args.data_dir, args.z_file))
        labels = torch.load(os.path.join(args.data_dir, args.attr_file))
        
        start, end = self.get_data_idxs(split)
        self.zs = zs[start:end]
        self.labels = labels[start:end]
        
    def get_data_idxs(self, split):
        if split == 'train':
            start = 0 
            end = int(len(self.zs) * self.args.train_perc)
        elif split == 'test':
            start = int(len(self.zs) * self.args.train_perc)
            end = start + int(self.args.test_perc * len(self.zs))
        else: # split == 'val'
            test_start = int(len(self.zs) * self.args.train_perc)
            start =  test_start + int(self.args.test_perc * len(self.zs))
            end = len(self.zs)
        
        return start, end

    def __getitem__(self, index):
        return (self.zs[index], self.labels[index])
    
    def __len__(self):
        return len(self.labels)


class SplitEncodedCelebA(Dataset):
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
        class1_indices = np.flatnonzero(self.labels[:, self.class_idx]==0)
        class2_indices = np.flatnonzero(self.labels[:, self.class_idx]==1)
        n_class1_total = len(class1_indices)
        n_class2_total = len(class2_indices)
        n_samples_total = len(self.encoded_data)

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

class CelebA(Dataset):
    processed_file = 'processed.pt'
    partition_file = 'list_eval_partition.txt'
    attr_file = 'list_attr_celeba.txt'
    img_folder = 'img_align_celeba'
    attr_names = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young'.split()

    def __init__(self, root, train=True, transform=None, mini_data_size=None):
        self.root = os.path.join(os.path.expanduser(root), 'celeba')
        self.transform = transform

        # check if processed
        if not os.path.exists(os.path.join(self.root, self.processed_file)):
            self._process_and_save()
        data = torch.load(os.path.join(self.root, self.processed_file))

        if train:
            self.data = data['train']
        else:
            self.data = data['val']


        if mini_data_size != None:
            self.data = self.data[:mini_data_size]

    def __getitem__(self, idx):
        filename, attr = self.data[idx]
        img = Image.open(os.path.join(self.root, self.img_folder, filename))  # loads in RGB mode
        if self.transform is not None:
            img = self.transform(img)
        attr = torch.from_numpy(attr)
        return img, attr

    def __len__(self):
        return len(self.data)

    def _process_and_save(self):
        if not os.path.exists(os.path.join(self.root, self.attr_file)):
            raise RuntimeError('Dataset attributes file not found at {}.'.format(os.path.join(self.root, self.attr_file)))
        if not os.path.exists(os.path.join(self.root, self.partition_file)):
            raise RuntimeError('Dataset evaluation partitions file not found at {}.'.format(os.path.join(self.root, self.partition_file)))
        if not os.path.isdir(os.path.join(self.root, self.img_folder)):
            raise RuntimeError('Dataset image folder not found at {}.'.format(os.path.join(self.root, self.img_folder)))

        # read attributes file: list_attr_celeba.txt
        # First Row: number of images
        # Second Row: attribute names
        # Rest of the Rows: <image_id> <attribute_labels>
        with open(os.path.join(self.root, self.attr_file), 'r') as f:
            lines = f.readlines()
        n_files = int(lines[0])
        attr = [[l.split()[0], l.split()[1:]] for l in lines[2:]]  # [image_id.jpg, <attr_labels>]

        assert len(attr) == n_files, \
                'Mismatch b/n num entries in attributes file {} and reported num files {}'.format(len(attr), n_files)

        # read partition file: list_eval_partition.txt;
        # All Rows: <image_id> <evaluation_status>
        # "0" represents training image,
        # "1" represents validation image,
        # "2" represents testing image;
        data = [[], [], []]  # train, val, test
        unmatched = 0
        with open(os.path.join(self.root, self.partition_file), 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            fname, split = line.split()
            if attr[i][0] != fname:
                unmatched += 1
                continue
            data[int(split)].append([fname, np.array(attr[i][1], dtype=np.float32)])  # [image_id.jpg, <attr_labels>] by train/val/test


        if unmatched > 0: print('Unmatched partition filenames to attribute filenames: ', unmatched)
        assert sum(len(s) for s in data) == n_files, \
                'Mismatch b/n num entries in partition {} and reported num files {}'.format(sum(len(s) for s in filenames), n_files)

        # check image folder
        filenames = os.listdir(os.path.join(self.root, self.img_folder))
        assert len(filenames) == n_files, \
                'Mismatch b/n num files in image folder {} and report num files {}'.format(len(filenames), n_files)

        # save
        data = {'train': data[0], 'val': data[1], 'test': data[2]}
        with open(os.path.join(self.root, self.processed_file), 'wb') as f:
            torch.save(data, f)



if __name__ == '__main__':
    d = CelebA('~/Data/')
    print('Length: ', len(d))
    print('Image: ', d[0][0])
    print('Attr: ', d[0][1])

    import timeit
    t = timeit.timeit('d[np.random.randint(0,len(d))]', number=1000, globals=globals())
    print('Retrieval time: ', t)

    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    n_bits = 5
    t = T.Compose([T.CenterCrop(148),  # RealNVP preprocessing
                   T.Resize(64),
                   T.Lambda(lambda im: np.array(im, dtype=np.float32)),  # to numpy
                   T.Lambda(lambda x: np.floor(x / 2**(8 - n_bits)) / 2**n_bits), # lower bits
                   T.ToTensor(),
                   T.Lambda(lambda t: t + torch.rand(t.shape)/ 2**n_bits)])                     # dequantize
    d_ = CelebA('~/Data/', transform=t)
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(np.array(d[0][0]))
    axs[1].imshow(d_[0][0].numpy().transpose(1,2,0))
    plt.show()
