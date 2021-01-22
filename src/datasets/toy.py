import os
import numpy as np
import torch
import torch.distributions as dist
from torch.distributions import Normal
from torch.utils.data import Dataset, TensorDataset
from .looping import LoopingDataset


class GaussianMixtures(Dataset):
    def __init__(self, args, config, split='train'):
        self.args = args
        self.config = config
        self.split = split

        self.perc = config.data.perc
        self.input_size = config.data.input_size
        self.label_size = 1

        self.p_mu = self.config.data.mus[0]
        self.q_mu = self.config.data.mus[1]
        self.p = Normal(self.p_mu, 1)  # bias
        self.q = Normal(self.q_mu, 1)  # ref

        fpath = os.path.join(self.config.training.data_dir, 'gmm')
        try:
            data = np.load(os.path.join(fpath, 'gmm_p{}_q{}.npz'.format(self.p_mu, self.q_mu)))
        except:
            print('gmm dataset not found...generating')
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            data = self.generate_data()

        bias_data = data['p']
        ref_data = data['q']

        # train/val/test split
        if split == 'train':
            bias_data = bias_data[0:40000]
            ref_data = ref_data[0:40000]
        elif split == 'val':
            bias_data = bias_data[40000:45000]
            ref_data = ref_data[40000:45000]
        else:
            bias_data = bias_data[45000:]
            ref_data = ref_data[45000:]

        # perc split
        if split != 'val':
            # keep validation set balanced for calibration
            to_keep = int(len(bias_data) * self.perc)
            ref_data = ref_data[0:to_keep]
        self.biased_dset = LoopingDataset(
            TensorDataset(
            torch.from_numpy(bias_data).float(),
            torch.zeros(len(bias_data))
        ))
        self.ref_dset = LoopingDataset(
            TensorDataset(
            torch.from_numpy(ref_data).float(),
            torch.ones(len(ref_data))
        ))
        print(self.split)
        print(len(bias_data))
        print(len(ref_data))

    def generate_data(self):
        p = np.random.randn(50000,2) + self.p_mu
        q = np.random.randn(50000,2) + self.q_mu
        np.savez(os.path.join(self.config.training.data_dir, 'gmm', 'gmm_p{}_q{}.npz'.format(self.p_mu, self.q_mu)), **{
            'p': p,
            'q': q
        })
        return {'p': p, 'q': q}

    def __len__(self):
        return len(self.biased_dset) + len(self.ref_dset)

    def __getitem__(self, i):
        px, _ = self.biased_dset[i]
        qx, _ = self.ref_dset[i]

        return (qx, px)  # ref, bias


class GMM(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.p = Normal(self.config.data.mus[0], 1)  # bias
        self.q = Normal(self.config.data.mus[1], 1)  # ref

        self.perc = config.data.perc
        self.input_size = config.data.input_size
        self.label_size = 1

    def sample(self, n, fair=False):
        if not fair:
            n_q = int(n * self.perc)
        else:
            n_q = n
        px = self.p.sample((n, self.input_size))
        qx = self.q.sample((n_q, self.input_size))
        
        xs = torch.cat([px, qx])
        ys = torch.cat([
            torch.zeros(n),
            torch.ones(n_q)
        ])
        
        idx = torch.randperm(n + n_q)
        xs = xs[idx]
        ys = ys[idx]

        return xs, ys

    def get_density_ratios(self, x, log=False):
        if x is None:
            x, _ = self.sample(2000)
        log_p = self.p.log_prob(x).sum(-1)
        log_q = self.q.log_prob(x).sum(-1)
        log_r = log_q - log_p  # ref/bias
        if log:
            r = log_r
        else:
            r = torch.exp(log_r)

        return r


class ToyDistribution(dist.Distribution):
    def __init__(self, flip_var_order):
        super().__init__()
        self.flip_var_order = flip_var_order
        self.p_x2 = dist.Normal(0, 4)
        self.p_x1 = lambda x2: dist.Normal(0.25 * x2**2, 1)


    def rsample(self, sample_shape=torch.Size()):
        x2 = self.p_x2.sample(sample_shape)
        x1 = self.p_x1(x2).sample()
        if self.flip_var_order:
            return torch.stack((x2, x1), dim=-1).squeeze()
        else:
            return torch.stack((x1, x2), dim=-1).squeeze()

    def log_prob(self, value):
        if self.flip_var_order:
            value = value.flip(1)
        return self.p_x1(value[:,1]).log_prob(value[:,0]) + self.p_x2.log_prob(value[:,1])


class TOY(Dataset):
    def __init__(self, dataset_size=25000, flip_var_order=False):
        self.input_size = 2
        self.label_size = 1
        self.dataset_size = dataset_size
        self.base_dist = ToyDistribution(flip_var_order)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.base_dist.sample(), torch.zeros(self.label_size)