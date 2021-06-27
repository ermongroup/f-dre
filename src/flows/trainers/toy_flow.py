import os
import sys
import time

import copy
import logging
from pprint import pprint
from tqdm import tqdm
import math
import numpy as np
import yaml

from datasets.data import fetch_dataloaders
from flows.models.maf import *
from flows.models.ema import EMAHelper
from classification.models.resnet import ResnetClassifier
from classification.models.mlp import MLPClassifier

from flows.functions import get_optimizer

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data_utils
from torchvision.utils import save_image

import wandb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')

# (TODO) for pre-trained attribute classifiers
CHECKPOINT_DIR = '/atlas/u/madeline/multi-fairgen/src/classification/checkpoints'
ATTR_CLFS = {
    'digit': os.path.join(
        CHECKPOINT_DIR, 'digits_attr_clf', 'model_best.pth'),
    'background': os.path.join(
        CHECKPOINT_DIR,'background_attr_clf', 'model_best.pth')     
}


class ToyFlow(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        
        # if necessary, load (pretrained) density ratio estimates
        # self.classifier = 

    def get_model(self):
        # TODO: do something about these args
        if self.config.model.name == 'maf':
            model = MAF(self.config.model.n_blocks, self.config.model.input_size, self.config.model.hidden_size, self.config.model.n_hidden, None, self.config.model.activation_fn, self.config.model.input_order, batch_norm=not self.config.model.no_batch_norm)
        else:
            raise ValueError('Unrecognized model.')
        return model

    def load_classifier(self, args, ckpt_path):
        config_path = os.path.dirname(os.path.dirname(ckpt_path.rstrip('/')))
        with open(os.path.join(config_path, 'config.yaml')) as f:
            config = yaml.safe_load(f)
        config = dict2namespace(config)

        if config.model.name == 'resnet':
            clf = ResnetClassifier(args)
        elif config.model.name == 'mlp':
            clf = MLPClassifier(config)
        else:
            raise NotImplementedError(f'Classification model [{config.model.name}] not implemented.')
        
        assert os.path.exists(ckpt_path)
        
        # load state dict
        state_dict = torch.load(ckpt_path)['state_dict']
        clf.load_state_dict(state_dict)
        clf = clf.to(self.device)
        return clf

    def train(self):
        args, config = self.args, self.config
        train_dataloader, val_dataloader, test_dataloader = fetch_dataloaders(self.config.data.dataset, self.config.training.batch_size, self.device, self.args, self.config, self.config.data.flip_toy_var_order)
        
        model = self.get_model()
        model = model.to(self.device)
        # model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        best_eval_logprob = float('-inf')
        if self.args.resume_training:
            print('restoring checkpoint from {}'.format(args.restore_file))
            state = torch.load(os.path.join(args.restore_file, "best_model_checkpoint.pt"), map_location=self.device)
            model.load_state_dict(state['model_state'])
            optimizer.load_state_dict(state['optimizer_state'])
            start_epoch = state['epoch'] + 1
        model = torch.nn.DataParallel(model)

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            # original maf code
            for i, data in enumerate(train_dataloader):
                model.train()
                step += 1

                # check if labeled dataset
                if len(data) == 1:
                    x, y = data[0], None
                else:
                    x, y = data
                    y = y.squeeze().to(self.device)
                x = x.view(x.shape[0], -1).to(self.device)
                loss = -model.module.log_prob(x).mean(0)
                # get summary to log to wandb
                summary = dict(
                    train_loss=loss.item(),
                    epoch=epoch,
                    batch=i
                )
                wandb.log(summary)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if i % self.config.training.log_interval == 0:
                    print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                        epoch, start_epoch + self.config.training.n_epochs, i, len(train_dataloader), loss.item()))

                data_start = time.time()

            # now evaluate and save metrics/checkpoints
            eval_logprob, _ = self.test(
                model, test_dataloader, epoch, self.args)
            # get summary to log to wandb
            summary = dict(
                test_logp=eval_logprob.item(),
                epoch=epoch,
            )
            wandb.log(summary)
            # save training checkpoint
            torch.save({
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer_state': optimizer.state_dict()},
                os.path.join(args.out_dir, 'model_checkpoint.pt'))
            # save best state
            if eval_logprob > best_eval_logprob:
                best_eval_logprob = eval_logprob
                torch.save({
                    'epoch': epoch,
                    'model_state': model.module.state_dict(),
                    'optimizer_state': optimizer.state_dict()},
                    os.path.join(args.out_dir, 'best_model_checkpoint.pt'))
                # generate samples
                # self.plot_sample_and_density(self.args, model, test_dataloader.dataset, step=epoch)

    @torch.no_grad()
    def test(self, model, dataloader, epoch, args):
        model.eval()
        logprobs = []

        # unconditional model
        for data in dataloader:
            # check if labeled dataset
            if len(data) == 1:
                x, y = data[0], None
            else:
                x, y = data
                y = y.squeeze().to(self.device)
            x = x.to(self.device)
            logprobs.append(model.module.log_prob(x))
        logprobs = torch.cat(logprobs, dim=0).to(self.device)

        logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.var(0).sqrt() / math.sqrt(len(dataloader.dataset))
        output = 'Evaluate ' + (epoch != None)*'(epoch {}) -- '.format(epoch) + 'logp(x) = {:.3f} +/- {:.3f}'.format(logprob_mean, logprob_std)
        print(output)
        results_file = os.path.join(args.out_dir, 'results.txt')
        print(output, file=open(results_file, 'a'))
        return logprob_mean, logprob_std

    def sample(self, args):
        model = self.get_model()
        print('restoring checkpoint from {}'.format(args.restore_file))
        state = torch.load(os.path.join(args.restore_file, "best_model_checkpoint.pt"), map_location=self.device)
        model = model.to(self.config.device)
        model.load_state_dict(state['model_state'], strict=True)
        model = torch.nn.DataParallel(model)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(state)
            ema_helper.ema(model)
        else:
            ema_helper = None

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.generate_samples:
            self.generate_samples(model)
        elif self.args.encode_z:
            self.encode_z(self.args, model)
        elif self.args.fair_generate:
            self.fair_generate(self.args, self.config, model)
        else:
            raise NotImplementedError("Sampling procedure not defined")

    @staticmethod
    def plot_density(teacher, ax, ranges):
        (xmin, xmax), (ymin, ymax) = ranges
        # sample uniform grid
        n = 200
        xx1 = torch.linspace(xmin, xmax, n)
        xx2 = torch.linspace(ymin, ymax, n)
        xx, yy = torch.meshgrid(xx1, xx2)
        xy = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze()

        # run uniform grid through model and plot
        # density = dist.log_prob(xy).exp()
        # i think we should be plotting the log density ratio actually
        ratios = teacher.get_density_ratios(xy)
        ax.contour(xx, yy, ratios.view(n,n).data.numpy())

        # format
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([xmin, xmax])
        ax.set_yticks([ymin, ymax])

    @staticmethod
    def plot_dist_sample(data, ax, ranges):
        ax.scatter(data[:,0].data.numpy(), data[:,1].data.numpy(), s=10)
        # format
        (xmin, xmax), (ymin, ymax) = ranges
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([xmin, xmax])
        ax.set_yticks([ymin, ymax])

    @staticmethod
    def plot_density_ratios(data, teacher, ax, name):
        log_ratios = teacher.get_density_ratios(data, log=True)
        ax.scatter(data[:,0].data.cpu().numpy(), data[:,1].data.cpu().numpy(), s=10, c=log_ratios.data.cpu().numpy(), label=name)
        # ax.colorbar()
        ax.set_title('Log ratios for {}'.format(name))
        sns.despine()
        plt.tight_layout()

    def get_density_ratios(self, x, log=False):
        from torch.distributions import Normal
        p_mu = self.config.data.mus[0]
        q_mu = self.config.data.mus[1]

        log_p = Normal(p_mu, 1).log_prob(x).sum(-1)
        log_q = Normal(q_mu, 1).log_prob(x).sum(-1)
        log_r = log_q - log_p  # ref/bias
        if log:
            r = log_r
        else:
            r = torch.exp(log_r)

        return r

    def plot_sample_and_density(self, args, model, teacher, step=None):
        model.eval()

        data = torch.cat([teacher.datasets[0].data, teacher.datasets[1].data])
        data = data.to(self.device)
        u, _ = model.module.forward(data)

        # can you actually get samples?
        ux = model.module.base_dist.sample((len(data),))
        samples, _ = model.module.inverse(ux)

        # get ratios first
        xs = data.detach().cpu()
        log_ratios = self.get_density_ratios(xs, log=True)
        log_ratios_samples = self.get_density_ratios(samples.detach().cpu(), log=True)
        rs = torch.cat([log_ratios, log_ratios_samples])
        min_, max_ = rs.min(), rs.max()

        # TODO: i think it'd be nice to look at densities learned by the flow too!

        # plotting
        plt.figure(figsize=(9,4))
        plt.subplot(1,2,1)
        plt.scatter(xs[:,0].data.cpu().numpy(), xs[:,1].data.cpu().numpy(), s=10, c=log_ratios.data.cpu().numpy(), label='data')
        plt.clim(min_, max_)
        plt.title('Data: log r(x)', fontsize=15)
        sns.despine()

        plt.subplot(1,2,2)
        xs = samples.detach().cpu()
        plt.scatter(xs[:,0].data.cpu().numpy(), xs[:,1].data.cpu().numpy(), s=10, c=log_ratios_samples.data.cpu().numpy(), label='samples')
        plt.clim(min_, max_)
        plt.title('Samples: log r(x)', fontsize=15)
        sns.despine()

        # format and save
        plt.colorbar()
        matplotlib.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.out_dir, 'sample' + (step != None)*'_epoch_{}'.format(step) + '.png'))
        plt.close()

    def sample(self, args):
        model = self.get_model()
        print('restoring checkpoint from {}'.format(args.restore_file))
        state = torch.load(os.path.join(args.restore_file, "best_model_checkpoint.pt"), map_location=self.device)
        model = model.to(self.config.device)
        model.load_state_dict(state['model_state'], strict=True)
        model = torch.nn.DataParallel(model)

        # no ema
        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.generate_samples:
            self.generate_samples(model)
        elif self.args.encode_z:
            self.encode_z(self.args, model)
        elif self.args.fair_generate:
            self.fair_generate(self.args, self.config, model)
        else:
            raise NotImplementedError("Sampling procedure not defined")

    @torch.no_grad()
    def encode_z(self, args, model):
        model_name = self.config.model.name
        model = model.to(self.device)
        model.eval()

        # data
        dataset = self.config.data.dataset
        train_dataloader, val_dataloader, test_dataloader = fetch_dataloaders(dataset, self.config.training.batch_size, self.device, args, self.config, self.config.data.flip_toy_var_order)

        # separate out biased mnist and ref mnist
        train_biased, train_ref = train_dataloader
        val_biased, val_ref = val_dataloader
        test_biased, test_ref = test_dataloader
        # HACK for GMM and MI encodings
        if dataset == 'GMM_flow':
            dataset = 'GMM'
        if dataset == 'MI_flow':
            dataset = 'MI'
        save_folder = os.path.join(self.config.training.data_dir, 'encodings', dataset)
        os.makedirs(save_folder, exist_ok=True)
        print(f'encoding dataset {dataset}')

        for split, loader in zip(('train', 'val', 'test'), (train_biased, val_biased, test_biased)):
            save_path = os.path.join(save_folder, f'{model_name}_{split}_biased_z_perc{self.config.data.perc}')
            print('saving encodings in: {}'.format(save_path))
            
            ys = []
            zs = []
            d_ys = []
            for i, (x,y) in enumerate(loader):
                x = x.to(self.device)
                z, _ = model(x.squeeze())
                zs.append(z.detach().cpu().numpy())
                ys.append(y.detach().numpy())
                d_ys.append(np.zeros_like(y))
            zs = np.vstack(zs)
            ys = np.hstack(ys)
            d_ys = np.hstack(d_ys)
            np.savez(save_path, **{'z': zs, 'y': ys, 'd_y': d_ys})
            print(f'Encoding of biased {split} set completed.')

        for split, loader in zip(('train', 'val', 'test'), (train_ref, val_ref, test_ref)):
            save_path = os.path.join(save_folder, f'{model_name}_{split}_ref_z_perc{self.config.data.perc}')
            print('saving encodings in: {}'.format(save_path))
            ys = []
            zs = []
            d_ys = []
            for i, (x,y) in enumerate(loader):
                x = x.to(self.device)
                z, _ = model(x.squeeze())
                zs.append(z.detach().cpu().numpy())
                ys.append(y.detach().numpy())
                d_ys.append(np.ones_like(y))
            zs = np.vstack(zs)
            ys = np.hstack(ys)
            d_ys = np.hstack(d_ys)
            np.savez(save_path, **{'z': zs, 'y': ys, 'd_y': d_ys})
            print(f'Encoding of ref {split} set completed.')
        # done
        print('Done encoding all x')


def dict2namespace(config):
    import argparse
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace