import os
import sys
import time

import copy
import logging
from pprint import pprint
from tqdm import tqdm
import numpy as np
import yaml

import flows.utils as utils
from datasets.toy import GMM
from flows.models.maf import *
from flows.models.ema import EMAHelper
from classification.models.resnet import ResnetClassifier
from classification.models.mlp import MLPClassifier

from flows.functions import get_optimizer
from flows.functions.ckpt_util import get_ckpt_path
from flows.functions.utils import (
    get_ratio_estimates,
    fairness_discrepancy,
    classify_examples
)

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
        self.teacher = GMM(args, config)
        
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
        teacher = GMM(args, config)
        
        model = self.get_model()
        model = model.to(self.device)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_step, step = 0, 0
        best_eval_logprob = float('-inf')
        model = torch.nn.DataParallel(model)

        for step in range(self.config.training.n_steps):
            model.train()
            data_start = time.time()
            data_time = 0
            data = teacher.sample(self.config.training.batch_size)

            # check if labeled dataset
            if len(data) == 1:
                x, y = data[0], None
            else:
                x, y = data
                y = y.to(self.device)
            x = x.view(x.shape[0], -1).to(self.device)
            loss = -model.module.log_prob(x, y=None).mean(0)
            # get summary to log to wandb
            summary = dict(
                train_loss=loss.item(),
                step=step,
            )
            wandb.log(summary)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % self.config.training.log_interval == 0 and step > 0:
                print('step {:3d} / {}; loss {:.4f}'.format(
                    step, start_step + self.config.training.n_steps, loss.item()))

                # now evaluate and save metrics/checkpoints
                eval_logprob, _ = self.test(model, teacher)
                # get summary to log to wandb
                summary = dict(
                    test_logp=eval_logprob.item(),
                    step=step,
                )
                wandb.log(summary)
                # save training checkpoint
                torch.save({
                    'step': step,
                    'model_state': model.module.state_dict(),
                    'optimizer_state': optimizer.state_dict()},
                    os.path.join(args.out_dir, 'model_checkpoint.pt'))
                # save best state
                if eval_logprob > best_eval_logprob:
                    best_eval_logprob = eval_logprob
                    torch.save({
                        'step': step,
                        'model_state': model.module.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                        os.path.join(args.out_dir, 'best_model_checkpoint.pt'))
                    # generate samples
                    self.plot_sample_and_density(model, teacher, step=step)

    @torch.no_grad()
    def test(self, model, teacher):
        model.eval()
        logprobs = []
        len_d = int(100 * self.config.training.eval_steps)

        # unconditional model
        for step in range(self.config.training.eval_steps):
            data = teacher.sample(100, fair=True)  # validation
            x = data[0].view(data[0].shape[0], -1).to(self.device)
            logprobs.append(model.module.log_prob(x))
        logprobs = torch.cat(logprobs, dim=0).to(self.device)

        logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.var(0).sqrt() / math.sqrt(len_d)
        output = 'Evaluate ' + (step != None)*'(step {}) -- '.format(step) + 'logp(x) = {:.3f} +/- {:.3f}'.format(logprob_mean, logprob_std)
        print(output)
        results_file = os.path.join(self.args.out_dir, 'results.txt')
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

    def plot_sample_and_density(self, model, teacher, ranges_density=[[-5,20],[-10,10]], ranges_sample=[[-4,4],[-4,4]], step=None):
        model.eval()
        # fig, axs = plt.subplots(1, 2, figsize=(6,3))

        # sample target distribution and pass through model
        data, _ = teacher.sample(2000)
        data = data.to(self.device)
        u, _ = model.module.forward(data)

        # can you actually get samples?
        ux = model.module.base_dist.sample((2000,))
        samples, _ = model.module.inverse(ux)

        # get ratios first
        xs = data.detach().cpu()
        log_ratios = teacher.get_density_ratios(xs, log=True)
        log_ratios_samples = teacher.get_density_ratios(samples.detach().cpu(), log=True)
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