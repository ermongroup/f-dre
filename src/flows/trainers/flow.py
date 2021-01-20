import os
import sys
import time

import copy
import logging
from pprint import pprint
from tqdm import tqdm
import numpy as np
import yaml

import utils
from src.flows.data import fetch_dataloaders
from src.flows.models.maf import *
from src.flows.models.ema import EMAHelper
from src.classification.models.resnet import ResnetClassifier
from src.classification.models.mlp import MLPClassifier

from functions import get_optimizer
from functions.ckpt_util import get_ckpt_path
from functions.utils import (
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# for pre-trained attribute classifiers
CHECKPOINT_DIR = '/atlas/u/madeline/multi-fairgen/src/classification/checkpoints'
ATTR_CLFS = {
    'digit': os.path.join(
        CHECKPOINT_DIR, 'digits_attr_clf', 'model_best.pth'),
    'background': os.path.join(
        CHECKPOINT_DIR,'background_attr_clf', 'model_best.pth')     
}


class Flow(object):
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
        if self.args.reweight:
            self.dre_clf = self.load_classifier(self.args, 'resnet')
        self.attr_clf = self.load_classifier(self.args, 'mlp')

    def get_model(self):
        # TODO: do something about these args
        if self.config.model.name == 'maf':
            model = MAF(self.config.model.n_blocks, self.config.model.input_size, self.config.model.hidden_size, self.config.model.n_hidden, self.config.model.cond_label_size, self.config.model.activation_fn, self.config.model.input_order, batch_norm=not self.config.model.no_batch_norm)
        else:
            raise ValueError('Unrecognized model.')
        return model

    def load_classifier(self, args, name):
        if name == 'resnet':
            clf = ResnetClassifier(args)
            ckpt_path = args.clf_ckpt
        else:
            # with open(os.path.join('src/classification/configs', args.config), 'r') as f:
            # TODO: HACK
            with open(os.path.join('src/classification/configs/mnist/attr_bkgd.yaml')) as f:
                config = yaml.load(f)
            config = dict2namespace(config)
            clf = MLPClassifier(config)
            # mnist_attr = 'digit' if args.subset else 'background'
            mnist_attr = 'background'
            ckpt_path = ATTR_CLFS[mnist_attr]
        assert os.path.exists(ckpt_path)
        print('loading pre-trained DRE classifier checkpoint from {}'.format(ckpt_path))

        # load state dict
        state_dict = torch.load(ckpt_path)['state_dict']
        clf.load_state_dict(state_dict)
        clf = clf.to(self.device)
        return clf

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        train_dataloader, val_dataloader, test_dataloader = fetch_dataloaders(self.config.data.dataset, self.config.training.batch_size, self.device, self.args, self.config, self.config.data.flip_toy_var_order)
        
        model = self.get_model()
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

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
                    y = y.to(self.device)
                x = x.view(x.shape[0], -1).to(self.device)
                # loss = -model.module.log_prob(x, y if self.config.model.cond_label_size else None).mean(0)
                loss = -model.module.log_prob(x, y=None).mean(0)

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
            # save training checkpoint
            torch.save({
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer_state': optimizer.state_dict()},
                os.path.join(args.output_dir, 'model_checkpoint.pt'))
            # save model only
            torch.save(
                model.state_dict(), os.path.join(
                    args.output_dir, 'model_state.pt'))
            # save best state
            if eval_logprob > best_eval_logprob:
                best_eval_logprob = eval_logprob
                torch.save({
                    'epoch': epoch,
                    'model_state': model.module.state_dict(),
                    'optimizer_state': optimizer.state_dict()},
                    os.path.join(args.output_dir, 'best_model_checkpoint.pt'))
                # generate samples
                self.visualize(self.args, model, epoch)

    @torch.no_grad()
    def test(self, model, dataloader, epoch, args):
        model.eval()
        logprobs = []

        # unconditional model
        for data in dataloader:
            x = data[0].view(data[0].shape[0], -1).to(self.device)
            logprobs.append(model.module.log_prob(x))
        logprobs = torch.cat(logprobs, dim=0).to(self.device)

        logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.var(0).sqrt() / math.sqrt(len(dataloader.dataset))
        output = 'Evaluate ' + (epoch != None)*'(epoch {}) -- '.format(epoch) + 'logp(x) = {:.3f} +/- {:.3f}'.format(logprob_mean, logprob_std)
        print(output)
        results_file = os.path.join(args.output_dir, 'results.txt')
        print(output, file=open(results_file, 'a'))
        return logprob_mean, logprob_std

    def sample(self, args):
        model = self.get_model()
        print('restoring checkpoint from {}'.format(args.restore_file))
        state = torch.load(os.path.join(args.restore_file, "best_model_checkpoint.pt"), map_location=self.device)
        model = model.to(self.config.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state['model_state'], strict=True)

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
            self.fair_generate(self.args, model)
        else:
            raise NotImplementedError("Sampling procedure not defined")

    def sample_fid(self, model):
        pass

    @torch.no_grad()
    def visualize(self, args, model, step, n_row=10):
        model.eval()

        u = model.module.base_dist.sample((n_row**2, self.config.model.n_components)).squeeze()
        samples, _ = model.module.inverse(u)
        log_probs = model.module.log_prob(samples).sort(0)[1].flip(0)
        samples = samples[log_probs]

        # convert and save images
        samples = samples.view((samples.shape[0], self.config.data.channels, self.config.data.image_size, self.config.data.image_size))
        samples = torch.sigmoid(samples)
        samples = torch.clamp(samples, 0., 1.)
        filename = 'generated_samples' + (step != None)*'_epoch_{}'.format(step) + '.png'
        save_image(samples, os.path.join(args.output_dir, filename), nrow=n_row, normalize=True)

    @torch.no_grad()
    def generate_samples(self, model):
        all_samples = []
        preds = []
        model.eval()
        self.attr_clf.eval()

        print('generating {} samples in batches of 1000...'.format(self.config.sampling.n_samples))
        n_batches = int(self.config.sampling.n_samples // 1000)
        for n in range(n_batches):
            if (n % 10 == 0) and (n > 0):
                print('on iter {}/{}'.format(n, n_batches))
            u = model.module.base_dist.sample(
                (1000, self.config.model.n_components)).squeeze()
            samples, _ = model.module.inverse(u)
            # sort by log_prob; take argsort idxs; flip high to low
            log_probs = model.module.log_prob(samples).sort(0)[1].flip(0)
            samples = samples[log_probs]
            samples = samples.view((samples.shape[0], self.config.data.channels, self.config.data.image_size, self.config.data.image_size))
            samples = torch.sigmoid(samples)
            samples = torch.clamp(samples, 0., 1.)  # check if we want to multiply by 255 and transpose if we're gonna do metric stuff on here

            # get classifier predictions
            logits, probas = self.attr_clf(samples.view(len(samples), -1))
            _, pred = torch.max(probas, 1)

            # save things
            preds.append(pred.detach().cpu().numpy())
            all_samples.append(samples.detach().cpu().numpy())
        all_samples = np.vstack(all_samples)
        preds = np.hstack(preds)
        fair_disc_l2, fair_disc_l1, fair_disc_kl = utils.fairness_discrepancy(preds, 2)
        np.savez(os.path.join(args.output_dir, f'{self.config.data.dataset}_maf_perc{self.self.config.data.perc}', 'samples'), **{'x': all_samples})
        np.savez(os.path.join(args.output_dir, f'{self.config.data.dataset}_maf_perc{self.self.config.data.perc}', 'metrics'), 
            **{'preds': preds,
            'l2_fair_disc': fair_disc_l2,
            })
        # maybe just save some samples just for visualizations?
        filename = 'samples'+ '.png'
        save_image(all_samples[0:100], os.path.join(self.args.output_dir, filename), nrow=n_row, normalize=True)

    @torch.no_grad()
    def encode_z(self, args, model):
        model = model.to(device)
        model.eval()

        # separate out regular mnist and flipped mnist
        train_mnist, train_cmnist = train_loader
        val_mnist, val_cmnist = val_loader
        test_mnist, test_cmnist = test_loader

        save_folder = os.path.join(data_dir, dataset, f'{model_name}_encodings')
        os.makedirs(save_folder, exist_ok=True)

        for split, loader in zip(('train', 'val', 'test'), (train_mnist, val_mnist, test_mnist)):
            data_type = 'mnist' if not args.subset else 'mnist_subset'
            print('encoding data type {}'.format(data_type))
            if not os.path.exists(os.path.join(data_dir, 'encodings', data_type)):
                os.makedirs(os.path.join(data_dir, 'encodings', data_type))
            # TODO: make sure this lines up with previous trained flow!!!
            save_path = os.path.join(data_dir, 'encodings', data_type, '{}_{}_mnist_z_perc{}'.format(model_name, split, self.config.data.perc))
            ys = []
            zs = []
            d_ys = []
            for i, (x,y) in enumerate(loader):
                x = x.to(device)
                z, _ = model(x.squeeze())
                zs.append(z.detach().cpu().numpy())
                ys.append(y.detach().numpy())
                d_ys.append(np.zeros_like(y))
            zs = np.vstack(zs)
            ys = np.hstack(ys)
            d_ys = np.hstack(d_ys)
            np.savez(save_path, **{'z': zs, 'y': ys, 'd_y': d_ys})
            print(f'Encoding of mnist {split} set completed.')

        for split, loader in zip(('train', 'val', 'test'), (train_cmnist, val_cmnist, test_cmnist)):
            save_path = os.path.join(data_dir, 'encodings', data_type, '{}_{}_cmnist_z_perc{}'.format(model_name, split, self.config.data.perc))
            ys = []
            zs = []
            d_ys = []
            for i, (x,y) in enumerate(loader):
                x = x.to(device)
                z, _ = model(x.squeeze())
                zs.append(z.detach().cpu().numpy())
                ys.append(y.detach().numpy())
                d_ys.append(np.ones_like(y))
            zs = np.vstack(zs)
            ys = np.hstack(ys)
            d_ys = np.hstack(d_ys)
            np.savez(save_path, **{'z': zs, 'y': ys, 'd_y': d_ys})
            print(f'Encoding of flipped mnist {split} set completed.')
        # done
        print('Done encoding all x')

    @torch.no_grad()
    def fair_generate(self, args, model):
        from torch.distributions import Categorical

        all_samples = []
        preds = []

        model.eval()
        self.dre_clf.eval()
        self.attr_clf.eval()

        # print('generating {} samples in batches of 1000...'.format(args.n_samples))
        print('running SIR sampling...as a sanity check, we are only going to generate 20 samples total.')
        n_batches = int(self.config.sampling.n_samples // 1000)
        # for n in range(n_batches):
        
        for n in range(20):  # HACK
            if (n % 5 == 0) and (n > 0):
                print('on iter {}/{}'.format(n, 20))
            u = model.module.base_dist.sample((1000, self.config.data.n_components)).squeeze().to(device)
            
            # TODO: reweight the samples via dre_clf
            logits, probas = self.dre_clf(u.view(1000, 1, 28, 28))
            # print('flattening importance weights by alpha={}'.format(args.alpha))
            # ratios = ratios**(args.alpha)
            ratios = (probas[:, 1]/probas[:, 0])
            r_probs = ratios/ratios.sum()
            sir_j = Categorical(r_probs).sample().item()

            samples, _ = model.module.inverse(u[sir_j].unsqueeze(0))
            log_probs = model.module.log_prob(samples).sort(0)[1].flip(0)  # sort by log_prob; take argsort idxs; flip high to low
            samples = samples[log_probs]
            samples = samples.view((samples.shape[0], self.config.data.channels, self.config.data.image_size, self.config.data.image_size))
            samples = torch.sigmoid(samples)
            samples = torch.clamp(samples, 0., 1.)  # check if we want to multiply by 255 and transpose if we're gonna do metric stuff on here

            # get classifier predictions
            logits, probas = self.attr_clf(samples.view(len(samples), -1))
            _, pred = torch.max(probas, 1)

            # save things
            preds.append(pred.detach().cpu().numpy())
            all_samples.append(samples.detach().cpu().numpy())
        all_samples = np.vstack(all_samples)
        preds = np.hstack(preds)
        fair_disc_l2, fair_disc_l1, fair_disc_kl = utils.fairness_discrepancy(preds, 2)
        print('prop of 1s:', np.sum(preds)/len(preds))
        print('L2 fairness discrepancy is: {}'.format(fair_disc_l2))
        np.savez(os.path.join(args.output_dir, 'samples'), **{'x': all_samples})
        np.savez(os.path.join(args.output_dir, 'metrics'), 
            **{
            'preds': preds,
            'l2_fair_disc': fair_disc_l2,
            })
        # maybe just save some samples?
        filename = 'fair_samples_sir'.format(self.config.dre.alpha) + '.png'
        save_image(torch.from_numpy(all_samples), os.path.join(args.output_dir, filename), nrow=n_row, normalize=True)


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