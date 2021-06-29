import os
import sys
import time
import math

import copy
import logging
from pprint import pprint
from tqdm import tqdm
import numpy as np
import yaml

import flows.utils as utils
from datasets.data import fetch_dataloaders
from flows.models.maf import *
from flows.models.ema import EMAHelper
from classification.models.resnet import ResnetClassifier
from classification.models.mlp import MLPClassifier, MLPClassifierv2, TREMLPClassifier

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
        if self.args.fair_generate:
            self.dre_clf, name = self.load_classifier(args, self.args.dre_clf_ckpt, attr=False)
            self.dre_clf_name = name
        if self.args.sample and self.args.attr_clf_ckpt is not None:
            self.attr_clf, name = self.load_classifier(args, self.args.attr_clf_ckpt)
            self.attr_clf_name = name

    def get_preds(self, logits, model_name):
        with torch.no_grad():
            if model_name == 'mlp':
                y_preds = torch.round(torch.sigmoid(logits))
            else:
                probs = F.softmax(logits, dim=1)
                _, y_preds = torch.max(probs, 1)
        return y_preds

    def get_model(self):
        # TODO: do something about these args
        if self.config.model.name == 'maf':
            model = MAF(self.config.model.n_blocks, self.config.model.input_size, self.config.model.hidden_size, self.config.model.n_hidden, None, self.config.model.activation_fn, self.config.model.input_order, batch_norm=not self.config.model.no_batch_norm)
        else:
            raise ValueError('Unrecognized model.')
        return model

    def load_classifier(self, args, ckpt_path, attr=True):
        config_path = os.path.dirname(os.path.dirname(ckpt_path.rstrip('/')))
        with open(os.path.join(config_path, 'config.yaml')) as f:
            config = yaml.safe_load(f)
        config = dict2namespace(config)

        if args.tre and not attr:
            self.m = config.tre.m
            self.p = config.tre.p
            self.interp = config.tre.interp
            if config.model.name == 'mlp':
                clf = TREMLPClassifier(config)
            else:
                raise NotImplementedError(f'Classification model [{config.model.name}] not implemented.')

        else:
            if config.model.name == 'resnet':
                clf = ResnetClassifier(args)
            elif config.model.name == 'mlp':
                clf = MLPClassifierv2(config)
            else:
                raise NotImplementedError(f'Classification model [{config.model.name}] not implemented.')
            
        assert os.path.exists(ckpt_path)
        
        # load state dict
        state_dict = torch.load(ckpt_path)['state_dict']
        clf.load_state_dict(state_dict)
        clf = clf.to(self.device)
        return clf, config.model.name

    def interpolate(self, d1, d2):
        # d1 and d2 are batches of datasets
        if self.interp == 'linear':
            # the +1 is for [0, 1, ..., m]
            a_k = torch.true_divide(torch.arange(self.m + 1), self.m) ** self.p
            a_k = a_k.to(self.device)  # shapes
            x_k = [torch.sqrt(1 - a ** 2) * d1 + (a * d2) for a in a_k]
            x_k = torch.stack(x_k)
        else:
            raise NotImplementedError

        return x_k  # (m+1, batch_size, x_dim)

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
                    y = y.to(self.device)
                x = x.view(x.shape[0], -1).to(self.device)
                # loss = -model.module.log_prob(x, y if self.config.model.cond_label_size else None).mean(0)
                loss = -model.module.log_prob(x, y=None).mean(0)
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
            # save model only
            torch.save(
                model.state_dict(), os.path.join(
                    args.out_dir, 'model_state.pt'))
            # save best state
            if eval_logprob > best_eval_logprob:
                best_eval_logprob = eval_logprob
                print('saving model at epoch {}'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'model_state': model.module.state_dict(),
                    'optimizer_state': optimizer.state_dict()},
                    os.path.join(args.out_dir, 'best_model_checkpoint.pt'))
                # generate samples
                self.visualize(self.args, model, epoch)

    @torch.no_grad()
    def test(self, model, dataloader, epoch, args):
        model.eval()
        logprobs = []

        # unconditional model
        for data in dataloader:
            # x = data[0].view(data[0].shape[0], -1).to(self.device)
            # check if labeled dataset
            if len(data) == 1:
                x, y = data[0], None
            else:
                x, y = data
                y = y.to(self.device)
            x = x.to(self.device).view(x.shape[0], -1)
            log_px = model.module.log_prob(x)
            logprobs.append(log_px)
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

    def sample_fid(self, model):
        pass

    @torch.no_grad()
    def visualize(self, args, model, step, n_row=10):
        model.eval()

        u = model.module.base_dist.sample((n_row**2, self.config.model.n_components)).squeeze()
        samples, _ = model.module.inverse(u)
        while torch.any(torch.isnan(samples)):
            print('nan entries exist! resampling...')
            u = model.module.base_dist.sample(
            (n_row**2, self.config.model.n_components)).squeeze()
            samples, _ = model.module.inverse(u)
        log_probs = model.module.log_prob(samples).sort(0)[1].flip(0)
        samples = samples[log_probs]

        # convert and save images
        samples = samples.view((samples.shape[0], self.config.data.channels, self.config.data.image_size, self.config.data.image_size))
        samples = torch.sigmoid(samples)
        samples = torch.clamp(samples, 0., 1.)
        if step % args.save_freq  == 0:
            filename = 'generated_samples' + (step != None)*'_epoch_{}'.format(step) + '.png'
            save_image(samples, os.path.join(args.out_dir, filename), nrow=n_row, normalize=True)
        # log generations to wandb
        wandb.log({"samples" : [wandb.Image(i) for i in samples[0:40]]})

    @torch.no_grad()
    def generate_samples(self, model, n_row=10):
        all_samples = []
        preds = []
        model.eval()
        if self.args.attr_clf_ckpt is not None:
            self.attr_clf.eval() 

        print('generating {} samples in batches of 100...'.format(self.config.sampling.n_samples))
        n_batches = int(self.config.sampling.n_samples // 100)
        for n in range(n_batches):
            if (n % 10 == 0) and (n > 0):
                print('on iter {}/{}'.format(n, n_batches))
            u = model.module.base_dist.sample(
                (100, self.config.model.n_components)).squeeze()
            samples, _ = model.module.inverse(u)
            while torch.any(torch.isnan(samples)):
                print('nan entries exist! resampling...')
                u = model.module.base_dist.sample(
                (100, self.config.model.n_components)).squeeze()
                samples, _ = model.module.inverse(u)
            print('generating samples...')
            # sort by log_prob; take argsort idxs; flip high to low
            log_probs = model.module.log_prob(samples).sort(0)[1].flip(0)
            samples = samples[log_probs]
            samples = samples.view((samples.shape[0], self.config.data.channels, self.config.data.image_size, self.config.data.image_size))
            samples = torch.sigmoid(samples)
            samples = torch.clamp(samples, 0., 1.)  # check if we want to multiply by 255 and transpose if we're gonna do metric stuff on here

            # get classifier predictions
            logits, probas = self.attr_clf(samples.view(100, 1, -1))
            pred = self.get_preds(logits.squeeze(), self.attr_clf_name)
            
            # save things
            preds.append(pred.detach().cpu().numpy())
            all_samples.append(samples.detach().cpu().numpy())
        all_samples = np.vstack(all_samples)
        preds = np.hstack(preds)
        prop_ones = np.sum(preds)/len(preds)
        print('prop of 1s:', prop_ones)

        fair_disc_l2, fair_disc_l1, fair_disc_kl = utils.fairness_discrepancy(preds, 2)
        np.savez(os.path.join(self.args.out_dir, 'samples'), **{'x': all_samples})
        np.savez(os.path.join(self.args.out_dir, 'metrics'), 
            **{'preds': preds,
            'l2_fair_disc': fair_disc_l2,
            'prop_ones': prop_ones
            })
        # maybe just save some samples just for visualizations?
        filename = 'samples'+ '.png'
        save_image(torch.from_numpy(all_samples[:100]), os.path.join(self.args.out_dir, filename), nrow=n_row, normalize=True)

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
        save_folder = os.path.join(self.config.training.data_dir, 'encodings', dataset)
        os.makedirs(save_folder, exist_ok=True)
        print(f'encoding dataset {dataset}')

        for split, loader in zip(('train', 'val', 'test'), (train_biased, val_biased, test_biased)):
            # data_type = 'mnist' if not self.config.data.subset else 'mnist_subset_same_bkgd'
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

    def get_mixture_ratios(self, ratios):
        perc = self.config.data.perc
        if perc == 1.0:
            new_ratios = ratios / (1/2 * (ratios + 1))
        elif perc == 0.5:
            new_ratios = ratios / (2/3 * (2*ratios + 1))
        elif perc == 0.25:
            new_ratios = ratios / (1/5 * (4 * ratios + 1))
        elif perc == 0.1:
            new_ratios = ratios / (1/11 * (10*ratios + 1))
        
        return new_ratios
    @torch.no_grad()
    def fair_generate(self, args, config, model, n_row=10):
        from torch.distributions import Categorical

        all_samples = []
        preds = []

        model.eval()
        self.dre_clf.eval()
        if self.args.attr_clf_ckpt is not None:
            self.attr_clf.eval()

        # alpha = config.dre.alpha
        alpha = 0
        print('flattening importance weights by alpha={}'.format(alpha))

        # print('generating {} samples in batches of 1000...'.format(args.n_samples))
        n_sir = config.sampling.n_sir
        print('running SIR sampling... we are going to generate {} samples total.'.format(n_sir))
             
        for n in range(n_sir):
            if (n % 5 == 0) and (n > 0):
                print('on iter {}/{}'.format(n, n_sir))
            u = model.module.base_dist.sample((100, self.config.model.n_components)).squeeze().to(self.device)

            if not self.args.dre_x and not self.args.tre:
                if self.dre_clf_name == 'mlp':
                    logits, probas = self.dre_clf(u.view(100, 1, -1))
                    logits = logits.squeeze()
                    log_ratios = utils.logsumexp_1p(-logits) - utils.logsumexp_1p(logits)
                    ratios = torch.exp(log_ratios)
                else:
                    logits, probas = self.dre_clf(u.view(100, 1, 28, 28))
                    logits = logits.squeeze()
                    ratios = (probas[:, 1]/probas[:, 0])
                
                ratios = self.get_mixture_ratios(ratios)
                
                if alpha > 0:
                    ratios = ratios**(alpha)
                r_probs = ratios/ratios.sum()
                sir_j = Categorical(r_probs).sample().item()
                
                samples, _ = model.module.inverse(u[sir_j].unsqueeze(0))
                while torch.any(torch.isnan(samples)):
                    print('nans found! resampling...')
                    u = model.module.base_dist.sample((100, self.config.model.n_components)).squeeze().to(self.device)
                    if self.dre_clf_name == 'mlp':
                        logits, probas = self.dre_clf(u.view(100, 1, -1))
                        logits = logits.squeeze()
                        log_ratios = utils.logsumexp_1p(-logits) - utils.logsumexp_1p(logits)
                        ratios = torch.exp(log_ratios)
                    else:
                        logits, probas = self.dre_clf(u.view(100, 1, 28, 28))
                        logits = logits.squeeze()
                        ratios = (probas[:, 1]/probas[:, 0])
                    ratios = self.get_mixture_ratios(ratios)
                    if alpha > 0:
                        ratios = ratios**(alpha)
                    r_probs = ratios/ratios.sum()
                    sir_j = Categorical(r_probs).sample().item()
                    
                    samples, _ = model.module.inverse(u[sir_j].unsqueeze(0))
            else:
                samples, _ = model.module.inverse(u)
                while torch.any(torch.isnan(samples)):
                    print('nans found! resampling...')
                    u = model.module.base_dist.sample((100, self.config.model.n_components)).squeeze().to(self.device)                    
                    samples, _ = model.module.inverse(u)
                    
                # logits, probas = self.dre_clf(samples.view(100, 1, -1))
                # logits = logits.squeeze()
                # # ratios = (probas[:, 1]/probas[:, 0])
                # log_ratios = utils.logsumexp_1p(-logits) - utils.logsumexp_1p(logits)
                # ratios = torch.exp(log_ratios)
                # if alpha > 0:
                #     ratios = ratios**(alpha)
                # r_probs = ratios/ratios.sum()

            log_probs = model.module.log_prob(samples).sort(0)[1].flip(0)  # sort by log_prob; take argsort idxs; flip high to low
            samples = samples[log_probs]
            samples = samples.view((samples.shape[0], self.config.data.channels, self.config.data.image_size, self.config.data.image_size))
            samples = torch.sigmoid(samples)
            samples = torch.clamp(samples, 0., 1.)
            
            if self.args.tre:
                if self.dre_clf_name == 'mlp':
                    # bridges = self.interpolate(samples)
                    logits_list, probas = self.dre_clf(samples.view(100, 1, -1))
                    ratios = []
                    for logits in logits_list:
                        logits = logits.squeeze()
                        log_ratios = utils.logsumexp_1p(-logits) - utils.logsumexp_1p(logits)
                        ratios.append(torch.exp(log_ratios))
                else:
                    raise NotImplementedError
                
                # if alpha > 0:
                #     ratios = ratios**(alpha)
                # ratios = self.get_mixture_ratios(ratios)
                # r_probs = ratios/ratios.sum()
                ratios = torch.stack(ratios)
                # print('ratios.shape: ', ratios.shape)
                r_probs = torch.prod(ratios, axis=0)
                sir_j = Categorical(r_probs).sample().item()
                samples = samples[sir_j].unsqueeze(0)

            elif self.args.dre_x:
                if self.dre_clf_name == 'mlp':
                    print('samples.shape: ', samples.shape)
                    logits, probas = self.dre_clf(samples.view(100, 1, -1))
                    logits = logits.squeeze()
                    log_ratios = utils.logsumexp_1p(-logits) - utils.logsumexp_1p(logits)
                    ratios = torch.exp(log_ratios)
                else:
                    logits, probas = self.dre_clf(samples.view(100, 1, 28, 28))
                    logits = logits.squeeze()
                    ratios = (probas[:, 1]/probas[:, 0])
                if alpha > 0:
                    ratios = ratios**(alpha)
                ratios = self.get_mixture_ratios(ratios)
                r_probs = ratios/ratios.sum()
                sir_j = Categorical(r_probs).sample().item()
                samples = samples[sir_j].unsqueeze(0)
 
            # get classifier predictions
            logits, probas = self.attr_clf(samples.view(samples.shape[0], 1, -1))
            pred = self.get_preds(logits.squeeze(), self.attr_clf_name)

            # save things
            preds.append(pred.detach().cpu().numpy())
            all_samples.append(samples.detach().cpu().numpy())
        all_samples = np.vstack(all_samples)
        preds = np.hstack(preds)
        # else:
        #     # regular sampling
        #     u = model.module.base_dist.sample((n_sir, self.config.model.n_components)).squeeze().to(self.device)
        #     samples, _ = model.module.inverse(u)
        #     log_probs = model.module.log_prob(samples).sort(0)[1].flip(0)  # sort by log_prob; take argsort idxs; flip high to low
        #     samples = samples[log_probs]
        #     samples = samples.view((samples.shape[0], self.config.data.channels, self.config.data.image_size, self.config.data.image_size))
        #     samples = torch.sigmoid(samples)
        #     samples = torch.clamp(samples, 0., 1.)
        #     # get classifier predictions
        #     if self.args.attr_clf_ckpt is not None:
        #         logits, probas = self.attr_clf(samples.view(len(samples), -1))
        #     _, pred = torch.max(probas, 1)

        #     all_samples = samples.detach().cpu().numpy()
        #     preds = pred.detach().cpu().numpy()
        # check metrics
        fair_disc_l2, fair_disc_l1, fair_disc_kl = utils.fairness_discrepancy(preds, 2)
        prop_ones = np.sum(preds)/len(preds)
        print('prop of 1s:', prop_ones)
        print('L2 fairness discrepancy is: {}'.format(fair_disc_l2))
        np.savez(os.path.join(args.out_dir, 'samples'), **{'x': all_samples})
        np.savez(os.path.join(args.out_dir, 'metrics'), 
            **{
            'preds': preds,
            'l2_fair_disc': fair_disc_l2,
            'prop_ones': prop_ones
            })
        # maybe just save some samples?
        filename = 'fair_samples_sir_alpha{}'.format(self.config.dre.alpha) + '.png'
        save_image(torch.from_numpy(all_samples[:100]), os.path.join(args.out_dir, filename), nrow=n_row, normalize=True)

    @torch.no_grad()
    def fair_evaluate(self, args, config, model, n_row=10):
        """
        this function is intended for computing expectations with respect to an importance-weighted distribution (different from generating samples wrt q)
        """
        from utils import logsumexp_1p

        all_samples = []
        preds = []

        model.eval()
        self.dre_clf.eval()

        # TODO: incorporate the statistic of interest
        g_x = None

        # sample z from your flow (TODO: fix number of samples)
        u = model.module.base_dist.sample((100, self.config.model.n_components)).squeeze().to(self.device)
        logits, probas = self.dre_clf(u.view(100, 1, 28, 28))
        
        # TODO: should make sure that logits are in the right form
        log_ratios = utils.logsumexp_1p(-logits) - utils.logsumexp_1p(logits)
        ratios = torch.exp(log_ratios)
        iw = self.get_mixture_ratios(ratios)

        # TODO: get statistic of interest
        # E_pf(x) [(2/(1 + r(x))) * g(x)]
        return (g_x * iw).mean()


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
