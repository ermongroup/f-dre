import os
import sys

import copy
import logging
from pprint import pprint
from tqdm import tqdm
import numpy as np

import classification.utils as utils
from datasets.data import fetch_dataloaders
from classification.models.mlp import (
    MLPClassifier,
    MLPClassifierv2,
    TREMLPClassifier
)
from classification.models.flow_mlp import FlowClassifier
from classification.models.resnet import ResnetClassifier
from classification.models.networks import *
# HACK: this is going to be really annoying to bake Glow into this
from classification.models.glow import Glow
from classification.trainers.base import BaseTrainer
from sklearn.calibration import calibration_curve
from losses import joint_gen_disc_loss

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class TREClassifier(BaseTrainer):
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.m = self.config.tre.m
        self.p = self.config.tre.p

        # model and optimizer
        self.device, device_ids = self._prepare_device(
            self.config.training.ngpu)
        self.model = self.get_model().to(self.device)
        self.optimizer = self.get_optimizer(self.model.parameters())
        self.loss = self.get_loss()

        # get data
        self.train_dataloader, self.val_dataloader, self.test_dataloader = fetch_dataloaders(config.data.dataset, config.training.batch_size, self.device, args, config)

        # saving
        self.output_dir = args.out_dir
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # z-encoding
        # self.flow = self.load_glow()
        if self.args.encode_z:
            print('using flow to encode x into z-space...')

    def get_model(self):
        model = self.get_model_cls(self.config.model.name)
        model = model.to(self.config.device)

        return model

    def get_model_cls(self, name):
        if name == 'mlp':
            print('using TREMLPClassifier')
            model_cls = TREMLPClassifier
        else:
            print('Model {} not found!'.format(name))
            raise NotImplementedError
        # elif name == 'resnet':
        #     model_cls = ResnetClassifier
        # elif name == 'resnet20':
        #     model_cls = resnet20()
        #     return model_cls
        # elif name == 'flow_mlp':
        #     print('Training flow + mlp...')
        #     model_cls = FlowClassifier
        
        return model_cls(self.config)

    # def load_glow(self):
    #     import json
    #     output_folder = '/atlas/u/kechoi/Glow-PyTorch/glow/'
    #     # model_name = 'glow_model_250.pth'
    #     model_name = 'glow_affine_coupling.pt'

    #     with open(output_folder + 'hparams.json') as json_file:  
    #         hparams = json.load(json_file)
    #     image_shape = (32, 32, 3)  # HACK for CIFAR10
    #     num_classes = 10

    #     model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'], hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes, hparams['learn_top'], hparams['y_condition'])

    #     # load checkpoint
    #     model.load_state_dict(torch.load(output_folder + model_name))
    #     model.set_actnorm_init()
    #     model = model.to(self.config.device)
    #     return model

    def get_loss(self):
        if self.config.loss.name == 'bce':
            loss = F.binary_cross_entropy_with_logits
        elif self.config.loss.name == 'joint':
            loss = joint_gen_disc_loss
        else:
            loss = F.cross_entropy
        return loss

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(
                parameters, lr=self.config.optim.lr, 
                weight_decay=self.config.optim.weight_decay, 
                betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9, weight_decay=self.config.optim.weight_decay)
        else:
            raise NotImplementedError()

    def get_datasets(self):
        train, val, test = dsets.get_dataset(self.args, self.config)

        # create dataloaders
        train = data_utils.DataLoader(train, batch_size=self.config.training.batch_size//2, shuffle=True)
        val = data_utils.DataLoader(val, batch_size=self.config.training.batch_size//2, shuffle=False)
        test = data_utils.DataLoader(test, batch_size=self.config.training.batch_size//2, shuffle=False)

        return train, val, test

    def accuracy(self, logits, y):
        with torch.no_grad():
            if self.config.loss.name in ['bce', 'joint']:
                y_preds = torch.round(torch.sigmoid(logits))
            else:
                probs = F.softmax(logits, dim=1)
                _, y_preds = torch.max(probs, 1)
            acc = (y_preds == y).sum()
            acc = torch.true_divide(acc, len(y_preds)).cpu().numpy()        
        
        return acc

    def get_preds(self, logits):
        with torch.no_grad():
            if self.config.loss.name in ['bce', 'joint']:
                y_preds = torch.round(torch.sigmoid(logits))
            else:
                probs = F.softmax(logits, dim=1)
                _, y_preds = torch.max(probs, 1)
        return y_preds
    
    def interpolate(self, d1, d2):
        # d1 and d2 are batches of datasets
        if self.config.tre.interp == 'linear':
            # the +1 is for [0, 1, ..., m]
            a_k = torch.true_divide(torch.arange(self.m + 1), self.m) ** self.p
            a_k = a_k.to(self.device)  # shapes
            x_k = [torch.sqrt(1 - a ** 2) * d1 + (a * d2) for a in a_k]
            x_k = torch.stack(x_k)
        else:
            raise NotImplementedError

        return x_k  # (m+1, batch_size, x_dim)
    def train_epoch(self, epoch):
        # get meters ready
        loss_meter = utils.AverageMeter()
 
        num_pos_correct = np.zeros(self.m)
        num_pos_samples = np.zeros(self.m)
        num_neg_correct = np.zeros(self.m)
        num_neg_samples = np.zeros(self.m)

        # train classifier
        self.model.train()
        # self.flow.eval()
        data_tqdm = tqdm(iter(self.train_dataloader), leave=False, total=len(self.train_dataloader))

        for i, (z_ref, z_biased) in enumerate(data_tqdm):
            
            z = self.interpolate(z_ref.to(self.device), z_biased.to(self.device)) # (self.m + 1, batch_size, x_dim)
            # z = torch.cat([z_ref, z_biased]).to(self.device)
            y = torch.cat([torch.ones(z_ref.shape[0]), torch.zeros(z_biased.shape[0])]).to(self.device)

            # idx = torch.randperm(len(z))
            # z = z[idx].to(self.device).float()
            # y = y[idx].to(self.device).long()

            if self.args.encode_z:
                # TODO: preprocessing before glow
                z = utils.glow_preprocess(z)
                with torch.no_grad():
                    z, _, _ = self.flow(z)

            # random permutation of data
            # if 'mlp' in self.config.model.name:
            #     z = z.view(len(z), -1)
            # else:
            #     z = z.view(len(z), self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
            
            # NOTE: here, biased (y=0) and reference (y=1)
            logits_list, probas = self.model(z)
            # print('logits_list.shape: ', logits_list.shape)
            if self.config.loss.name == 'joint':
                loss_list = [self.loss(self.model, z, logits, y, self.config.loss.alpha) for logits in logits_list]
            else:
                loss_list = [self.loss(logits.squeeze(), y.float()) for logits in logits_list]
            
            # average across bridges
            loss = sum(loss_list)/len(loss_list)
            loss_meter.update(loss)

            # check accuracy
            acc_list = []
            for i, logits in enumerate(logits_list):
                y_preds = self.get_preds(logits.squeeze())
                num_pos_samples[i] += y.sum()
                num_neg_samples[i] += y.size(0) - y.sum()
                num_pos_correct[i] += (y_preds[y == 1] == y[y == 1]).sum()
                num_neg_correct[i] += (y_preds[y == 0] == y[y == 0]).sum()
                acc = (num_pos_correct[i] / num_pos_samples[i] + num_neg_correct[i] / num_neg_samples[i]) / 2
                acc_list.append(acc)

            avg_acc = sum(acc_list)/len(acc_list)
            # gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # get summary
            summary = dict(avg_loss=loss.item(), clf_acc=avg_acc)

            if (i + 1) % self.config.training.iter_log == 0:
                summary.update(
                    dict(avg_loss=np.round(loss.float().item(), 3),
                        clf_acc=avg_acc))
                print()
                pprint(summary)

            # pbar
            desc = 'loss: {}'.format(loss.item())
            data_tqdm.set_description(desc)
            data_tqdm.update(z.shape[0])
        # end of training epoch
        avg_acc_list = (num_pos_correct / num_pos_samples + num_neg_correct / num_neg_samples) / 2
        print('avg_acc_list: ', avg_acc_list)
        avg_acc = sum(avg_acc_list)/len(avg_acc_list)
        print()
        print('Completed epoch {}: train loss: {}, train acc: {}'.format(
            epoch, 
            np.round(loss_meter.avg.detach().cpu().numpy(), 3), 
            avg_acc))
        summary.update(dict(
            avg_loss=loss_meter.avg,
            avg_acc=avg_acc))
        # pprint(summary)

        return loss_meter.avg, avg_acc

    def train(self):
        best = False
        best_loss = sys.maxsize
        best_acc = -sys.maxsize
        best_epoch = 1
        tr_loss_db = np.zeros(self.config.training.n_epochs)
        test_loss_db = np.zeros(self.config.training.n_epochs)
        tr_acc_db = np.zeros(self.config.training.n_epochs)
        test_acc_db = np.zeros(self.config.training.n_epochs)

        # adjust learning rate as you go
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            self.config.training.n_epochs, 
            eta_min=0, 
            last_epoch=-1, 
            verbose=False
        )

        for epoch in range(1, self.config.training.n_epochs+1):
            print('training epoch {}'.format(epoch))
            tr_loss, tr_acc = self.train_epoch(epoch)
            val_loss, val_acc, val_labels, val_probs, val_ratios, val_data = self.test(self.val_dataloader, 'val')
            
            # evaluate on test (i just added this in)
            test_loss, test_acc, test_labels, test_probs, test_ratios, test_data = self.test(self.test_dataloader, 'test')
            scheduler.step()
            
            # check performance on validation set
            if val_loss < best_loss:
            # if val_acc > best_acc:
                print('saving best model..')
                best_acc = val_acc
                best_loss = val_loss
                best_epoch = epoch
                best = True
                for i in range(self.m):
                    self.clf_diagnostics(val_labels, val_probs[i], val_ratios[i], val_data[i], split='val', bridge_num=i)
                if self.config.model.name == 'flow_mlp':
                    self.flow_diagnostics(step=epoch, n_row=10)
            else:
                best = False
            self._save_checkpoint(epoch, save_best=best)
            
            # save metrics (TODO: test is actually validation here)
            tr_loss_db[epoch-1] = tr_loss
            test_loss_db[epoch-1] = val_loss
            tr_acc_db[epoch - 1] = tr_acc
            test_acc_db[epoch - 1] = val_acc
        # evaluate on test
        test_loss, test_acc, test_labels, test_probs, test_ratios, test_data = self.test(self.test_dataloader, 'test')
        
        # TODO: save metrics
        self.plot_train_test_curves(tr_loss_db, test_loss_db)
        self.plot_train_test_curves(tr_acc_db, test_acc_db, metric='Accuracy', title='train_curve_acc')
        print('Completed training! Best performance at epoch {}, loss: {}, acc: {}'.format(best_epoch, best_loss, best_acc))
        # TODO: also save test metrics
        np.save(os.path.join(self.output_dir, 'tr_loss.npy'), tr_loss_db)
        np.save(os.path.join(self.output_dir, 'val_loss.npy'), test_loss_db)
        np.save(os.path.join(self.output_dir, 'tr_acc.npy'), tr_acc_db)
        np.save(os.path.join(self.output_dir, 'val_acc.npy'), test_acc_db)

    def test(self, loader, test_type):
        # get meters ready
        loss_meter = utils.AverageMeter()
        summary = {'avg_loss': 0, 'avg_acc': 0}

        num_pos_samples = np.zeros(self.m)
        num_neg_samples = np.zeros(self.m)
        num_pos_correct = np.zeros(self.m)
        num_neg_correct = np.zeros(self.m)

        # other items
        labels = []
        p_y1_list = [[] for _ in range(self.m)]
        data_list = [[] for _ in range(self.m)]
        ratios_list = [[] for _ in range(self.m)]

        with torch.no_grad():
            self.model.eval()
            # self.flow.eval()

            # test classifier
            t = tqdm(iter(loader), leave=False, total=len(loader))
            for i, (z_ref, z_biased) in enumerate(t):
                z = self.interpolate(z_ref.to(self.device), z_biased.to(self.device)) # (self.m + 1, batch_size, x_dim)
                y = torch.cat([torch.ones(z_ref.shape[0]), torch.zeros(z_biased.shape[0])])

                # TODO: ENCODE WITH FLOW!!!! (REFER TO TRAIN)
                if self.args.encode_z:
                    # TODO: preprocessing before glow
                    z = utils.glow_preprocess(z)
                    z, _, _ = self.flow(z)
                
                # if 'mlp' in self.config.model.name:
                #     z = z.view(len(z), -1)
                # else:
                #     z = z.view(len(z), self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
                y = y.to(self.device).long()

                logits_list, probas_list = self.model(z)
                if self.config.loss.name == 'joint':
                    loss_list = [self.loss(self.model, z, logits, y, self.config.loss.alpha) for logits in logits_list]
                else:
                    loss_list = [self.loss(logits.squeeze(), y.float()) for logits in logits_list]
                
                # average across bridges
                loss = sum(loss_list)/len(loss_list)
                loss_meter.update(loss)
                
                # check accuracy
                for i, logits in enumerate(logits_list):
                    y_preds = self.get_preds(logits.squeeze())
                    num_pos_samples[i] += y.sum()
                    num_neg_samples[i] += y.size(0) - y.sum()
                    num_pos_correct[i] += (y_preds[y == 1] == y[y == 1]).sum()
                    num_neg_correct[i] += (y_preds[y == 0] == y[y == 0]).sum()

                # save items
                labels.append(y.cpu())

                for i in range(self.m):
                    p_y1_list[i].append(probas_list[i])
                    data_list[i].append(z[i])
        
        avg_acc_list = (num_pos_correct / num_pos_samples + num_neg_correct / num_neg_samples) / 2
        avg_acc = sum(avg_acc_list)/len(avg_acc_list) #avg_acc.detach().cpu().numpy()
        # Completed running test
        print('Completed evaluation: {} loss: {}, {} acc: {}'.format(
            test_type,
            np.round(loss_meter.avg.detach().cpu().numpy(), 3), 
            test_type,
            avg_acc))
        summary.update(
            dict(avg_loss=np.round(loss_meter.avg.detach().cpu().numpy(), 3),
                avg_acc=avg_acc))
        print()
        # pprint(summary)

        # correctly format items to return
        labels = torch.cat(labels).data.cpu().numpy().squeeze()

        for i in range(self.m):
            p_y1_list[i] = torch.cat(p_y1_list[i])
            data_list[i] = torch.cat(data_list[i])

            if self.config.loss.name not in ['bce', 'joint']:
                ratios_list[i] = (p_y1_list[i][:,1]/p_y1_list[i][:,0]).data.cpu().numpy()
            else:
                ratios_list[i] = (p_y1_list[i]/(1-p_y1_list[i])).data.cpu().numpy()
            p_y1_list[i] = p_y1_list[i].data.cpu().numpy().squeeze()
            data_list[i] = data_list[i].data.cpu().numpy()

        return loss_meter.avg, avg_acc, labels, p_y1_list, ratios_list, data_list

    def clf_diagnostics(self, y_valid, valid_prob_pos, ratios, val_x, split, bridge_num):
            """
            function to check (1) classifier calibration; and (2) save weights
            """
            # assess calibration
            # fraction_of_positives, mean_predicted_value = calibration_curve(y_valid, valid_prob_pos[:, 1])
            fraction_of_positives, mean_predicted_value = calibration_curve(y_valid, valid_prob_pos)

            # save calibration results
            np.save(os.path.join(self.output_dir, f'{split}_fraction_of_positives_bridge{bridge_num}'), fraction_of_positives)
            np.save(os.path.join(self.output_dir, f'{split}_mean_predicted_value_bridge{bridge_num}.npy'), mean_predicted_value)

            # obtain figure
            plt.figure(figsize=(10,5))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='dset_clf')
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

            plt.title(f'{str.title(split)} Set: Calibration Curve',fontsize=22)
            plt.ylabel('Fraction of positives',fontsize=22)
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.tick_params(axis='both', which='minor', labelsize=20)
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, f'{split}_calibration_curve_bridge{bridge_num}.pdf'))

            # save density ratios
            np.savez(
                os.path.join(self.output_dir, f'{split}_ratios_bridge{bridge_num}.npz'), **{'ratios': ratios, 'd_labels': y_valid, 'data': val_x})

    def plot_train_test_curves(self, tr_loss, test_loss, metric='Loss', title='train_curve_loss'):
        sns.set_context('paper', font_scale=2)
        sns.set_style('whitegrid')

        n = len(tr_loss)
        plt.figure(figsize=(8,5))
        plt.plot(range(1, n+1), tr_loss, '-o', label='train')
        plt.plot(range(1, n+1), test_loss, '-o', label='test')

        plt.title('Train vs. Test {}'.format(metric), fontsize=22)
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('{}'.format(metric), fontsize=20)
        # plt.xticks(range(1, n+1))
        plt.legend(loc='upper right', fontsize=15)

        sns.despine()
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, '{}.png'.format(title)), dpi=200)

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

    def flow_diagnostics(self, step, n_row=10):
        self.model.eval()

        dset1 = torch.stack([x[0] for x in self.test_dataloader.dataset.ref_dset.dataset])
        dset2 = torch.stack([x[0] for x in self.test_dataloader.dataset.biased_dset.dataset])
        data = torch.cat([dset1, dset2])
        data = data.to(self.device)
        u, _ = self.model.flow.forward(data)

        # can you actually get samples?
        ux = self.model.flow.base_dist.sample((len(data),))
        samples, _ = self.model.flow.inverse(ux)

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
        plt.savefig(os.path.join(self.output_dir, 'sample' + (step != None)*'_epoch_{}'.format(step) + '.png'))
        plt.close()
