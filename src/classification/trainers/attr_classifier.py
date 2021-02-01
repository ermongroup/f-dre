import os
import sys
# for module imports:
sys.path.append(os.path.abspath(os.getcwd()))
import copy
import logging
from pprint import pprint
from tqdm import tqdm
import numpy as np

from datasets.data import fetch_dataloaders

import classification.utils as utils
from classification.models.mlp import MLPClassifier
from classification.models.resnet import ResnetClassifier
from classification.models.networks import *
from classification.trainers.base import BaseTrainer
from sklearn.calibration import calibration_curve

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

class AttrClassifier(BaseTrainer):
    def __init__(self, args, config):
        self.args = args
        self.config = config

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

    def get_model(self):
        model = self.get_model_cls(self.config.model.name)
        model = model.to(self.config.device)

        return model

    def get_model_cls(self, name):
        if name == 'mlp':
            model_cls = MLPClassifierv2
        elif name == 'resnet':
            model_cls = ResnetClassifier
            # model_cls = Resnet_v2
        elif name == 'resnet20':
            model_cls = resnet20()
            return model_cls
        else:
            print('Model {} not found!'.format(name))
            raise NotImplementedError

        return model_cls(self.config)

    def get_loss(self):
        if self.config.loss.name == 'bce':
            loss = F.binary_cross_entropy_with_logits
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

    # def get_datasets(self):
    #     train, val, test = dsets.get_dataset(self.args, self.config)

    #     # create dataloaders
    #     train = data_utils.DataLoader(train, batch_size=self.config.training.batch_size//2, shuffle=True)
    #     val = data_utils.DataLoader(val, batch_size=self.config.training.batch_size//2, shuffle=False)
    #     test = data_utils.DataLoader(test, batch_size=self.config.training.batch_size//2, shuffle=False)

    #     return train, val, test
    def get_preds(self, logits):
        with torch.no_grad():
            if self.config.loss.name in ['bce', 'joint']:
                y_preds = torch.round(torch.sigmoid(logits))
            else:
                probs = F.softmax(logits, dim=1)
                _, y_preds = torch.max(probs, 1)
        return y_preds
    def accuracy(self, logits, y):
        with torch.no_grad():
            if self.config.loss.name == 'bce':
                y_preds = torch.round(torch.sigmoid(logits))
            else:
                probs = F.softmax(logits, dim=1)
                _, y_preds = torch.max(probs, 1)
            acc = (y_preds == y).sum()
            acc = torch.true_divide(acc, len(y_preds)).cpu().numpy()        
        
        return acc

    def train_epoch(self, epoch):
        # get meters ready
        loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()

        # train classifier
        self.model.train()
        data_tqdm = tqdm(iter(self.train_dataloader), leave=False, total=len(self.train_dataloader))

        for i, (z, y) in enumerate(data_tqdm):
            # random permutation of data
            if self.config.model.name =='mlp':
                z = z.to(self.device).view(len(z), -1)
            else:
                z = z.to(self.device).view(len(z), self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
            idx = torch.randperm(len(z))
            z = z[idx].to(self.device).float()
            if self.config.loss.name == 'bce':
                y = y[idx].to(self.device).float()
            else:
                y = y[idx].to(self.device).long()
            
            # NOTE: here, biased (y=0) and reference (y=1)
            logits, _ = self.model(z)
            loss = self.loss(logits.squeeze(), y.float())
            loss_meter.update(loss.item())

            # check accuracy
            accs = self.accuracy(logits.squeeze(), y.squeeze())
            acc_meter.update(accs)

            # gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # get summary
            summary = dict(avg_loss=loss.item(), clf_acc=acc_meter.avg)

            if (i + 1) % self.config.training.iter_log == 0:
                summary.update(
                    dict(avg_loss=np.round(loss.float().item(), 3),
                        clf_acc=np.round(acc_meter.avg, 3)))
                print()
                pprint(summary)

            # pbar
            desc = 'loss: {}'.format(loss.item())
            data_tqdm.set_description(desc)
            data_tqdm.update(z.shape[0])
        # end of training epoch
        print()
        print('Completed epoch {}: train loss: {}, train acc: {}'.format(
            epoch, 
            np.round(loss_meter.avg, 3), 
            np.round(acc_meter.avg, 3)))
        summary.update(dict(
            avg_loss=loss_meter.avg,
            avg_acc=acc_meter.avg))
        # pprint(summary)

        return loss_meter.avg, acc_meter.avg

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
            scheduler.step()
            
            # check performance on validation set
            # if val_acc >= best_acc:
            if val_loss < best_loss:
                best_loss = val_loss
                best_acc = val_acc
                best_epoch = epoch
                best = True
                self.clf_diagnostics(val_labels, val_probs, val_ratios, val_data, split='val')
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
        acc_meter = utils.AverageMeter()
        summary = {'avg_loss': 0, 'avg_acc': 0}

        # other items
        labels = []
        data = []
        p_y1 = []
        num_examples = 0.

        with torch.no_grad():
            self.model.eval()

            # test classifier
            t = tqdm(iter(loader), leave=False, total=len(loader))
            for i, (z, y) in enumerate(t):
                if self.config.model.name =='mlp':
                    z = z.to(self.device).view(len(z), -1)
                else:
                    z = z.to(self.device).view(len(z), self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
                z = z.to(self.device).float()
                if self.config.loss.name == 'bce':
                    y = y.to(self.device).float()
                else:
                    y = y.to(self.device).long()

                logits, probs = self.model(z)
                loss = self.loss(logits.squeeze(), y.float())
                # _, pred = torch.max(probs, 1)
                num_examples += y.size(0)
                loss_meter.update(loss.item())
                
                # TODO: check accuracy between classes
                accs = self.accuracy(logits.squeeze(), y.squeeze())
                acc_meter.update(accs)

                # save items
                labels.append(y.cpu())
                p_y1.append(probs)
                data.append(z)
        
        # Completed running test
        print('Completed evaluation: {} loss: {}, {} acc: {}'.format(
            test_type,
            np.round(loss_meter.avg, 3), 
            test_type,
            np.round(acc_meter.avg, 3)))
        summary.update(
            dict(avg_loss=np.round(loss_meter.avg, 3),
                avg_acc=np.round(acc_meter.avg, 3)))
        print()
        # pprint(summary)

        # correctly format items to return
        labels = torch.cat(labels).squeeze()
        labels = labels.data.cpu().numpy()
        p_y1 = torch.cat(p_y1).squeeze()
        # from utils import logsumexp_1p
        ratios = p_y1/(1-p_y1)
        ratios = ratios.data.cpu().numpy()
        # ratios = (p_y1[:,1]/p_y1[:,0]).data.cpu().numpy()
        p_y1 = p_y1.data.cpu().numpy()
        data = data.data.cpu().numpy()

        return loss_meter.avg, acc_meter.avg, labels, p_y1, ratios, data

    def clf_diagnostics(self, y_valid, valid_prob_pos, ratios, val_x, split):
            """
            function to check (1) classifier calibration; and (2) save weights
            """
            # assess calibration
            if self.config.model.name != 'mlp':
                valid_prob_pos = valid_prob_pos[:, 1]
            fraction_of_positives, mean_predicted_value = calibration_curve(y_valid, valid_prob_pos)

            # save calibration results
            np.save(os.path.join(self.output_dir, f'{split}_fraction_of_positives'), fraction_of_positives)
            np.save(os.path.join(self.output_dir, f'{split}_mean_predicted_value.npy'), mean_predicted_value)

            # obtain figure
            plt.figure(figsize=(10,5))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='dset_clf')
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

            plt.title(f'{str.title(split)} Set: Calibration Curve',fontsize=22)
            plt.ylabel('Fraction of positives',fontsize=22)
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.tick_params(axis='both', which='minor', labelsize=20)
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, f'{split}_calibration_curve.pdf'))

            # save density ratios
            np.savez(
                os.path.join(self.output_dir, f'{split}_ratios.npz'), **{'ratios': ratios, 'd_labels': y_valid, 'data': val_x})

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
