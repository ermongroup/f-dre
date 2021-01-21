import os
import sys

import copy
import logging
from pprint import pprint
from tqdm import tqdm
import numpy as np

import utils
import dsets
from dsets.flipped_mnist import (
    SplitEncodedMNIST,
    SplitMNIST
)
from models.mlp import MLPClassifier
from models.resnet import ResnetClassifier
from trainers.base import BaseTrainer
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

# device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Classifier(BaseTrainer):
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
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self.get_datasets()

        # saving
        self.checkpoint_dir = config.ckpt_dir
        self.output_dir = config.out_dir

    def get_model(self):
        model = self.get_model_cls(self.config.model.name)
        model = model.to(self.config.device)

        return model

    def get_model_cls(self, name):
        if name == 'mlp':
            model_cls = MLPClassifier
        elif name == 'resnet':
            model_cls = ResnetClassifier
        else:
            print('Model {} not found!'.format(name))
            raise NotImplementedError

        return model_cls(self.config)

    def get_loss(self):
        if self.config.loss == 'bce':
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

    def get_datasets(self):
        train, val, test = dsets.get_dataset(self.args, self.config)

        # create dataloaders
        train = data_utils.DataLoader(train, batch_size=self.config.training.batch_size//2, shuffle=True)
        val = data_utils.DataLoader(val, batch_size=self.config.training.batch_size//2, shuffle=False)
        test = data_utils.DataLoader(test, batch_size=self.config.training.batch_size//2, shuffle=False)

        return train, val, test

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

    def get_preds(self, logits):
        with torch.no_grad():
            if self.config.loss.name == 'bce':
                y_preds = torch.round(torch.sigmoid(logits))
            else:
                probs = F.softmax(logits, dim=1)
                _, y_preds = torch.max(probs, 1)
        return y_preds

    def train_epoch(self, epoch):
        # get meters ready
        loss_meter = utils.AverageMeter()

        # train classifier
        self.model.train()
        data_tqdm = tqdm(iter(self.train_dataloader), leave=False, total=len(self.train_dataloader))

        num_pos_correct = 0
        num_pos_samples = 0
        num_neg_correct = 0
        num_neg_samples = 0

        for i, (z_ref, z_biased) in enumerate(data_tqdm):
            z = torch.cat([z_ref, z_biased])
            y = torch.cat([torch.ones(z_ref.shape[0]), torch.zeros(z_biased.shape[0])])

            # random permutation of data
            if self.config.model.name =='mlp':
                z = z.to(self.device).view(len(z), -1)
            else:
                z = z.to(self.device).view(len(z), self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
            idx = torch.randperm(len(z))
            z = z[idx].to(self.device).float()
            y = y[idx].to(self.device).long()
            
            # NOTE: here, biased (y=0) and reference (y=1)
            logits, _ = self.model(z)
            loss = self.loss(logits, y)
            loss_meter.update(loss.item())

            # check accuracy
            y_preds = self.get_preds(logits.squeeze())
            num_pos_samples += y.sum()
            num_neg_samples += y.size(0) - y.sum()
            num_pos_correct += (y_preds[y == 1] == y[y == 1]).sum()
            num_neg_correct += (y_preds[y == 0] == y[y == 0]).sum()
            acc = (num_pos_correct / num_pos_samples + num_neg_correct / num_neg_samples) / 2

            # gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # get summary
            summary = dict(avg_loss=loss.item(), clf_acc=acc)

            if (i + 1) % self.config.training.iter_log == 0:
                summary.update(
                    dict(avg_loss=np.round(loss.float().item(), 3),
                        clf_acc=np.round(acc.detach().cpu().numpy(), 3)))
                print()
                pprint(summary)

            # pbar
            desc = 'loss: {}'.format(loss.item())
            data_tqdm.set_description(desc)
            data_tqdm.update(z.shape[0])
        # end of training epoch
        avg_acc = (num_pos_correct / num_pos_samples + num_neg_correct / num_neg_samples) / 2
        avg_acc = avg_acc.detach().cpu().numpy()
        print()
        print('Completed epoch {}: train loss: {}, train acc: {}'.format(
            epoch, 
            np.round(loss_meter.avg, 3), 
            np.round(avg_acc, 3)))
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
            val_loss, val_acc, val_labels, val_probs, val_ratios = self.test(self.val_dataloader, 'val')
            scheduler.step()
            
            # check performance on validation set
            if val_loss < best_loss:
                print('saving best model..')
                best_acc = val_acc
                best_loss = val_loss
                best_epoch = epoch
                best = True
                self.clf_diagnostics(val_labels, val_probs, val_ratios, split='val')
            else:
                best = False
            self._save_checkpoint(epoch, save_best=best)
            
            # save metrics (TODO: test is actually validation here)
            tr_loss_db[epoch-1] = tr_loss
            test_loss_db[epoch-1] = val_loss
            tr_acc_db[epoch - 1] = tr_acc
            test_acc_db[epoch - 1] = val_acc
        # evaluate on test
        test_loss, test_acc, test_labels, test_probs, test_ratios = self.test(self.test_dataloader, 'test')
        
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

        num_pos_samples = 0
        num_neg_samples = 0
        num_pos_correct = 0
        num_neg_correct = 0

        # other items
        labels = []
        p_y1 = []
        num_examples = 0.

        with torch.no_grad():
            self.model.eval()

            # test classifier
            t = tqdm(iter(loader), leave=False, total=len(loader))
            for i, (z_ref, z_biased) in enumerate(t):
                z = torch.cat([z_ref, z_biased])
                y = torch.cat([
                    torch.ones(z_ref.shape[0]),
                    torch.zeros(z_biased.shape[0])
                ])
                
                if self.config.model.name =='mlp':
                    z = z.to(self.device).view(len(z), -1)
                else:
                    z = z.to(self.device).view(len(z), self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
                y = y.to(self.device).long()

                logits, probs = self.model(z)
                loss = self.loss(logits, y, reduction='sum')
                _, pred = torch.max(probs, 1)
                num_examples += y.size(0)
                loss_meter.update(loss.item())
                
                # TODO: check accuracy between classes
                y_preds = self.get_preds(logits.squeeze())
                num_pos_samples += y.sum()
                num_neg_samples += y.size(0) - y.sum()
                num_pos_correct += (y_preds[y == 1] == y[y == 1]).sum()
                num_neg_correct += (y_preds[y == 0] == y[y == 0]).sum()

                # save items
                labels.append(y.cpu())
                p_y1.append(probs)
        
        avg_acc = (num_pos_correct / num_pos_samples + num_neg_correct / num_neg_samples) / 2
        avg_acc = avg_acc.detach().cpu().numpy()
        # Completed running test
        print('Completed evaluation: {} loss: {}, {} acc: {}'.format(
            test_type,
            np.round(loss_meter.avg, 3), 
            test_type,
            np.round(avg_acc, 3)))
        summary.update(
            dict(avg_loss=np.round(loss_meter.avg, 3),
                avg_acc=np.round(avg_acc, 3)))
        print()
        # pprint(summary)

        # correctly format items to return
        labels = torch.cat(labels).data.cpu().numpy()
        p_y1 = torch.cat(p_y1)
        ratios = (p_y1[:,1]/p_y1[:,0]).data.cpu().numpy()
        p_y1 = p_y1.data.cpu().numpy()

        return loss_meter.avg, avg_acc, labels, p_y1, ratios

    def clf_diagnostics(self, y_valid, valid_prob_pos, ratios, split):
            """
            function to check (1) classifier calibration; and (2) save weights
            """
            # assess calibration
            fraction_of_positives, mean_predicted_value = calibration_curve(y_valid, valid_prob_pos[:, 1])

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
                os.path.join(self.output_dir, f'{split}_ratios.npz'), **{'ratios': ratios, 'd_labels': y_valid})

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
