import glob
import logging
import os
import sys
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('poster')
sns.set_style('white')

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.utils as tvu
from datasets import data_transform, get_dataset, inverse_data_transform
from datasets.celeba import SplitEncodedCelebA
from functions import get_optimizer
from functions.ckpt_util import get_ckpt_path
from functions.losses import loss_registry
from models.classifier import BasicBlock, ResNet18, build_model
from models.ema import EMAHelper
from sklearn.calibration import calibration_curve
from torch import optim
from tqdm import tqdm


class Classifier(object):
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

    def test(self, model, loader):
        test_loss = 0
        correct = 0
        num_examples = 0

        # save items
        labels = []
        p_y1 = []

        model.eval()
        with torch.no_grad():
            for i, (x_ref, x_bias)  in enumerate(train_loader):

                # concatenate elements together
                data = torch.cat([x_ref, x_bias])
                target = torch.cat([
                    torch.ones(x_ref.shape[0]),
                    torch.zeros(x_bias.shape[0])
                ])

                # random permutation of data
                idx = torch.randperm(len(data))
                data = data[idx]
                target = target[idx]

                # TODO: check transformation of z's
                data = data.to(self.device).float()
                target = target.to(self.device).long()

                # run through model
                logits, probas = model(data)
                test_loss += F.cross_entropy(logits, target, reduction='sum').item() # sum up batch loss
                _, pred = torch.max(probas, 1)
                num_examples += target.size(0)
                correct += (pred == target).sum()

                # save items
                labels.append(target.cpu())
                p_y1.append(probas)

        test_loss /= num_examples

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, num_examples,
            100. * correct / num_examples))

        # correctly format items to return
        # TODO: careful, may run out of memory here
        labels = torch.cat(labels).data.cpu().numpy()
        p_y1 = torch.cat(p_y1)
        ratios = (p_y1[:,1]/p_y1[:,0]).data.cpu().numpy()
        p_y1 = p_y1.data.cpu().numpy()

        return test_loss, labels, p_y1, ratios

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)

        train_loader = data.DataLoader(
            dataset,
            batch_size=config.classifier.batch_size//2,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.classifier.batch_size//2,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        model_cls = build_model(config.classifier.name)
        model = model_cls(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=self.config.classifier.n_classes, grayscale=False)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        # TODO: do i need to fix this optimizer? (interface w config)
        # optimizer = get_optimizer(self.config, model.parameters())
        optimizer = optim.Adam(
            model.parameters(), lr=self.config.classifier.lr)

        # classifier has finished training, evaluate sample diversity
        best_loss = sys.maxsize

        print('beginning training...')
        start_epoch, step = 0, 0
        for epoch in range(start_epoch, self.config.classifier.n_epochs):
            data_start = time.time()
            data_time = 0

            # run through each batch
            for i, (x_ref, x_bias)  in enumerate(train_loader):
                data_time += time.time() - data_start
                model.train()
                step += 1

                # concatenate elements together
                x = torch.cat([x_ref, x_bias])
                y = torch.cat([
                    torch.ones(x_ref.shape[0]),
                    torch.zeros(x_bias.shape[0])
                ])

                # random permutation of data
                idx = torch.randperm(len(x))
                x = x[idx]
                y = y[idx]
 
                # TODO: may need to equalize sizes

                # TODO: check transformation of z's
                # x = data_transform(self.config, x)
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                
                # NOTE: here, biased (y=0) and target (y=1)
                logits, probas = model(x)
                loss = F.cross_entropy(logits, y)         

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # save results to tensorboard
                tb_logger.add_scalar("loss", loss, global_step=step)

                if (i % 100) == 0:
                    logging.info(
                        f"(epoch: {epoch}) step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                    )
            # done with one epoch of training
            valid_loss, val_labels, val_probs, val_ratios = self.test(model, test_loader)

            is_best = valid_loss < best_loss
            best_loss = min(valid_loss, best_loss)
            if is_best:
                best_state = model.state_dict()
                states = [
                    best_state,
                    optimizer.state_dict(),
                    epoch,
                ]
                torch.save(states, os.path.join(self.args.log_path, "clf_ckpt.pth"))
                # save classifier diagnostics
                self.clf_diagnostics(val_labels, val_probs, val_ratios)

        # final assessment of model quality (if we use a validation set..)
        # print('finished training...testing on final test set with epoch {} ckpt'.format(best_idx))
        # # reload best model
        # model_cls = build_model(config.classifier.name)
        # model = model_cls(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=self.config.classifier.n_classes, grayscale=False)
        # model = model.to(self.device)
        # model.load_state_dict(best_state)

        # # get test
        # test_loss = test(epoch, test_loader)

    def clf_diagnostics(self, y_valid, valid_prob_pos, ratios):
        """
        function to check (1) classifier calibration; and (2) save weights
        """
        # assess calibration
        # TODO: we may not want to iterate through these bins, and just use the default value
        # for bins in [5, 6, 7, 8, 9, 10]:
        # fraction_of_positives, mean_predicted_value = calibration_curve(y_valid, valid_prob_pos[:, 1], n_bins=bins)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_valid, valid_prob_pos[:, 1])

        # save calibration results
        np.save(os.path.join(self.args.log_path, 'fraction_of_positives'), fraction_of_positives)
        np.save(os.path.join(self.args.log_path, 'mean_predicted_value.npy'), mean_predicted_value)

        # obtain figure
        plt.figure(figsize=(10,5))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='dset_clf')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        plt.title('Validation Set: Calibration Curve',fontsize=22)
        plt.ylabel('Fraction of positives',fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tick_params(axis='both', which='minor', labelsize=20)
        plt.legend()
        # plt.savefig(os.path.join(self.args.log_path, 'calibration_curve_{}bins.png'.format(bins)), dpi=300)
        plt.savefig(os.path.join(self.args.log_path, 'calibration_curve.pdf'))

        # save density ratios
        np.savez(
            os.path.join(self.args.log_path, 'ratios.npz'), **{'x': ratios})
