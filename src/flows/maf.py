"""
Masked Autoregressive Flow for Density Estimation
arXiv:1705.07057v4
(DEPRECATED -> see trainers/flow.py)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import yaml
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.distributions import Normal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, os
# for module imports:
sys.path.append(os.path.abspath(os.getcwd()))
import math
import argparse
import pprint
import copy
import numpy as np
from copy import deepcopy

from data import fetch_dataloaders
from models.maf import *
from src.classification.models.resnet import ResnetClassifier
from src.classification.models.mlp import MLPClassifier
import utils

CHECKPOINT_DIR = '/atlas/u/madeline/multi-fairgen/src/classification/checkpoints'
ATTR_CLFS = {
    # attr: digits
    'digit': os.path.join(CHECKPOINT_DIR, 'digits_attr_clf', 'model_best.pth'),
    # attr: background color   
    'background': os.path.join(CHECKPOINT_DIR,'background_attr_clf', 'model_best.pth')     
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser()
# action
parser.add_argument('--perc', type=float, help='Used with CMNIST; percentage of reference dataset size relative to original dataset')
parser.add_argument('--subset', type=bool, help='if True, uses version of MNIST that is split into (0,7)')
parser.add_argument('--output_dir', type=str)
# others
parser.add_argument('--alpha', type=float, default=0.06, help='flattening coefficient')
parser.add_argument('--generate_samples', action='store_true', help='generate samples')
parser.add_argument('--train', action='store_true', help='Train a flow.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a flow.')
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--encode', action='store_true', help='Save data z-encodings using a trained model.')
parser.add_argument('--n-samples', type=int, default=50000, help='number of samples to generate after training')
parser.add_argument('--channels', type=int, default=1, help='number of channels in image')
parser.add_argument('--image-size', type=int, default=28, help='H/W of image')
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--clf-ckpt', type=str, default=None, help='Checkpoint for pretrained classifier for DRE.')
parser.add_argument('--fair-generate', action='store_true', help='Generate samples from a model using importance weights.')
parser.add_argument('--data_dir', default='./data/', help='Location of datasets.')
parser.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')
parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
# data
parser.add_argument('--config', default='/mnist/attr_bkgd.yaml', help='Which dataset to use.')
parser.add_argument('--dataset', default='toy', help='Which dataset to use.')
parser.add_argument('--digits', type=int, nargs='+', help='Used with FLippedMNISTSubset/MNISTSubset; which digits to include in dataset.')
parser.add_argument('--digit_percs', type=float, nargs='+', help='Used with --digits; perc of each digit to include in dataset.')
parser.add_argument('--flipped_digits', type=int, nargs='+', help='Used with FLippedMNISTSubset/MNISTSubset; which digits to include in dataset.')
parser.add_argument('--flipped_digit_percs', type=float, nargs='+', help='Used with --digits; perc of each digit to include in dataset.')
parser.add_argument('--flip_toy_var_order', action='store_true', help='Whether to flip the toy dataset variable order to (x2, x1).')
parser.add_argument('--seed', type=int, default=1, help='Random seed to use.')
# model
parser.add_argument('--model', default='maf', help='Which model to use: made, maf.')
# made parameters
parser.add_argument('--n_blocks', type=int, default=5, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=100, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--activation_fn', type=str, default='relu', help='What activation function to use in the MADEs.')
parser.add_argument('--input_order', type=str, default='sequential', help='What input order to use (sequential | random).')
parser.add_argument('--conditional', default=False, action='store_true', help='Whether to use a conditional model.')
parser.add_argument('--no_batch_norm', action='store_true')
# training params
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--log_interval', type=int, default=1000, help='How often to show loss statistics and save samples.')

# --------------------
# Train and evaluate
# --------------------

def train(model, dataloader, optimizer, epoch, args):

    for i, data in enumerate(dataloader):
        model.train()

        # check if labeled dataset
        if len(data) == 1:
            x, y = data[0], None
        else:
            x, y = data
            y = y.to(device)
        x = x.view(x.shape[0], -1).to(device)
        loss = - model.log_prob(x, y if args.cond_label_size else None).mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch, args.start_epoch + args.n_epochs, i, len(dataloader), loss.item()))

@torch.no_grad()
def evaluate(model, dataloader, epoch, args):
    model.eval()

    # conditional model
    if args.cond_label_size is not None:
        logprior = torch.tensor(1 / args.cond_label_size).log().to(device)
        loglike = [[] for _ in range(args.cond_label_size)]

        for i in range(args.cond_label_size):
            # make one-hot labels
            labels = torch.zeros(args.batch_size, args.cond_label_size).to(device)
            labels[:,i] = 1

            for x, y in dataloader:
                x = x.view(x.shape[0], -1).to(device)
                loglike[i].append(model.log_prob(x, labels))

            loglike[i] = torch.cat(loglike[i], dim=0)   # cat along data dim under this label
        loglike = torch.stack(loglike, dim=1)           # cat all data along label dim

        # log p(x) = log ∑_y p(x,y) = log ∑_y p(x|y)p(y)
        # assume uniform prior      = log p(y) ∑_y p(x|y) = log p(y) + log ∑_y p(x|y)
        logprobs = logprior + loglike.logsumexp(dim=1)
        # TODO -- measure accuracy as argmax of the loglike

    # unconditional model
    else:
        logprobs = []
        for data in dataloader:
            x = data[0].view(data[0].shape[0], -1).to(device)
            logprobs.append(model.log_prob(x))
        logprobs = torch.cat(logprobs, dim=0).to(device)

    logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.var(0).sqrt() / math.sqrt(len(dataloader.dataset))
    output = 'Evaluate ' + (epoch != None)*'(epoch {}) -- '.format(epoch) + 'logp(x) = {:.3f} +/- {:.3f}'.format(logprob_mean, logprob_std)
    print(output)
    print(output, file=open(args.results_file, 'a'))
    return logprob_mean, logprob_std


@torch.no_grad()
def generate(model, dataset_lam, args, step=None, n_row=10):
    model.eval()

    # conditional model
    if args.cond_label_size:
        samples = []
        labels = torch.eye(args.cond_label_size).to(device)

        for i in range(args.cond_label_size):
            # sample model base distribution and run through inverse model to sample data space
            u = model.base_dist.sample((n_row, args.n_components)).squeeze()
            labels_i = labels[i].expand(n_row, -1)
            sample, _ = model.inverse(u, labels_i)
            log_probs = model.log_prob(sample, labels_i).sort(0)[1].flip(0)  # sort by log_prob; take argsort idxs; flip high to low
            samples.append(sample[log_probs])

        samples = torch.cat(samples, dim=0)

    # unconditional model
    else:
        u = model.base_dist.sample((n_row**2, args.n_components)).squeeze()
        samples, _ = model.inverse(u)
        log_probs = model.log_prob(samples).sort(0)[1].flip(0)  # sort by log_prob; take argsort idxs; flip high to low
        samples = samples[log_probs]

    # convert and save images
    
    # samples = samples.view(samples.shape[0], *args.input_dims)
    samples = samples.view((samples.shape[0], args.channels, args.image_size, args.image_size))
    # samples = (torch.sigmoid(samples) - dataset_lam) / (1 - 2 * dataset_lam)
    samples = torch.sigmoid(samples)
    samples = torch.clamp(samples, 0., 1.)
    print('args.save_freq: ', args.save_freq)
    print('step: ', step)
    if step % args.save_freq == 0:
        filename = 'generated_samples' + (step != None)*'_epoch_{}'.format(step) + '.png'
        save_image(samples, os.path.join(args.output_dir, filename), nrow=n_row, normalize=True)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_classifier(args, name):
    if name == 'resnet':
        clf = ResnetClassifier(args)
        ckpt_path = args.clf_ckpt
    else:
        with open(os.path.join('src/classification/configs', args.config), 'r') as f:
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
    clf = clf.to(device)
    return clf


@torch.no_grad()
def generate_many_samples(model, args, clf, n_row=10):
    all_samples = []
    preds = []
    model.eval()
    clf.eval()

    print('generating {} samples in batches of 1000...'.format(args.n_samples))
    n_batches = int(args.n_samples // 1000)
    for n in range(n_batches):
        if (n % 10 == 0) and (n > 0):
            print('on iter {}/{}'.format(n, n_batches))
        u = model.base_dist.sample((1000, args.n_components)).squeeze()
        samples, _ = model.inverse(u)
        log_probs = model.log_prob(samples).sort(0)[1].flip(0)  # sort by log_prob; take argsort idxs; flip high to low
        samples = samples[log_probs]
        samples = samples.view((samples.shape[0], args.channels, args.image_size, args.image_size))
        samples = torch.sigmoid(samples)
        samples = torch.clamp(samples, 0., 1.)  # check if we want to multiply by 255 and transpose if we're gonna do metric stuff on here

        # get classifier predictions
        logits, probas = clf(samples.view(len(samples), -1))
        _, pred = torch.max(probas, 1)

        # save things
        preds.append(pred.detach().cpu().numpy())
        all_samples.append(samples.detach().cpu().numpy())
    all_samples = np.vstack(all_samples)
    preds = np.hstack(preds)
    fair_disc_l2, fair_disc_l1, fair_disc_kl = utils.fairness_discrepancy(preds, 2)
    np.savez(os.path.join(args.output_dir, f'{args.dataset}_maf_perc{args.perc}', 'samples'), **{'x': all_samples})
    np.savez(os.path.join(args.output_dir, f'{args.dataset}_maf_perc{args.perc}', 'metrics'), 
        **{
        'preds': preds,
        'l2_fair_disc': fair_disc_l2,
        })
    # maybe just save some samples just for visualizations?
    filename = 'samples'+ '.png'
    save_image(all_samples[0:100], os.path.join(args.output_dir, filename), nrow=n_row, normalize=True)


@torch.no_grad()
def fair_generate(model, args, dre_clf, attr_clf, step=None, n_row=10):
    from torch.distributions import Categorical

    all_samples = []
    preds = []

    model.eval()
    dre_clf.eval()
    attr_clf.eval()

    # print('generating {} samples in batches of 1000...'.format(args.n_samples))
    print('running SIR sampling...as a sanity check, we are only going to generate 20 samples total.')
    n_batches = int(args.n_samples // 1000)
    # for n in range(n_batches):
    
    for n in range(20):
        if (n % 5 == 0) and (n > 0):
            print('on iter {}/{}'.format(n, 20))
        u = model.base_dist.sample((1000, args.n_components)).squeeze().to(device)
        
        # TODO: reweight the samples via dre_clf
        logits, probas = dre_clf(u.view(1000, 1, 28, 28))
        # print('flattening importance weights by alpha={}'.format(args.alpha))
        # ratios = ratios**(args.alpha)
        ratios = (probas[:, 1]/probas[:, 0])
        r_probs = ratios/ratios.sum()
        sir_j = Categorical(r_probs).sample().item()

        samples, _ = model.inverse(u[sir_j].unsqueeze(0))
        log_probs = model.log_prob(samples).sort(0)[1].flip(0)  # sort by log_prob; take argsort idxs; flip high to low
        samples = samples[log_probs]
        samples = samples.view((samples.shape[0], args.channels, args.image_size, args.image_size))
        samples = torch.sigmoid(samples)
        samples = torch.clamp(samples, 0., 1.)  # check if we want to multiply by 255 and transpose if we're gonna do metric stuff on here

        # get classifier predictions
        logits, probas = attr_clf(samples.view(len(samples), -1))
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
    filename = 'fair_samples_sir'.format(args.alpha) + '.png'
    save_image(torch.from_numpy(all_samples), os.path.join(args.output_dir, filename), nrow=n_row, normalize=True)


def save_encodings(model, train_loader, val_loader, test_loader, model_name, data_dir, dataset):
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
        save_path = os.path.join(data_dir, 'encodings', data_type, '{}_{}_mnist_z_perc{}'.format(model_name, split, args.perc))
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
        save_path = os.path.join(data_dir, 'encodings', data_type, '{}_{}_cmnist_z_perc{}'.format(model_name, split, args.perc))
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

    
def train_and_evaluate(model, train_loader, test_loader, optimizer, args):
    best_eval_logprob = float('-inf')

    for i in range(args.start_epoch, args.start_epoch + args.n_epochs):
        train(model, train_loader, optimizer, i, args)
        eval_logprob, _ = evaluate(model, test_loader, i, args)

        # save training checkpoint
        torch.save({'epoch': i,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()},
                    os.path.join(args.output_dir, 'model_checkpoint.pt'))
        # save model only
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_state.pt'))

        # save best state
        if eval_logprob > best_eval_logprob:
            best_eval_logprob = eval_logprob
            torch.save({'epoch': i,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                        os.path.join(args.output_dir, 'best_model_checkpoint.pt'))

        # plot sample
        if args.dataset == 'TOY':
            plot_sample_and_density(model, train_loader.dataset.base_dist, args, step=i)
        if args.dataset in ['MNIST', 'MNIST_combined', 'FlippedMNIST', 'MNISTSubset_combined']:
            generate(model, train_loader.dataset.lam, args, step=i)

# --------------------
# Plot
# --------------------

def plot_density(dist, ax, ranges, flip_var_order=False):
    (xmin, xmax), (ymin, ymax) = ranges
    # sample uniform grid
    n = 200
    xx1 = torch.linspace(xmin, xmax, n)
    xx2 = torch.linspace(ymin, ymax, n)
    xx, yy = torch.meshgrid(xx1, xx2)
    xy = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze()

    if flip_var_order:
        xy = xy.flip(1)

    # run uniform grid through model and plot
    density = dist.log_prob(xy).exp()
    ax.contour(xx, yy, density.view(n,n).data.numpy())

    # format
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([xmin, xmax])
    ax.set_yticks([ymin, ymax])


def plot_dist_sample(data, ax, ranges):
    ax.scatter(data[:,0].data.numpy(), data[:,1].data.numpy(), s=10, perc=0.4)
    # format
    (xmin, xmax), (ymin, ymax) = ranges
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([xmin, xmax])
    ax.set_yticks([ymin, ymax])


def plot_sample_and_density(model, target_dist, args, ranges_density=[[-5,20],[-10,10]], ranges_sample=[[-4,4],[-4,4]], step=None):
    model.eval()
    fig, axs = plt.subplots(1, 2, figsize=(6,3))

    # sample target distribution and pass through model
    data = target_dist.sample((2000,))
    u, _ = model(data)

    # plot density and sample
    plot_density(model, axs[0], ranges_density, args.flip_var_order)
    plot_dist_sample(u, axs[1], ranges_sample)

    # format and save
    matplotlib.rcParams.update({'xtick.labelsize': 'xx-small', 'ytick.labelsize': 'xx-small'})
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'sample' + (step != None)*'_epoch_{}'.format(step) + '.png'))
    plt.close()



# --------------------
# Run
# --------------------

if __name__ == '__main__':

    args = parser.parse_args()
    # setup file ops
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    torch.manual_seed(args.seed)
    if device.type == 'cuda': torch.cuda.manual_seed(args.seed)

    # load data
    # TODO: clean up these data options
    if args.conditional: assert args.dataset in ['MNIST', 'CIFAR10', 'MNIST_combined'], 'Conditional inputs only available for labeled datasets MNIST and CIFAR10.'
    train_dataloader, val_dataloader, test_dataloader = fetch_dataloaders(args.dataset, args.batch_size, device, args, args.flip_toy_var_order)
    if args.dataset not in ['MNIST_combined_z', 'MNISTSubset_combined_z']:
        args.input_size = train_dataloader.dataset.input_size
        args.input_dims = train_dataloader.dataset.input_dims
        args.cond_label_size = train_dataloader.dataset.label_size if args.conditional else None
    else:
        args.input_size = train_dataloader[0].dataset.input_size
        args.input_dims = train_dataloader[0].dataset.input_dims
        args.cond_label_size = train_dataloader[0].dataset.label_size if args.conditional else None

    # model
    if args.model == 'made':
        model = MADE(args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                     args.activation_fn, args.input_order)
    elif args.model == 'mademog':
        assert args.n_components > 1, 'Specify more than 1 component for mixture of gaussians models.'
        model = MADEMOG(args.n_components, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                     args.activation_fn, args.input_order)
    elif args.model == 'maf':
        model = MAF(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                    args.activation_fn, args.input_order, batch_norm=not args.no_batch_norm)
    elif args.model == 'mafmog':
        assert args.n_components > 1, 'Specify more than 1 component for mixture of gaussians models.'
        model = MAFMOG(args.n_blocks, args.n_components, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                    args.activation_fn, args.input_order, batch_norm=not args.no_batch_norm)
    elif args.model =='realnvp':
        model = RealNVP(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.cond_label_size,
                        batch_norm=not args.no_batch_norm)
    else:
        raise ValueError('Unrecognized model.')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    if args.restore_file:
        # load model and optimizer states
        print('restoring model checkpoint from {}'.format(args.restore_file))
        state = torch.load(args.restore_file, map_location=device)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        args.start_epoch = state['epoch'] + 1
        # set up paths
        # args.output_dir = os.path.dirname(args.restore_file)
    args.results_file = os.path.join(args.output_dir, args.results_file)
    print('saving outputs in {}'.format(args.output_dir))

    # print('Loaded settings and model:')
    # print(pprint.pformat(args.__dict__))
    # print(model)
    # print(pprint.pformat(args.__dict__), file=open(args.results_file, 'a'))
    # print(model, file=open(args.results_file, 'a'))

    if args.train:
        train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, args)

    if args.evaluate:
        evaluate(model, test_dataloader, None, args)
    
    if args.encode:
        assert args.restore_file is not None, "Must specify --restore_file to encode dataset."
        print('saving z-encodings from pretrained model...')
        save_encodings(model, train_dataloader, val_dataloader, test_dataloader, args.model, args.data_dir, args.dataset)
    if args.generate:
        if args.dataset == 'TOY':
            base_dist = train_dataloader.dataset.base_dist
            plot_sample_and_density(model, base_dist, args, ranges_density=[[-15,4],[-3,3]], ranges_sample=[[-1.5,1.5],[-3,3]])
        elif args.dataset == 'MNIST' or args.dataset == 'FlippedMNIST' or args.dataset == 'MNIST_combined':
            generate(model, train_dataloader.dataset.lam, args)
    if args.generate_samples:
        if 'MNIST' in args.dataset:
            attr_clf = load_classifier(args, 'mlp')
            if not args.fair_generate:
                generate_many_samples(model, args, attr_clf)
            else:
                dre_clf = load_classifier(args, 'resnet')
                fair_generate(model, args, dre_clf, attr_clf)
                # fair_generate(model, args, dre_clf, dre_clf)
