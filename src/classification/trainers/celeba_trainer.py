import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dsets.celeba import SplitEncodedCelebA
from utils import AverageMeter
from sklearn.calibration import calibration_curve
from torch import optim
from torch.utils.data import DataLoader

from models.mlp import MLPClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument('--data_dir', type=str, default='/atlas/u/kechoi/multi-fairgen/data', help='path to data')
    parser.add_argument('--z_file', type=str, default='z.npy', help='z encoding filename')
    parser.add_argument('--attr_file', type=str, default='attr.npy', help='attr filename')
    parser.add_argument('--class_idx', type=int, default=20, help='class index of attribute to split dataset by')
    parser.add_argument('--train_perc', type=float, default=0.8, help='%\ of data for test set')
    parser.add_argument('--test_perc', type=float, default=0.1, help='%\ of data for test set')
    parser.add_argument('--biased_split', type=float, default=0.9, help='bias split for dataset')
    parser.add_argument('--ref_split', type=float, default=0.5, help='ref split for dataset')
    parser.add_argument('--perc', type=float, default=1.0, help='size of ref dataset is \{--perc\} * size of biased')
    # model
    parser.add_argument('--n_classes', type=int, default=2, help='num classes')
    parser.add_argument('--pixels', type=int, default=196608, help='input dim / size of image')
    parser.add_argument('--h_dim', type=int, default=500, help='size of hidden layer')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--n_epochs', type=int, default=10, help='num epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of training epochs')
    
    # saving
    parser.add_argument('--out_dir', type=str, default='/atlas/u/kechoi/multi-fairgen/classifier/results')
    
    parser.add_argument('--num_workers', type=int, default=4, help='num workers for dataloader')
    
    return parser.parse_args()


def train(args):

    # TODO: handle any dataset
    train_loader = DataLoader(
        SplitEncodedCelebA(args),
        batch_size=args.batch_size//2,
        shuffle=True)

    val_loader = DataLoader(
        SplitEncodedCelebA(args, split='val'),
        batch_size=args.batch_size//2,
        shuffle=False)

    test_loader = DataLoader(
        SplitEncodedCelebA(args, split='test'),
        batch_size=args.batch_size//2,
        shuffle=False)
    
    model = MLPClassifier(args).to(args.device)
    model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_loss = float('inf')
    best_state = []

    train_losses, val_losses, test_losses = [], [], []

    print('beginning training...')

    for epoch in range(args.num_epochs):
        model.train()
        avg_loss_meter = AverageMeter()

        for z_ref, z_biased in train_loader:
            z = torch.cat([z_ref, z_biased])
            y = torch.cat([torch.ones(z_ref.shape[0]), torch.zeros(z_biased.shape[0])])

            # random permutation of data
            idx = torch.randperm(len(z))
            z = z[idx].to(args.device).view(len(idx), -1)
            y = y[idx].to(args.device).long()
            
            # NOTE: here, biased (y=0) and reference (y=1)
            logits, _ = model(z)
            loss = F.cross_entropy(logits, y)         
            avg_loss_meter.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_losses.append(avg_loss_meter.avg)
        # checkpoint by best validation loss
        val_loss, val_labels, val_probs, val_ratios = evaluate(args, model, val_loader, 'val')

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch
            ]

        # evaluation
        test_loss, test_labels, test_probs, test_ratios = evaluate(args, model, test_loader, 'test')
        clf_diagnostics(args, val_labels, val_probs, val_ratios)
        clf_diagnostics(args, test_labels, test_probs, test_ratios)

        val_losses.append(val_loss)
        test_losses.append(test_loss)

    np.save(os.path.join(args.out_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(args.out_dir, 'val_losses.npy'), val_losses)
    np.save(os.path.join(args.out_dir, 'test_losses.npy'), test_losses)   

    torch.save(best_state, os.path.join(args.out_dir, 'clf_ckpt.pth'))

def evaluate(args, model, loader, split):
    model.eval()

    loss = 0
    correct = 0
    num_examples = 0
    labels = []
    p_y1 = []

    with torch.no_grad():
        for (z_ref, z_biased) in loader:
            z = torch.cat([z_ref, z_biased])
            y = torch.cat([
                torch.ones(z_ref.shape[0]),
                torch.zeros(z_biased.shape[0])
            ])
            
            z = z.to(args.device).view(len(z), -1)
            y = y.to(args.device).long()

            logits, probs = model(z)
            loss += F.cross_entropy(logits, y, reduction='sum').item()
            _, pred = torch.max(probs, 1)
            num_examples += y.size(0)
            correct += (pred == y).sum()

            # save items
            labels.append(y.cpu())
            p_y1.append(probs)

    loss /= num_examples

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            split, loss, correct, num_examples,
            100. * correct / num_examples))

    # correctly format items to return
    labels = torch.cat(labels).data.cpu().numpy()
    p_y1 = torch.cat(p_y1)
    ratios = (p_y1[:,1]/p_y1[:,0]).data.cpu().numpy()
    p_y1 = p_y1.data.cpu().numpy()

    return loss, labels, p_y1, ratios
    

def clf_diagnostics(args, y_valid, valid_prob_pos, ratios):
        """
        function to check (1) classifier calibration; and (2) save weights
        """
        # assess calibration
        fraction_of_positives, mean_predicted_value = calibration_curve(y_valid, valid_prob_pos[:, 1])

        # save calibration results
        np.save(os.path.join(args.out_dir, 'fraction_of_positives'), fraction_of_positives)
        np.save(os.path.join(args.out_dir, 'mean_predicted_value.npy'), mean_predicted_value)

        # obtain figure
        plt.figure(figsize=(10,5))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='dset_clf')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        plt.title('Validation Set: Calibration Curve',fontsize=22)
        plt.ylabel('Fraction of positives',fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tick_params(axis='both', which='minor', labelsize=20)
        plt.legend()
        plt.savefig(os.path.join(args.out_dir, 'calibration_curve.pdf'))

        # save density ratios
        np.savez(
            os.path.join(args.out_dir, 'ratios.npz'), **{'x': ratios})

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.device = device

    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
    