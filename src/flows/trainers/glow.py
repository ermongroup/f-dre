"""
Glow: Generative Flow with Invertible 1x1 Convolutions
arXiv:1807.03039v2
"""

import torch
import torch.nn as nn
import torch.distributions as D
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

from torchvision.datasets import MNIST
import sys
sys.path.append('../')
from src.datasets.celeba import CelebA

from src.models.glow.glow import Glow

import numpy as np
from tensorboardX import SummaryWriter

import os
import time
import math
import argparse
import pprint


parser = argparse.ArgumentParser()
# action
parser.add_argument('--train', action='store_true', help='Train a flow.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate a flow.')
parser.add_argument('--encode', action='store_true', help='Save data z-encodings using a trained model.')
parser.add_argument('--generate', action='store_true', help='Generate samples from a model.')
parser.add_argument('--fair-generate', action='store_true', help='Generate samples from a model using importance weights.')
parser.add_argument('--visualize', action='store_true', help='Visualize manipulated attribures.')
parser.add_argument('--restore_file', type=str, help='Path to model to restore.')
parser.add_argument('--seed', type=int, help='Random seed to use.')
# paths and reporting
parser.add_argument('--data_dir', default='/mnt/disks/data/', help='Location of datasets.')
parser.add_argument('--output_dir', default='./results/{}'.format(os.path.splitext(__file__)[0]))
parser.add_argument('--results_file', default='results.txt', help='Filename where to store settings and test results.')
parser.add_argument('--log_interval', type=int, default=2, help='How often to show loss statistics and save samples.')
parser.add_argument('--save_interval', type=int, default=50, help='How often to save during training.')
parser.add_argument('--eval_interval', type=int, default=1, help='Number of epochs to eval model and save model checkpoint.')
# data
parser.add_argument('--dataset', type=str, help='Which dataset to use.')
# model parameters
parser.add_argument('--depth', type=int, default=32, help='Depth of the network (cf Glow figure 2).')
parser.add_argument('--n_levels', type=int, default=3, help='Number of levels of of the network (cf Glow figure 2).')
parser.add_argument('--width', type=int, default=512, help='Dimension of the hidden layers.')
parser.add_argument('--z_std', type=float, help='Pass specific standard devition during generation/sampling.')
# training params
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
parser.add_argument('--batch_size_init', type=int, default=256, help='Batch size for the data dependent initialization.')
parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--n_epochs_warmup', type=int, default=2, help='Number of warmup epochs for linear learning rate annealing.')
parser.add_argument('--start_epoch', default=0, help='Starting epoch (for logging; to be overwritten when restoring file.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--mini_data_size', type=int, default=None, help='Train only on this number of datapoints.')
parser.add_argument('--grad_norm_clip', default=50, type=float, help='Clip gradients during training.')
parser.add_argument('--checkpoint_grads', action='store_true', default=False, help='Whether to use gradient checkpointing in forward pass.')
parser.add_argument('--n_bits', default=5, type=int, help='Number of bits for input images.')
# distributed training params
parser.add_argument('--distributed', action='store_true', default=False, help='Whether to use DistributedDataParallels on multiple machines and GPUs.')
parser.add_argument('--world_size', type=int, default=1, help='Number of nodes for distributed training.')
parser.add_argument('--local_rank', type=int, help='When provided, run model on this cuda device. When None, used by torch.distributed.launch utility to manage multi-GPU training.')
# visualize
parser.add_argument('--vis_img', type=str, help='Path to image file to manipulate attributes and visualize.')
parser.add_argument('--vis_attrs', nargs='+', type=int, help='Which attribute to manipulate.')
parser.add_argument('--vis_alphas', nargs='+', type=float, help='Step size on the manipulation direction.')

best_eval_logprob = float('-inf')


# --------------------
# Data
# --------------------

def fetch_dataloader(args, train=True, data_dependent_init=False):
    args.input_dims = {'mnist': (3,32,32), 'celeba': (3,64,64)}[args.dataset]

    transforms = {'mnist': T.Compose([T.Pad(2),                                         # image to 32x32 same as CIFAR
                                      T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # random shifts to fill the padded pixels
                                      T.ToTensor(),
                                      T.Lambda(lambda t: t + torch.rand_like(t)/2**8),  # dequantize
                                      T.Lambda(lambda t: t.expand(3,-1,-1))]),          # expand to 3 channels

                  'celeba': T.Compose([T.CenterCrop(148),  # RealNVP preprocessing
                                       T.Resize(64),
                                       T.Lambda(lambda im: np.array(im, dtype=np.float32)),                     # to numpy
                                       T.Lambda(lambda x: np.floor(x / 2**(8 - args.n_bits)) / 2**args.n_bits), # lower bits
                                       T.ToTensor(),  # note: if input to this transform is uint8, it divides by 255 and returns float
                                       T.Lambda(lambda t: t + torch.rand_like(t) / 2**args.n_bits)])            # dequantize
                  }[args.dataset]

    dataset = {'mnist': MNIST, 'celeba': CelebA}[args.dataset]

    # load the specific dataset
    dataset = dataset(root=args.data_dir, train=train, transform=transforms)

    if args.mini_data_size:
        dataset.data = dataset.data[:args.mini_data_size]

    # load sampler and dataloader
    if args.distributed and train is True and not data_dependent_init:  # distributed training; but exclude initialization
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    batch_size = args.batch_size_init if data_dependent_init else args.batch_size  # if data dependent init use init batch size
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.device.type is 'cuda' else {}
    return DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), drop_last=True, sampler=sampler, **kwargs)


# --------------------
# Train and evaluate
# --------------------

@torch.no_grad()
def data_dependent_init(model, args):
    # set up an iterator with batch size = batch_size_init and run through model
    dataloader = fetch_dataloader(args, train=True, data_dependent_init=True)
    model(next(iter(dataloader))[0].requires_grad_(True if args.checkpoint_grads else False).to(args.device))
    del dataloader
    return True

def train_epoch(model, dataloader, optimizer, writer, epoch, args):
    model.train()

    tic = time.time()
    for i, (x,y) in enumerate(dataloader):
        args.step += args.world_size
        # warmup learning rate
        if epoch <= args.n_epochs_warmup:
            optimizer.param_groups[0]['lr'] = args.lr * min(1, args.step / (len(dataloader) * args.world_size * args.n_epochs_warmup))
        
        x = x.requires_grad_(True if args.checkpoint_grads else False).to(args.device)  # requires_grad needed for checkpointing

        # print('reached iteration', i)
        loss = - model.log_prob(x, bits_per_pixel=True).mean(0)

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)

        optimizer.step()

        # report stats
        if i % args.log_interval == 0:
            # compute KL divergence between base and each of the z's that the model produces
            with torch.no_grad():
                zs, _ = model(x)
                kls = [D.kl.kl_divergence(D.Normal(z.mean(), z.std()), model.base_dist) for z in zs]

            # write stats
            if args.on_main_process:
                et = time.time() - tic              # elapsed time
                tt = len(dataloader) * et / (i+1)   # total time per epoch
                print('Epoch: [{}/{}][{}/{}]\tStep: {}\tTime: elapsed {:.0f}m{:02.0f}s / total {:.0f}m{:02.0f}s\tLoss {:.4f}\t'.format(
                      epoch, args.start_epoch + args.n_epochs, i+1, len(dataloader), args.step, et//60, et%60, tt//60, tt%60, loss.item()))
                # update writer
                for j, kl in enumerate(kls):
                    writer.add_scalar('kl_level_{}'.format(j), kl.item(), args.step)
                writer.add_scalar('train_bits_x', loss.item(), args.step)

        # save and generate
        if i % args.save_interval == 0:
            # generate samples
            samples = generate(model, n_samples=4, z_stds=[0., 0.25, 0.7, 1.0])
            images = make_grid(samples.cpu(), nrow=4, pad_value=1)

            # write stats and save checkpoints
            if args.on_main_process:
                save_image(images, os.path.join(args.output_dir, 'generated_sample_{}.png'.format(args.step)))

                # save training checkpoint
                torch.save({'epoch': epoch,
                            'global_step': args.step,
                            'state_dict': model.state_dict()},
                            os.path.join(args.output_dir, 'checkpoint.pt'))
                torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'optim_checkpoint.pt'))



@torch.no_grad()
def evaluate(model, dataloader, args):
    model.eval()
    print('Evaluating ...', end='\r')

    logprobs = []
    for x,y in dataloader:
        x = x.to(args.device)
        logprobs.append(model.log_prob(x, bits_per_pixel=True))
    logprobs = torch.cat(logprobs, dim=0).to(args.device)
    logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.std(0) / math.sqrt(len(dataloader.dataset))
    return logprob_mean, logprob_std

@torch.no_grad()
def generate(model, n_samples, z_stds):
    model.eval()
    print('Generating ...', end='\r')

    samples = []
    for z_std in z_stds:
        sample, _ = model.inverse(batch_size=n_samples, z_std=z_std)
        log_probs = model.log_prob(sample, bits_per_pixel=True)
        samples.append(sample[log_probs.argsort().flip(0)])  # sort by log_prob; flip high (left) to low (right)
    return torch.cat(samples,0)

def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, writer, args):
    global best_eval_logprob

    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        train_epoch(model, train_dataloader, optimizer, writer, epoch, args)

        # evaluate
        if False:#epoch % args.eval_interval == 0:
            eval_logprob_mean, eval_logprob_std = evaluate(model, test_dataloader, args)
            print('Evaluate at epoch {}: bits_x = {:.3f} +/- {:.3f}'.format(epoch, eval_logprob_mean, eval_logprob_std))

            # save best state
            if args.on_main_process and eval_logprob_mean > best_eval_logprob:
                best_eval_logprob = eval_logprob_mean
                torch.save({'epoch': epoch,
                            'global_step': args.step,
                            'state_dict': model.state_dict()},
                            os.path.join(args.output_dir, 'best_model_checkpoint.pt'))



def save_encodings(model, train_loader, test_loader, data_dir, dataset):
    '''
    Given trained model, save train and test encodings of dataset
    '''
    from copy import deepcopy

    model.eval()

    ys = []
    idx = 1
    save_folder = os.path.join(data_dir, f'encoded_{dataset}')
    os.makedirs(save_folder, exist_ok=True)
    for split, loader in zip(('train', 'test'), (train_loader, test_loader)):
        for i, (x,y) in enumerate(loader):
            # if i % 100 == 0:
            #     print(f'Encoding batch [{i+1}/{len(dataloader)}]')
            x = x.to(args.device)
            zs, _ = model(x)
            for z in zs:
                save_path = os.path.join(save_folder, f'{split}_encodings', f'{idx}.pt')
                torch.save(z, save_path)
                idx += 1
            ys.append(deepcopy(y))
    
        save_path = os.path.join(save_folder, f'{split}_encodings_labels.pt')
        ys = torch.cat(ys, dim=0)
        torch.save(ys, save_path)
        print(f'Encoding of {dataset} {split} set completed.')


# --------------------
# Visualizations
# --------------------

def encode_dataset(model, dataloader):
    model.eval()

    zs = []
    attrs = []
    for i, (x,y) in enumerate(dataloader):
        print('Encoding [{}/{}]'.format(i+1, len(dataloader)), end='\r')
        x = x.to(args.device)
        zs_i, _ = model(x)
        zs.append(torch.cat([z.flatten(1) for z in zs_i], dim=1))
        attrs.append(y)

    zs = torch.cat(zs, dim=0)
    attrs = torch.cat(attrs, dim=0)
    print('Encoding completed.')
    return zs, attrs

def compute_dz(zs, attrs, idx):
    """ for a given attribute idx, compute the mean for all encoded z's corresponding to the positive and negative attribute """
    z_pos = [zs[i] for i in range(len(zs)) if attrs[i][idx] == +1]
    z_neg = [zs[i] for i in range(len(zs)) if attrs[i][idx] == -1]
    # dz = z_pos - z_neg; where z_pos is mean of all encoded datapoints where attr is present;
    return torch.stack(z_pos).mean(0) - torch.stack(z_neg).mean(0)   # out tensor of shape (flattened zs dim,)

def get_manipulators(zs, attrs):
    """ compute dz (= z_pos - z_neg) for each attribute """
    print('Extracting manipulators...', end=' ')
    dzs = 1.6 * torch.stack([compute_dz(zs, attrs, i) for i in range(attrs.shape[1])], dim=0)  # compute dz for each attribute official code multiplies by 1.6 scalar here
    print('Completed.')
    return dzs  # out (n_attributes, flattened zs dim)

def manipulate(model, z, dz, z_std, alpha):
    # 1. record incoming shapes
    z_dims   = [z_.squeeze().shape   for z_ in z]
    z_numels = [z_.numel() for z_ in z]
    # 2. flatten z into a vector and manipulate by alpha in the direction of dz
    z = torch.cat([z_.flatten(1) for z_ in z], dim=1).to(dz.device)
    z = z + dz * torch.tensor(alpha).float().view(-1,1).to(dz.device)  # out (n_alphas, flattened zs dim)
    # 3. reshape back to z shapes from each level of the model
    zs = [z_.view((len(alpha), *dim)) for z_, dim in zip(z.split(z_numels, dim=1), z_dims)]
    # 4. decode
    return model.inverse(zs, z_std=z_std)[0]

def load_manipulators(model, args):
    # construct dataloader with limited number of images
    args.mini_data_size = 30000
    # load z manipulators for each attribute
    if os.path.exists(os.path.join(args.output_dir, 'z_manipulate.pt')):
        z_manipulate = torch.load(os.path.join(args.output_dir, 'z_manipulate.pt'), map_location=args.device)
    else:
        # encode dataset, compute manipulators, store zs, attributes, and dzs
        dataloader = fetch_dataloader(args, train=True)
        zs, attrs = encode_dataset(model, dataloader)
        z_manipulate = get_manipulators(zs, attrs)
        torch.save(zs, os.path.join(args.output_dir, 'zs.pt'))
        torch.save(attrs, os.path.join(args.output_dir, 'attrs.pt'))
        torch.save(z_manipulate, os.path.join(args.output_dir, 'z_manipulate.pt'))
    return z_manipulate

@torch.no_grad()
def visualize(model, args, attrs=None, alphas=None, img_path=None, n_examples=1):
    """ manipulate an input image along a given attribute """
    dataset = fetch_dataloader(args, train=False).dataset  # pull the dataset to access transforms and attrs
    # if no attrs passed, manipulate all of them
    if not attrs:
        attrs = list(range(len(dataset.attr_names)))
    # if image is passed, manipulate only the image
    if img_path:
        from PIL import Image
        img = Image.open(img_path)
        x = dataset.transform(img)  # transform image to tensor and encode
    else:  # take first n_examples from the dataset
        x, _ = dataset[0]
    z, _ = model(x.unsqueeze(0).to(args.device))
    # get manipulors
    z_manipulate = load_manipulators(model, args)
    # decode the varied attributes
    dec_x =[]
    for attr_idx in attrs:
        dec_x.append(manipulate(model, z, z_manipulate[attr_idx].unsqueeze(0), args.z_std, alphas))
    return torch.stack(dec_x).cpu()


# --------------------
# Main
# --------------------

if __name__ == '__main__':
    args = parser.parse_args()
    args.step = 0  # global step
    args.output_dir = os.path.dirname(args.restore_file) if args.restore_file else os.path.join(args.output_dir, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
    writer = None  # init as None in case of multiprocessing; only main process performs write ops

    # setup device and distributed training
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device('cuda:{}'.format(args.local_rank))

        # initialize
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

        # compute total world size (used to keep track of global step)
        args.world_size = int(os.environ['WORLD_SIZE'])  # torch.distributed.launch sets this to nproc_per_node * nnodes
    else:
        if torch.cuda.is_available(): args.local_rank = 0
        args.device = torch.device('cuda:{}'.format(args.local_rank) if args.local_rank is not None else 'cpu')

    # write ops only when on_main_process
    # NOTE: local_rank unique only to the machine; only 1 process on each node is on_main_process;
    #       if shared file system, args.local_rank below should be replaced by global rank e.g. torch.distributed.get_rank()
    args.on_main_process = (args.distributed and args.local_rank == 0) or not args.distributed

    # setup seed
    if args.seed:
        torch.manual_seed(args.seed)
        if args.device.type == 'cuda': torch.cuda.manual_seed(args.seed)

    # load data; sets args.input_dims needed for setting up the model
    train_dataloader = fetch_dataloader(args, train=True)
    test_dataloader = fetch_dataloader(args, train=False)

    # load model
    model = Glow(args.width, args.depth, args.n_levels, args.input_dims, args.checkpoint_grads).to(args.device)
    if args.distributed:
        # NOTE: DistributedDataParallel will divide and allocate batch_size to all available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        # for compatibility of saving/loading models, wrap non-distributed cpu/gpu model as well;
        # ie state dict is based on model.module.layer keys, which now match between training distributed and running then locally
        model = torch.nn.parallel.DataParallel(model)
    # DataParallel and DistributedDataParallel are wrappers around the model; expose functions of the model directly
    model.base_dist = model.module.base_dist
    model.log_prob = model.module.log_prob
    model.inverse = model.module.inverse

    # load optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load checkpoint if provided
    if args.restore_file:
        model_checkpoint = torch.load(args.restore_file, map_location=args.device)
        model.load_state_dict(model_checkpoint['state_dict'])
        optimizer.load_state_dict(torch.load(os.path.dirname(args.restore_file) + '/optim_checkpoint.pt', map_location=args.device))
        args.start_epoch = model_checkpoint['epoch']
        args.step = model_checkpoint['global_step']

    # setup writer and outputs
    if args.on_main_process:
        writer = SummaryWriter(log_dir = args.output_dir)

        # save settings
        config = 'Parsed args:\n{}\n\n'.format(pprint.pformat(args.__dict__)) + \
                 'Num trainable params: {:,.0f}\n\n'.format(sum(p.numel() for p in model.parameters())) + \
                 'Model:\n{}'.format(model)
        config_path = os.path.join(args.output_dir, 'config.txt')
        writer.add_text('model_config', config)
        if not os.path.exists(config_path):
            with open(config_path, 'a') as f:
                print(config, file=f)

    if args.train:
        # run data dependent init and train
        data_dependent_init(model, args)
        train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, writer, args)

    if args.evaluate:
        logprob_mean, logprob_std = evaluate(model, test_dataloader, args)
        print('Evaluate: bits_x = {:.3f} +/- {:.3f}'.format(logprob_mean, logprob_std))

    if args.generate:
        n_samples = 4
        z_std = [0., 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] if not args.z_std else n_samples * [args.z_std]
        samples = generate(model, n_samples, z_std)
        images = make_grid(samples.cpu(), nrow=n_samples, pad_value=1)
        save_image(images, os.path.join(args.output_dir,
                                        'generated_samples_at_z_std_{}.png'.format('range' if args.z_std is None else args.z_std)))

    if args.encode:
        assert args.restore_file is not None, "Must specify --restore_file to encode dataset."
        save_encodings(model, train_dataloader, test_dataloader, args.data_dir, args.dataset)
    if args.visualize:
        if not args.z_std: args.z_std = 0.6
        if not args.vis_alphas: args.vis_alphas = [-2,-1,0,1,2]
        dec_x = visualize(model, args, args.vis_attrs, args.vis_alphas, args.vis_img)   # output (n_attr, n_alpha, 3, H, W)
        filename = 'manipulated_sample' if not args.vis_img else \
                   'manipulated_img_{}'.format(os.path.basename(args.vis_img).split('.')[0])
        if args.vis_attrs:
            filename += '_attr_' + ','.join(map(str, args.vis_attrs))
        save_image(dec_x.view(-1, *args.input_dims), os.path.join(args.output_dir, filename + '.png'), nrow=dec_x.shape[1])

    if args.on_main_process:
        writer.close()
