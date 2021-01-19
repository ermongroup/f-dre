"""Train script.

Usage:
    train.py <hparams> <dataset> <dataset_root> <z_dir> --encode
"""
import os
import torch
import vision
from docopt import docopt
from torchvision import transforms
from torch.utils.data import DataLoader

from glow1.builder import build
from glow1.trainer import Trainer
from glow1.config import JsonConfig


if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset = args["<dataset>"]
    dataset_root = args["<dataset_root>"]
    z_dir = args["<z_dir>"]
    encode = args["--encode"]

    assert dataset in vision.Datasets, (
        "`{}` is not supported, use `{}`".format(dataset, vision.Datasets.keys()))
    assert os.path.exists(dataset_root), (
        "Failed to find root dir `{}` of dataset.".format(dataset_root))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))
    hparams = JsonConfig(hparams)
    dataset = vision.Datasets[dataset]
    # set transform of dataset
    transform = transforms.Compose([
        transforms.CenterCrop(hparams.Data.center_crop),
        transforms.Resize(hparams.Data.resize),
        transforms.ToTensor()])
    
    is_train = not encode
    # build graph and dataset
    built = build(hparams, is_train)
    dataset = dataset(dataset_root, transform=transform)

    if is_train:
        # begin to train
        trainer = Trainer(**built, dataset=dataset, hparams=hparams)
        trainer.train()
    elif encode:
        from copy import deepcopy

        # pretrained model
        model = built["graph"]

        loader = DataLoader(dataset, batch_size=hparams.Train.batch_size)
        ys = []
        idx = 1
        save_dir = os.path.join(z_dir, 'encodings')
        os.makedirs(save_dir, exist_ok=True)
        for i, batch in enumerate(loader):
            if i % 20 == 0:
                print(f'Encoding batch #{i}')
            x_batch = batch["x"]
            print('x_batch.shape: ', x_batch.shape)
            y_batch = batch["y_onehot"]
            for x in x_batch:
                z = model.generate_z(x)
                save_file = os.path.join(save_dir, f'{idx}.pt')
                torch.save(z, save_file)
                idx += 1
            ys.append(deepcopy(y_batch))
        save_file = os.path.join(z_dir, f'labels.pt')
        ys = torch.cat(ys)
        torch.save(ys, save_file)

