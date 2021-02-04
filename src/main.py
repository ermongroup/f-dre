import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
sys.path.append(os.path.abspath(os.getcwd()))

from flows.trainers.flow import Flow
from flows.trainers.toy_flow import ToyFlow
from classification.trainers.classifier import Classifier
from classification.trainers.old_classifier import OldClassifier
from classification.trainers.attr_classifier import AttrClassifier
from classification.trainers.mi_classifier import MIClassifier
from classification.trainers.downstream_classifier import DownstreamClassifier
from classification.trainers.omniglot_downstream_clf import OmniglotDownstreamClassifier

import getpass
import wandb

WANDB = {
    'kechoi': 'kristychoi',
    'madeline': 'madeline'
}

torch.set_printoptions(sci_mode=False)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    # ======== Data and output ========
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--exp_id', type=str, help='Exp id; output will be saved in config.data.out_dir/exp_id')
    parser.add_argument('--ni', action='store_true', help='No interaction. Suitable for Slurm Job launcher',)
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical',)

    # ======== Flow model task ========
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--test', action='store_true', help='Whether to test the model; if not true, trains model')
    parser.add_argument('--sample', action='store_true', help='Whether to produce samples from a pretrained model')
    parser.add_argument('--encode_z', action='store_true', help='Whether to encode data using a pretrained model')
    parser.add_argument('--restore_file', default=None, help='If restoring a pretrained flow checkpoint, path to saved model')

    # ======== Sampling: must specify --sample and one of following sampling procedures ========
    parser.add_argument('--generate_samples', action='store_true', help='Regular sampling from flow')
    parser.add_argument('--fair_generate', action='store_true', help='Sample with DRE reweighting using SIR. Must specify --dre_clf_ckpt.')
    parser.add_argument('--fid', action='store_true', help='FID sampling')
    
    # ======== Sampling classifiers: --attr_clf_ckpt for classifying sample attributes, --dre_clf_ckpt for --fair-generate ========
    parser.add_argument('--attr_clf_ckpt', default=None, help='Path to pretrained attribute classifier checkpoint; if provided, classify the flow samples.')
    parser.add_argument('--dre_clf_ckpt', default=None, help='Path to pretrained DRE classifier checkpoint to use for reweighting.')
    
    # ======== Classification-related (TODO: integrate classifier and flow into same main.py?) ========
    parser.add_argument('--classify', action='store_true', help='To run classification')
    parser.add_argument('--attr', default=None, help='For attr classification, provide one of \{\'background\', \'digit\'\}')
    parser.add_argument('--mi', action='store_true', help='To run MI estimation')
    parser.add_argument('--downstream', action='store_true', help='Run downstream classifier for domain adaptation experiment')
    # parser.add_argument('--dre', action='store_true', help='Run DRE classification of z-encodings')  
    
    # parse args and config
    args = parser.parse_args()
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info('Using device: {}'.format(device))
    new_config.device = device

    args.out_dir = os.path.join(new_config.training.out_dir, args.exp_id)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    args.log_path = os.path.join(args.out_dir, 'logs')
    os.makedirs(args.out_dir, exist_ok=True)
    # set up wandb
    if not args.sample:
        # only for training
        wandb.init(
            project='multi-fairgen', 
            entity=WANDB[getpass.getuser()], 
            name=args.exp_id, 
            config=new_config, 
            sync_tensorboard=True,
        )

    # set up logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s'
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    if not args.test and not args.sample: # training model
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input('Folder already exists. Overwrite? (Y/N)')
                    if response.upper() == 'Y':
                        overwrite = True
                if overwrite:
                    shutil.rmtree(args.log_path)
                    os.makedirs(args.log_path)
                else:
                    print('Folder exists. Program halted.')
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.out_dir, 'config.yaml'), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler2 = logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))
        handler2.setFormatter(formatter)
        logger.addHandler(handler2)
        logger.setLevel(level)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    if isinstance(config, list):
        # from creating config files
        for i in range(len(config)):
            for key, value in config[i].items():
                if isinstance(value, dict):
                    new_value = dict2namespace(value)
                else:
                    new_value = value
                setattr(namespace, key, new_value)
    else:
        # vanilla training
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info('Saving output to {}'.format(args.out_dir))
    logging.info('Writing log file to {}'.format(args.log_path))
    logging.info('Exp instance id = {}'.format(os.getpid()))

    try:
        if args.classify:
            if args.attr is not None:
                print('training attribute/standard (non-dre) classifier...')
                trainer = AttrClassifier(args, config)
            else:
                if args.mi:
                    print('training MI classifier...')
                    trainer = MIClassifier(args, config)
                elif args.downstream:
                    print('Training downstream classifier...')
                    if config.data.dataset == 'Omniglot':
                        print('omniglot downstream task')
                        trainer = OmniglotDownstreamClassifier(args, config)
                    else:
                        trainer = DownstreamClassifier(args, config)
                else:
                    print('Using DRE classifier...')
                    trainer = Classifier(args, config)
                    # trainer = OldClassifier(args, config)
        else:
            if config.data.dataset not in ['KMM', 'GMM', 'GMM_flow', 'MI', 'MI_flow']:
                trainer = Flow(args, config)
            else:
                trainer = ToyFlow(args, config)
        
        if args.sample:
            trainer.sample(args)
        elif args.test:
            trainer.test()
        else:
            trainer.train()
            if args.classify and config.data.dataset != 'Omniglot':
                # test_loss, test_acc, test_labels, test_probs, test_ratios, test_data = trainer.test(trainer.test_dataloader, 'test')
                # trainer.clf_diagnostics(test_labels, test_probs, test_ratios, test_data, 'test')
                test_loss, test_acc, test_labels, test_probs, test_ratios, test_data = trainer.test(trainer.test_dataloader, 'test')
                trainer.clf_diagnostics(test_labels, test_probs, test_ratios, test_data, 'test')
    
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())