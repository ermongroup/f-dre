import os
import yaml


# alphas = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# alphas = [0.01, 0.02, 0.05, 0.07]
alphas = [0.001, 0.005, 0.007]
for alpha in alphas:
    exp_id = 'joint_gmm_flow_mlp_perc1.0_alpha{}'.format(alpha)

    dict_file = [
    {'training': {
    'batch_size': 128,
    'n_epochs': 200,
    'ngpu': 1,
    'iter_log': 1000,
    'iter_save': 100,
    'exp_id': exp_id,
    'out_dir': "/atlas/u/kechoi/multi-fairgen/src/classification/results/",
    'data_dir': "/atlas/u/kechoi/multi-fairgen/data/"
    }},
    {'data': {
    'dataset': "GMM",
    'subset': False,
    'x_space': True,
    'input_size': 2,
    'perc': 1.0,
    'mus': [0, 3],
    'class_idx': 20,
    'num_workers': 4,
    }},
    {'model': {
    'name': "flow_mlp",
    'spectral_norm': True,
    'batch_norm': True,
    'in_dim': 2,
    'h_dim': 200,
    'dropout': 0.1,
    'n_classes': 2,
    }},
    {'optim': {
    'weight_decay': 0.000001,
    'optimizer': "Adam",
    'lr': 0.0001,
    'beta1': 0.9,
    'amsgrad': False,
    }},
    {'loss': {
    'name': "joint",
    'alpha': alpha,
    }}
    ]

    # filenames
    fname = exp_id = 'joint_flow_mlp_perc1.0_alpha{}.yaml'.format(alpha)
    fpath = os.path.join('/atlas/u/kechoi/multi-fairgen/src/configs/classification/gmm/joint_sweep/', fname)

    with open(fpath, 'w') as file:
        documents = yaml.dump(dict_file, file)