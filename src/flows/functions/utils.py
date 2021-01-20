# TODO: (1) classifier outputs for attribute labels; (2) FID scores
# NOTE: this code is untested and needs to be adapted for our setup!
import math
import functools
import numpy as np
from tqdm import tqdm, trange
import os
import glob
import pickle

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import torchvision


def get_ratio_estimates(model, z):
    """
    get density ratio estimates in z-space
    """
    model.eval()
    logits, probas = model(z)
    ratios = probas[:,1]/probas[:,0]

    return ratios


def fairness_discrepancy(data, n_classes):
    """
    computes fairness discrepancy metric for single or multi-attribute
    this metric computes L2, L1, AND KL-total variation distance
    """
    unique, freq = torch.unique(data, return_counts=True)
    props = freq / len(data)
    truth = 1./n_classes

    # L2 and L1
    l2_fair_d = math.sqrt(((props - truth)**2).sum())
    l1_fair_d = abs(props - truth).sum()

    # q = props, p = truth
    kl_fair_d = (props * (math.log(props) - math.log(truth))).sum()
    
    return l2_fair_d, l1_fair_d, kl_fair_d


def classify_examples(model, x):
    model.eval()

    logits, probas = model(x)
    _, pred = torch.max(probas, 1)

    return preds, probas