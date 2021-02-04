import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from synthetic_clf import Classifier


class TREClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_dim = config.data.channels
        self.m = config.dre.m
        self.p = config.dre.p
        self.spectral_norm = config.model.spectral_norm

        # data-dependent params
        self.mu_0 = config.data.mus[:-1]
        self.mu_m = config.data.mus[-1]


        # multi-head logistic regressor
        self.linear_fc = nn.ModuleList(
            [nn.Linear(1, 1) for _ in range(self.m)])
        for w in self.linear_fc:
            nn.init.xavier_normal_(w.weight)

    def forward(self, x):
        """Summary
        """
        # construct pairs per minibatch
        xs = [torch.cat([x[i], x[i+1]]).unsqueeze(1) for i in range(self.m)]

        # right now testing out binary cross entropy with logits
        out = []
        for i, fc in enumerate(self.linear_fc):
            out.append(fc(xs[i]**2))
        out = torch.stack(out)
        return out

    def loss(self, x, out, y):
        """Summary
        """
        raise NotImplementedError