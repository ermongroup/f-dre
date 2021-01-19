# https://github.com/ermongroup/ncsnv2
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from functools import partial
from src.classification.models.layers import *
from src.classification.models.normalizers import get_normalization


class ResnetClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args  # deprecate this
        # self.norm = get_normalization(config, conditional=False)
        self.norm = nn.GroupNorm
        self.ngf = ngf = 64
        self.act = act = nn.SiLU()
        self.num_scales = self.ngf  # TODO
        self.channels = 1

        self.begin_conv = nn.Conv2d(self.channels, ngf, 3, stride=1, padding=1)

        self.normalizer = self.norm(ngf, self.num_scales)
        self.end_conv = nn.Conv2d(ngf, self.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)]
        )

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act, normalization=self.norm, adjust_padding=True, dilation=4),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act, normalization=self.norm, dilation=4)]
        )

        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

        self.fc = nn.Linear(784, 2)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x):
        h = x
        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        # classification
        output = output.view(len(x), -1)
        logits = self.fc(output)
        probas = F.softmax(logits, dim=1)

        return logits, probas