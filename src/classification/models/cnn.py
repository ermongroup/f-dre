# https://github.com/BoyuanJiang/matching-networks-pytorch/blob/master/matching_networks.py

import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F


def convLayer(in_channels, out_channels, keep_prob=0.0):
    """3*3 convolution with padding,ever time call it the output size become half"""
    cnn_seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
        nn.MaxPool2d(2)
        # nn.Dropout(keep_prob)
    )
    return cnn_seq


class CNNClassifier(nn.Module):
    def __init__(self, layer_size=64, num_channels=1, keep_prob=1.0, image_size=28):
        super(CNNClassifier, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:
        """
        self.layer1 = convLayer(num_channels, layer_size, keep_prob)
        self.layer2 = convLayer(layer_size, layer_size, keep_prob)
        self.layer3 = convLayer(layer_size, layer_size, keep_prob)
        self.layer4 = convLayer(layer_size, layer_size, keep_prob)

        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size

        # TODO: how many classes???
        self.fc = nn.Linear(self.outSize, 1622)

    def forward(self, image_input):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size()[0], -1)

        logits = self.fc(x)
        probas = F.softmax(logits, dim=-1)

        return logits, probas


<<<<<<< Updated upstream
class BinaryCNNClassifier(nn.Module):
    def __init__(self, layer_size=64, num_channels=1, keep_prob=1.0, image_size=28):
        super(BinaryCNNClassifier, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:
        """
=======
class TRECNNClassifier(nn.Module):
    def __init__(self, config, layer_size=64, num_channels=1, keep_prob=1.0, image_size=28):
        super(CNNClassifier, self).__init__()
        
        self.m = self.config.tre.m
        self.p = self.config.tre.p

>>>>>>> Stashed changes
        self.layer1 = convLayer(num_channels, layer_size, keep_prob)
        self.layer2 = convLayer(layer_size, layer_size, keep_prob)
        self.layer3 = convLayer(layer_size, layer_size, keep_prob)
        self.layer4 = convLayer(layer_size, layer_size, keep_prob)

        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size

<<<<<<< Updated upstream
        self.fc = nn.Linear(self.outSize, 1)
=======
        self.fc_list = [nn.Linear(self.outSize, 1622) for _ in range(self.m)]
>>>>>>> Stashed changes

    def forward(self, image_input):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
<<<<<<< Updated upstream
=======
        # x = x.view(-1, ) TODO
>>>>>>> Stashed changes
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
<<<<<<< Updated upstream
        x = x.view(x.size()[0], -1)

        logits = self.fc(x)
        probas = torch.sigmoid(logits)
=======
        x = x.view(self.m, x.size()[0], -1) # TODO
        
        logits = []
        for x_i, fc in zip(x, self.fc_list):
            logits.append(fc(x_i))

        logits = torch.stack(logits)
        probas = F.softmax(logits, dim=-1)
>>>>>>> Stashed changes

        return logits, probas