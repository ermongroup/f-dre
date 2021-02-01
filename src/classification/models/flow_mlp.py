import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.flows.models.maf import MAF


class FlowClassifier(nn.Module):
  """
  Deeper classifier that uses a normalizing flow to map data points into z-space before performing classification.
  """
  def __init__(self, config):
    super(FlowClassifier, self).__init__()
    self.config = config
    self.h_dim = config.model.h_dim
    self.n_classes = config.model.n_classes
    self.in_dim = config.model.in_dim

    # HACK: hardcoded flow architecture that we've been using!
    # TODO: fix flow architecture to vary with dataset size
    if 'CIFAR' in self.config.data.dataset:
      self.flow = MAF(5, self.in_dim, 1024, 2, None, 'relu', 'sequential', batch_norm=True)
    else:
      self.flow = MAF(5, self.in_dim, 100, 1, None, 'relu', 'sequential', batch_norm=True)

    self.fc1 = nn.Linear(self.in_dim, self.h_dim)
    self.fc2 = nn.Linear(self.h_dim, self.h_dim)
    self.fc3 = nn.Linear(self.h_dim, self.h_dim)
    self.fc4 = nn.Linear(self.h_dim, 1)

  def forward(self, x):
    # map data into z-space
    z, _ = self.flow.forward(x)

    # then train classifier
    z = F.relu(self.fc1(z))
    z = F.relu(self.fc2(z))
    z = F.relu(self.fc3(z))
    logits = self.fc4(z)
    probas = torch.sigmoid(logits)

    return logits, probas

  @torch.no_grad()
  def flow_encode(self, x):
    z, _ = self.flow.forward(x)
    return z