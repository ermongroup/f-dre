import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifierv2(nn.Module):
  """
  simple MLP classifier (e.g. for classifying in z-space)
  """
  def __init__(self, config):
      super(MLPClassifierv2, self).__init__()
      self.config = config
      self.h_dim = config.model.h_dim
      self.n_classes = config.model.n_classes
      self.in_dim = config.model.in_dim

      self.fc1 = nn.Linear(self.in_dim, self.h_dim)
      self.fc2 = nn.Linear(self.h_dim, self.h_dim)
      self.fc3 = nn.Linear(self.h_dim, self.h_dim)
      self.fc4 = nn.Linear(self.h_dim, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    logits = self.fc4(x)
    probas = torch.sigmoid(logits)

    return logits, probas


class TREMLPClassifier(nn.Module):
  def __init__(self, config):
    super(TREMLPClassifier, self).__init__()
    self.config = config
    self.h_dim = config.model.h_dim
    self.n_classes = config.model.n_classes
    self.in_dim = config.model.in_dim

    # m = number of bridges (intermediate ratios)
    self.m = config.tre.m
    self.p = config.tre.p

    self.fc1 = nn.Linear(self.in_dim, self.h_dim)
    self.fc2 = nn.Linear(self.h_dim, self.h_dim)
    self.fc3 = nn.Linear(self.h_dim, self.h_dim)

    # bridge-specific heads
    self.fc4s = nn.ModuleList(
          [nn.Linear(self.h_dim, 1) for _ in range(self.m)])
    
    # do we need this?
    # for w in self.fc4s:
    #     nn.init.xavier_normal_(w.weight)

  def forward(self, x):
    '''
    Returns logits, probas where len(logits) = len(probas) = m
    '''

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))

    # separate xs into m chunks
    # xs = [torch.cat([x[i], x[i+1]]) for i in range(self.m)]
    xs = [x for _ in range(self.m)]
    logits = []
    probas = []
    for x, fc4 in zip(xs, self.fc4s):
      curr_logits = F.relu(fc4(x)) # quadratic head
      curr_probas = torch.sigmoid(curr_logits)

      logits.append(curr_logits)

    logits = torch.stack(logits)
    probas = torch.sigmoid(logits)

    return logits, probas


class MLPClassifier(nn.Module):
  """
  simple MLP classifier (e.g. for classifying in z-space)
  """
  def __init__(self, config):
      super(MLPClassifier, self).__init__()
      self.config = config
      self.h_dim = config.model.h_dim
      self.n_classes = config.model.n_classes
      self.in_dim = config.model.in_dim

      self.fc1 = nn.Linear(self.in_dim, self.h_dim)
      self.fc2 = nn.Linear(self.h_dim, self.h_dim)
      self.fc3 = nn.Linear(self.h_dim, self.n_classes)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    logits = self.fc3(x)
    probas = F.softmax(logits, dim=1)

    return logits, probas