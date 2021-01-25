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


class OldMLPClassifier(nn.Module):
  """
  simple MLP classifier (e.g. for classifying in z-space)
  """
  def __init__(self, config):
      super(OldMLPClassifier, self).__init__()
      self.config = config
      self.h_dim = config.model.h_dim
      self.n_classes = config.model.n_classes
      self.in_dim = config.model.in_dim
      self.dropout = nn.Dropout(config.model.dropout)
      self.bn = nn.BatchNorm1d(self.h_dim)

      self.fc1 = nn.Linear(self.in_dim, self.h_dim)
      self.fc2 = nn.Linear(self.h_dim, self.h_dim)
      self.fc3 = nn.Linear(self.h_dim, self.n_classes)

  def forward(self, x):
    x = self.dropout(F.relu(self.bn(self.fc1(x))))
    x = self.dropout(F.relu(self.bn(self.fc2(x))))
    logits = self.fc3(x)
    probas = F.softmax(logits, dim=1)

    return logits, probas