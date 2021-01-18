import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
  """
  simple MLP classifier (e.g. for classifying in z-space)
  """
  def __init__(self, args):
      super(MLPClassifier, self).__init__()
      self.args = args
      self.h_dim = args.h_dim
      self.n_classes = args.n_classes
      self.in_dim = args.pixels
      self.dropout = nn.Dropout(args.dropout)
      self.bn = nn.BatchNorm1d(self.h_dim)

      # TODO: don't hardcode to celeba
      self.fc1 = nn.Linear(self.in_dim, self.h_dim)
      self.fc2 = nn.Linear(self.h_dim, self.h_dim)
      self.fc3 = nn.Linear(self.h_dim, self.n_classes)

  def forward(self, x):
    x = self.dropout(F.relu(self.bn(self.fc1(x))))
    x = self.dropout(F.relu(self.bn(self.fc2(x))))
    logits = self.fc3(x)
    probas = F.softmax(logits, dim=1)

    return logits, probas