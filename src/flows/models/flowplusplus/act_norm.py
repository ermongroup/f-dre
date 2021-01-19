import torch
import torch.nn as nn

from util import mean_dim


class _BaseNorm(nn.Module):
    """Base class for ActNorm (Glow) and PixNorm (Flow++).

    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.

    Adapted from:
        > https://github.com/openai/glow
    """
    def __init__(self, num_channels, height, width):
        super(_BaseNorm, self).__init__()

        # Input gets concatenated along channel axis
        num_channels *= 2

        self.register_buffer('is_initialized', torch.zeros(1))
        self.mean = nn.Parameter(torch.zeros(1, num_channels, height, width))
        self.inv_std = nn.Parameter(torch.zeros(1, num_channels, height, width))
        self.eps = 1e-6

    def initialize_parameters(self, x):
        if not self.training:
            return

        with torch.no_grad():
            mean, inv_std = self._get_moments(x)
            self.mean.data.copy_(mean.data)
            self.inv_std.data.copy_(inv_std.data)
            self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if reverse:
            return x + self.mean
        else:
            return x - self.mean

    def _get_moments(self, x):
        raise NotImplementedError('Subclass of _BaseNorm must implement _get_moments')

    def _scale(self, x, sldj, reverse=False):
        raise NotImplementedError('Subclass of _BaseNorm must implement _scale')

    def forward(self, x, ldj=None, reverse=False):
        x = torch.cat(x, dim=1)
        if not self.is_initialized:
            self.initialize_parameters(x)

        if reverse:
            x, ldj = self._scale(x, ldj, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, ldj, reverse)
        x = x.chunk(2, dim=1)

        return x, ldj


class ActNorm(_BaseNorm):
    """Activation Normalization used in Glow

    The mean and inv_std get initialized using the mean and variance of the
    first mini-batch. After the init, mean and inv_std are trainable parameters.
    """
    def __init__(self, num_channels):
        super(ActNorm, self).__init__(num_channels, 1, 1)

    def _get_moments(self, x):
        mean = mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
        var = mean_dim((x.clone() - mean) ** 2, dim=[0, 2, 3], keepdims=True)
        inv_std = 1. / (var.sqrt() + self.eps)

        return mean, inv_std

    def _scale(self, x, sldj, reverse=False):
        if reverse:
            x = x / self.inv_std
            sldj = sldj - self.inv_std.log().sum() * x.size(2) * x.size(3)
        else:
            x = x * self.inv_std
            sldj = sldj + self.inv_std.log().sum() * x.size(2) * x.size(3)

        return x, sldj


class PixNorm(_BaseNorm):
    """Pixel-wise Activation Normalization used in Flow++

    Normalizes every activation independently (note this differs from the variant
    used in in Glow, where they normalize each channel). The mean and stddev get
    initialized using the mean and stddev of the first mini-batch. After the
    initialization, `mean` and `inv_std` become trainable parameters.
    """
    def _get_moments(self, x):
        mean = torch.mean(x.clone(), dim=0, keepdim=True)
        var = torch.mean((x.clone() - mean) ** 2, dim=0, keepdim=True)
        inv_std = 1. / (var.sqrt() + self.eps)

        return mean, inv_std

    def _scale(self, x, sldj, reverse=False):
        if reverse:
            x = x / self.inv_std
            sldj = sldj - self.inv_std.log().sum()
        else:
            x = x * self.inv_std
            sldj = sldj + self.inv_std.log().sum()

        return x, sldj
