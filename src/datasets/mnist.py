import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt
import os

import datasets
import datasets.util as util


class MNIST:
    """
    The MNIST dataset of handwritten digits.
    """

    alpha = 1.0e-6

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data, logit, dequantize, rng, rgb):

            x = self._dequantize(data[0], rng) if dequantize else data[0]  # dequantize pixels
            self.x = self._logit_transform(x) if logit else x              # logit
            if rgb:
                self.x = np.stack([self.x, self.x, self.x], axis=1)
            self.labels = data[1]                                          # numeric labels
            self.y = util.one_hot_encode(self.labels, 10)                  # 1-hot encoded labels
            self.N = self.x.shape[0]                                       # number of datapoints

        @staticmethod
        def _dequantize(x, rng):
            """
            Adds noise to pixels to dequantize them.
            """
            return x + rng.rand(*x.shape) / 256.0

        @staticmethod
        def _logit_transform(x):
            """
            Transforms pixel values with logit to be unconstrained.
            """
            return util.logit(MNIST.alpha + (1 - 2*MNIST.alpha) * x)

    def __init__(self, logit=True, dequantize=True, rgb=False):

        # load dataset
        f = gzip.open(os.path.join(datasets.root, 'mnist/mnist.pkl.gz'), 'rb')
        trn, val, tst = pickle.load(f, encoding='latin1')
        f.close()

        rng = np.random.RandomState(42)
        self.trn = self.Data(trn, logit, dequantize, rng, rgb)
        self.val = self.Data(val, logit, dequantize, rng, rgb)
        self.tst = self.Data(tst, logit, dequantize, rng, rgb)

        im_dim_idx = 2 if rgb else 1
        im_dim = int(np.sqrt(self.trn.x.shape[im_dim_idx]))
        self.n_dims = (3, im_dim, im_dim) if rgb else (1, im_dim, im_dim)
        self.n_labels = self.trn.y.shape[1]
        self.image_size = [im_dim, im_dim]

    def show_pixel_histograms(self, split, pixel=None):
        """
        Shows the histogram of pixel values, or of a specific pixel if given.
        """

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        if pixel is None:
            data = data_split.x.flatten()

        else:
            row, col = pixel
            idx = row * self.image_size[0] + col
            data = data_split.x[:, idx]

        n_bins = int(np.sqrt(data_split.N))
        fig, ax = plt.subplots(1, 1)
        ax.hist(data, n_bins, normed=True)
        plt.show()

    def show_images(self, split):
        """
        Displays the images in a given split.
        :param split: string
        """

        # get split
        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        # display images
        util.disp_imdata(data_split.x, self.image_size, [6, 10])

        plt.show()
