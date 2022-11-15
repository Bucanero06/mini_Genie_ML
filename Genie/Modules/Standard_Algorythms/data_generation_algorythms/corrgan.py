# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/genieml_src/blob/master/LICENSE.txt
"""
Implementation of sampling realistic financial correlation matrices from
"CorrGAN: Sampling Realistic Financial Correlation Matrices using
Generative Adversarial Networks" by Gautier Marti.
https://arxiv.org/pdf/1910.09504.pdf
"""
from os import listdir, path
import numpy as np
from scipy.cluster import hierarchy
from statsmodels.stats.correlation_tools import corr_nearest


def sample_from_corrgan(model_loc, dim=10, n_samples=1):
    """
    Samples correlation matrices from the pre-trained CorrGAN network.

    It is reproduced with modifications from the following paper:
    `Marti, G., 2020, May. CorrGAN: Sampling Realistic Financial Correlation Matrices Using
    Generative Adversarial Networks. In ICASSP 2020-2020 IEEE International Conference on
    Acoustics, Speech and Signal Processing (ICASSP) (pp. 8459-8463). IEEE.
    <https://arxiv.org/pdf/1910.09504.pdf>`_

    It loads the appropriate CorrGAN model for the required dimension. Generates a matrix output
    from this network. Symmetries this matrix and finds the nearest correlation matrix
    that is positive semi-definite. Finally, it maximizes the sum of the similarities between
    adjacent leaves to arrange it with hierarchical clustering.

    The CorrGAN network was trained on the correlation profiles of the S&P 500 stocks. Therefore
    the output retains these properties. In addition, the final output retains the following
    6 stylized facts:

    1. Distribution of pairwise correlations is significantly shifted to the positive.

    2. Eigenvalues follow the Marchenko-Pastur distribution, but for a very large first
    eigenvalue (the market).

    3. Eigenvalues follow the Marchenko-Pastur distribution, but for a couple of other
    large eigenvalues (industries).

    4. Perron-Frobenius property (first eigenvector has positive entries).

    5. Hierarchical structure of correlations.

    6. Scale-free property of the corresponding Minimum Spanning Tree (MST).

    :param model_loc: (str) Location of folder containing CorrGAN models.
    :param dim: (int) Dimension of correlation matrix to sample.
        In the range [2, 200].
    :param n_samples: (int) Number of samples to generate.
    :return: (np.array) Sampled correlation matrices of shape (n_samples, dim, dim).
    """

    # Check if the dimension is in the range [2, 200].
    if dim < 2 or dim > 200:
        raise ValueError('Dimension must be in the range [2, 200].')

    # Check if the number of samples is positive.
    if n_samples < 1:
        raise ValueError('Number of samples must be positive.')

    # Check if the model location exists.
    if not path.exists(model_loc):
        raise ValueError('Model location does not exist.')

    # Check if the model location contains the required model.
    if not path.exists(path.join(model_loc, f'corrgan_{dim}.h5')):
        raise ValueError('Model location does not contain the required model.')

    # Load the model.
    from keras.models import load_model
    model = load_model(path.join(model_loc, f'corrgan_{dim}.h5'))

    # Generate the samples.
    samples = model.predict(np.random.normal(size=(n_samples, 100)))

    # Symmetrize the samples.
    samples = (samples + samples.transpose(0, 2, 1)) / 2

    # Find the nearest correlation matrix.
    samples = np.array([corr_nearest(sample) for sample in samples])

    # Hierarchical clustering.
    for i in range(n_samples):
        samples[i] = hierarchy.dendrogram(
            hierarchy.linkage(samples[i], method='complete'),
            no_plot=True
        )['leaves']
    samples = samples[:, :, np.newaxis]
    samples = samples.transpose(0, 2, 1)
    samples = samples[:, :, np.arange(dim)]

    return samples

