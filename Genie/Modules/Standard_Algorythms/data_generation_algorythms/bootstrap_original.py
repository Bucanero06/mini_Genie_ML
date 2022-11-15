"""
Implementation of generating bootstrapped matrices from
"Bootstrap validation of links of a minimum spanning tree" by F. Musciotto,
L. Marotta, S. Miccichè, and R. N. Mantegna https://arxiv.org/pdf/1802.03395.pdf.
"""

import numpy as np
import pandas as pd


def row_bootstrap(mat, n_samples=1, size=None):
    """
    Uses the Row Bootstrap method to generate a new matrix of size equal or smaller than the given matrix.

    It samples with replacement a random row from the given matrix. If the required bootstrapped
    columns' size is less than the columns of the original matrix, it randomly samples contiguous
    columns of the required size. It cannot generate a matrix greater than the original.

    It is inspired by the following paper:
    `Musciotto, F., Marotta, L., Miccichè, S. and Mantegna, R.N., 2018. Bootstrap validation of
    links of a minimum spanning tree. Physica A: Statistical Mechanics and its Applications,
    512, pp.1032-1043. <https://arxiv.org/pdf/1802.03395.pdf>`_.

    :param mat: (pd.DataFrame/np.array) Matrix to sample from.
    :param n_samples: (int) Number of matrices to generate.
    :param size: (tuple) Size of the bootstrapped matrix.
    :return: (np.array) The generated bootstrapped matrices. Has shape (n_samples, size[0], size[1]).
    """

    # Check if the size of the matrix is greater than the required size.
    if mat.shape[0] < size[0] or mat.shape[1] < size[1]:
        raise ValueError("The required size is greater than the original matrix.")

    # Check if the required size is less than the columns of the original matrix.
    if mat.shape[1] > size[1]:
        # Randomly sample the required number of columns.
        cols = np.random.choice(mat.shape[1], size[1], replace=False)
        # Sort the columns.
        cols.sort()
        # Get the required columns.
        mat = mat[:, cols]

    # Initialize the bootstrapped matrices.
    bootstrapped = np.zeros((n_samples, size[0], size[1]))

    # Generate the bootstrapped matrices.
    for i in range(n_samples):
        # Randomly sample the rows.
        rows = np.random.choice(mat.shape[0], size[0], replace=True)
        # Sort the rows.
        rows.sort()
        # Get the required rows.
        bootstrapped[i] = mat[rows]

    return bootstrapped



def pair_bootstrap(mat, n_samples=1, size=None):
    """
    Uses the Pair Bootstrap method to generate a new correlation matrix of returns.

    It generates a correlation matrix based on the number of columns of the returns matrix given. It
    samples with replacement a pair of columns from the original matrix, the rows of the pairs generate
    a new row-bootstrapped matrix. The correlation value of the pair of assets is calculated and
    its value is used to fill the corresponding value in the generated correlation matrix.

    It is inspired by the following paper:
    `Musciotto, F., Marotta, L., Miccichè, S. and Mantegna, R.N., 2018. Bootstrap validation of
    links of a minimum spanning tree. Physica A: Statistical Mechanics and its Applications,
    512, pp.1032-1043. <https://arxiv.org/pdf/1802.03395.pdf>`_.

    :param mat: (pd.DataFrame/np.array) Returns matrix to sample from.
    :param n_samples: (int) Number of matrices to generate.
    :param size: (int) Size of the bootstrapped correlation matrix.
    :return: (np.array) The generated bootstrapped correlation matrices. Has shape (n_samples, mat.shape[1], mat.shape[1]).
    """

    # Initialize the bootstrapped matrices.
    bootstrapped = np.zeros((n_samples, mat.shape[1], mat.shape[1]))

    # Generate the bootstrapped matrices.
    for i in range(n_samples):
        # Randomly sample the pairs.
        pairs = np.random.choice(mat.shape[1], size=(size, 2), replace=True)
        # Sort the pairs.
        pairs.sort(axis=1)
        # Get the required pairs.
        pairs = mat[:, pairs]
        # Generate the bootstrapped matrix.
        bootstrapped[i] = np.corrcoef(row_bootstrap(pairs, size=size))

    return bootstrapped


def block_bootstrap(mat, n_samples=1, size=None, block_size=None):
    """
    Uses the Block Bootstrap method to generate a new matrix of size equal to or smaller than the given matrix.

    It divides the original matrix into blocks of the given size. It samples with replacement random
    blocks to populate the bootstrapped matrix. It cannot generate a matrix greater than the original.

    It is inspired by the following paper:
    `Künsch, H.R., 1989. The jackknife and the bootstrap for general stationary observations.
    Annals of Statistics, 17(3), pp.1217-1241. <https://projecteuclid.org/euclid.aos/1176347265>`_.

    :param mat: (pd.DataFrame/np.array) Matrix to sample from.
    :param n_samples: (int) Number of matrices to generate.
    :param size: (tuple) Size of the bootstrapped matrix.
    :param block_size: (tuple) Size of the blocks.
    :return: (np.array) The generated bootstrapped matrices. Has shape (n_samples, size[0], size[1]).
    """

    # Check if the size of the matrix is greater than the required size.
    if mat.shape[0] < size[0] or mat.shape[1] < size[1]:
        raise ValueError("The required size is greater than the original matrix.")

    # Check if the required size is less than the columns of the original matrix.
    if mat.shape[1] > size[1]:
        # Randomly sample the required number of columns.
        cols = np.random.choice(mat.shape[1], size[1], replace=False)
        # Sort the columns.
        cols.sort()
        # Get the required columns.
        mat = mat[:, cols]

    # Initialize the bootstrapped matrices.
    bootstrapped = np.zeros((n_samples, size[0], size[1]))

    # Generate the bootstrapped matrices.
    for i in range(n_samples):
        # Randomly sample the blocks.
        blocks = np.random.choice(mat.shape[0], size=(size[0]//block_size[0], size[1]//block_size[1]), replace=True)
        # Sort the blocks.
        blocks.sort()
        # Get the required blocks.
        blocks = mat[blocks]
        # Generate the bootstrapped matrix.
        bootstrapped[i] = blocks.reshape(size)

    return bootstrapped

