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
    if type(mat) == pd.DataFrame:
        mat = mat.to_numpy()
    if size is None:
        size = mat.shape
    if size[0] > mat.shape[0]:
        raise ValueError("The number of rows of the bootstrapped matrix cannot be greater than the original matrix.")
    if size[1] > mat.shape[1]:
        raise ValueError("The number of columns of the bootstrapped matrix cannot be greater than the original matrix.")

    bootstrapped_matrices = np.zeros((n_samples, size[0], size[1]))
    for i in range(n_samples):
        for j in range(size[0]):
            row = np.random.randint(0, mat.shape[0])
            if size[1] == mat.shape[1]:
                bootstrapped_matrices[i, j, :] = mat[row, :]
            else:
                col = np.random.randint(0, mat.shape[1] - size[1])
                bootstrapped_matrices[i, j, :] = mat[row, col:col + size[1]]
    return bootstrapped_matrices


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

    if type(mat) == pd.DataFrame:
        mat = mat.values
    if size is None:
        size = mat.shape[1]
    if size > mat.shape[1]:
        raise ValueError("The size of the bootstrapped matrix cannot be greater than the original matrix.")

    bootstrapped_matrices = np.zeros((n_samples, size, size))
    for i in range(n_samples):
        for j in range(size):
            for k in range(j + 1, size):
                pair = np.random.randint(0, mat.shape[1], 2)
                bootstrapped_matrices[i, j, k] = np.corrcoef(mat[:, pair[0]], mat[:, pair[1]])[0, 1]
                bootstrapped_matrices[i, k, j] = bootstrapped_matrices[i, j, k]
    return bootstrapped_matrices


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
    if type(mat) == pd.DataFrame:
        mat = mat.values
    if size is None:
        size = mat.shape
    if size[0] > mat.shape[0]:
        raise ValueError("The number of rows of the bootstrapped matrix cannot be greater than the original matrix.")
    if size[1] > mat.shape[1]:
        raise ValueError("The number of columns of the bootstrapped matrix cannot be greater than the original matrix.")
    if block_size is None:
        block_size = (size[0], size[1])
    if block_size[0] > size[0]:
        raise ValueError(
            "The number of rows of the blocks cannot be greater than the number of rows of the bootstrapped matrix.")
    if block_size[1] > size[1]:
        raise ValueError(
            "The number of columns of the blocks cannot be greater than the number of columns of the bootstrapped matrix.")

    bootstrapped_matrices = np.zeros((n_samples, size[0], size[1]))
    for i in range(n_samples):
        for j in range(size[0] // block_size[0]):
            for k in range(size[1] // block_size[1]):
                row = np.random.randint(0, mat.shape[0] - block_size[0] + 1)
                col = np.random.randint(0, mat.shape[1] - block_size[1] + 1)
                bootstrapped_matrices[i, j * block_size[0]:(j + 1) * block_size[0],
                k * block_size[1]:(k + 1) * block_size[1]] = mat[row:row + block_size[0], col:col + block_size[1]]
    return bootstrapped_matrices

if __name__ == "__main__":
    mat = np.random.rand(10, 10)
    print(mat)
    print(row_bootstrap(mat, n_samples=1, size=(5, 5)))
    print(pair_bootstrap(mat, n_samples=1, size=5))
    print(block_bootstrap(mat, n_samples=1, size=(5, 5), block_size=(2, 2)))