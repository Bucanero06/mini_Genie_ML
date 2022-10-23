"""
Implementation of hierarchical clustering algorithms. FIXME Missing!
"""
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy


def optimal_hierarchical_cluster(mat: np.array, method: str = "ward") -> np.array:
    """
    Calculates the optimal clustering of a matrix.

    It calculates the hierarchy clusters from the distance of the matrix. Then it calculates
    the optimal leaf ordering of the hierarchy clusters, and returns the optimally clustered matrix.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    This method relies and acts as a wrapper for the `scipy.cluster.hierarchy` module.
    `<https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html>`_

    :param mat: (np.array/pd.DataFrame) Correlation matrix.
    :param method: (str) Method to calculate the hierarchy clusters. Can take the values
        ["single", "complete", "average", "weighted", "centroid", "median", "ward"].
    :return: (np.array) Optimal hierarchy cluster matrix.
    """

    # Convert matrix to numpy array if it is a pandas dataframe
    if isinstance(mat, pd.DataFrame):
        mat = mat.values

    matrix = np.array(mat)
    labels = np.array(mat.columns)
    d = hierarchy.distance.pdist(matrix)  # vector of upper triangle of distance matrix
    z = hierarchy.linkage(d, method=method)  # linkage matrix
    optimal_leaf_ordering = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(z, d))
    optimal_hierarchy_cluster = pd.DataFrame(np.array(matrix[optimal_leaf_ordering, :])[:, optimal_leaf_ordering],
                                             index=labels[optimal_leaf_ordering],
                                             columns=labels[optimal_leaf_ordering])

    return optimal_hierarchy_cluster
