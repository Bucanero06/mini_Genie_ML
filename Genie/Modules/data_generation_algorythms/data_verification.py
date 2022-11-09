"""
Contains methods for verifying synthetic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from Modules.codependence_algorythms import get_dependence_matrix
from Modules.clustering_algorythms.hierarchical_clustering import optimal_hierarchical_cluster

import seaborn as sns


def plot_time_series_dependencies(time_series, dependence_method="gnpr_distance", **kwargs):
    """
    Plots the dependence matrix of a time series returns.

    Used to verify a time series' underlying distributions via the GNPR distance method.
    ``**kwargs`` are used to pass arguments to the `get_dependence_matrix` function used here.

    :param time_series: (pd.DataFrame) Dataframe containing time series.
    :param dependence_method: (str) Distance method to use by `get_dependence_matrix`
    :return: (plt.Axes) Figure's axes.
    """

    # Get dependence matrix
    dep_mat = get_dependence_matrix(time_series, method=dependence_method, **kwargs)

    # Plot dependence matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(dep_mat, cmap="viridis")
    ax.set_title("Dependence Matrix")
    ax.set_xlabel("Time Series")
    ax.set_ylabel("Time Series")
    ax.set_xticks(np.arange(len(time_series.columns)))
    ax.set_yticks(np.arange(len(time_series.columns)))
    ax.set_xticklabels(time_series.columns)
    ax.set_yticklabels(time_series.columns)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.show()

    return ax


def _compute_eigenvalues(mats):
    """
    Computes the eigenvalues of each matrix.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param mats: (np.array) List of matrices to calculate eigenvalues from.
        Has shape (n_sample, dim, dim)
    :return: (np.array) Resulting eigenvalues from mats.
    """

    # Get eigenvalues
    eigenvalues = np.linalg.eigvals(mats)

    # Sort eigenvalues
    eigenvalues = np.sort(eigenvalues, axis=1)

    return eigenvalues


def _compute_pf_vec(mats):
    """
    Computes the Perron-Frobenius vector of each matrix.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    The Perron-Frobenius property asserts that for a strictly positive square matrix, the
    corresponding eigenvector of the largest eigenvalue has strictly positive components.

    :param mats: (np.array) List of matrices to calculate Perron-Frobenius vector from.
        Has shape (n_sample, dim, dim)
    :return: (np.array) Resulting Perron-Frobenius vectors from mats.
    """

    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(mats)

    # Get index of largest eigenvalue
    largest_eigenvalue_idx = np.argmax(eigenvalues, axis=1)

    # Get largest eigenvector
    pf_vec = eigenvectors[np.arange(len(eigenvalues)), largest_eigenvalue_idx]

    return pf_vec


def _compute_degree_counts(mats):
    """
    Computes the number of degrees in MST of each matrix.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    The degree count is calculated by computing the MST of the matrix, and counting
    how many times each nodes appears in each edge produced by the MST. This count is normalized
    by the size of the matrix.

    :param mats: (np.array) List of matrices to calculate the number of degrees in MST from.
        Has shape (n_sample, dim, dim)
    :return: (np.array) Resulting number of degrees in MST from mats.
    """

    # Get number of nodes
    n_nodes = mats.shape[1]

    # Get number of samples
    n_samples = mats.shape[0]

    # Get degree counts
    degree_counts = np.zeros((n_samples, n_nodes))
    for i in range(n_samples):
        # Get MST
        mst = minimum_spanning_tree(csr_matrix(mats[i]))

        # Get degree counts
        degree_counts[i] = np.bincount(mst.nonzero()[0], minlength=n_nodes)

    # Normalize degree counts
    degree_counts = degree_counts / n_nodes

    return degree_counts


def plot_pairwise_dist(emp_mats, gen_mats, n_hist=100):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Distribution of pairwise correlations is significantly shifted to the positive.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
        Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
        Has shape (n_samples_b, dim_b, dim_b)
    :param n_hist: (int) Number of bins for histogram plots. (100 by default).
    :return: (plt.Axes) Figure's axes.
    """

    # Get pairwise correlations
    emp_pairwise = emp_mats[np.triu_indices(emp_mats.shape[1], k=1)]
    gen_pairwise = gen_mats[np.triu_indices(gen_mats.shape[1], k=1)]

    # Plot pairwise correlations
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(emp_pairwise, bins=n_hist, density=True, label='Empirical')
    ax[0].hist(gen_pairwise, bins=n_hist, density=True, label='Generated')
    ax[0].set_title('Pairwise Correlations')
    ax[0].set_xlabel('Correlation')
    ax[0].set_ylabel('Density')
    ax[0].legend()

    # Plot pairwise correlations (zoomed in)
    ax[1].hist(emp_pairwise, bins=n_hist, density=True, label='Empirical')
    ax[1].hist(gen_pairwise, bins=n_hist, density=True, label='Generated')
    ax[1].set_title('Pairwise Correlations (Zoomed In)')
    ax[1].set_xlabel('Correlation')
    ax[1].set_ylabel('Density')
    ax[1].legend()
    ax[1].set_xlim(-0.1, 0.1)

    return ax


def plot_eigenvalues(emp_mats, gen_mats, n_hist=100):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Eigenvalues follow the Marchenko-Pastur distribution, but for a very large first eigenvalue (the market).

    - Eigenvalues follow the Marchenko-Pastur distribution, but for a couple of other large eigenvalues (industries).

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
        Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
        Has shape (n_samples_b, dim_b, dim_b)
    :param n_hist: (int) Number of bins for histogram plots. (100 by default).
    :return: (plt.Axes) Figure's axes.
    """

    # Get eigenvalues
    emp_eigenvalues = _compute_eigenvalues(emp_mats)
    gen_eigenvalues = _compute_eigenvalues(gen_mats)

    # Plot eigenvalues
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(emp_eigenvalues, bins=n_hist, density=True, label='Empirical')
    ax[0].hist(gen_eigenvalues, bins=n_hist, density=True, label='Generated')
    ax[0].set_title('Eigenvalues')
    ax[0].set_xlabel('Eigenvalue')
    ax[0].set_ylabel('Density')
    ax[0].legend()

    # Plot eigenvalues (zoomed in)
    ax[1].hist(emp_eigenvalues, bins=n_hist, density=True, label='Empirical')
    ax[1].hist(gen_eigenvalues, bins=n_hist, density=True, label='Generated')
    ax[1].set_title('Eigenvalues (Zoomed In)')
    ax[1].set_xlabel('Eigenvalue')
    ax[1].set_ylabel('Density')
    ax[1].legend()
    ax[1].set_xlim(0, 0.1)

    return ax


def _compute_eigenvectors(mats):
    """
    Computes the eigenvectors of a set of correlation matrices.

    :param mats: (np.array) Correlation matrices. Has shape (n_samples, dim, dim).
    :return: (np.array) Eigenvectors. Has shape (n_samples, dim, dim).
    """

    # Get number of samples
    n_samples = mats.shape[0]

    # Get eigenvectors
    eigenvectors = np.zeros((n_samples, mats.shape[1], mats.shape[2]))
    for i in range(n_samples):
        eigenvectors[i] = np.linalg.eig(mats[i])[1]

    return eigenvectors


def plot_eigenvectors(emp_mats, gen_mats, n_hist=100):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Perron-Frobenius property (first eigenvector has positive entries).

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
       Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
       Has shape (n_samples_b, dim_b, dim_b)
    :param n_hist: (int) Number of bins for histogram plots. (100 by default).
    :return: (plt.Axes) Figure's axes.
    """

    # Get eigenvectors
    emp_eigenvectors = _compute_eigenvectors(emp_mats)
    gen_eigenvectors = _compute_eigenvectors(gen_mats)

    # Plot eigenvectors
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(emp_eigenvectors[:, 0], bins=n_hist, density=True, label='Empirical')
    ax[0].hist(gen_eigenvectors[:, 0], bins=n_hist, density=True, label='Generated')
    ax[0].set_title('Eigenvectors')
    ax[0].set_xlabel('Eigenvector')
    ax[0].set_ylabel('Density')
    ax[0].legend()

    # Plot eigenvectors (zoomed in)
    ax[1].hist(emp_eigenvectors[:, 0], bins=n_hist, density=True, label='Empirical')
    ax[1].hist(gen_eigenvectors[:, 0], bins=n_hist, density=True, label='Generated')
    ax[1].set_title('Eigenvectors (Zoomed In)')
    ax[1].set_xlabel('Eigenvector')
    ax[1].set_ylabel('Density')
    ax[1].legend()
    ax[1].set_xlim(0, 0.1)

    return ax


# Fntion for _compute_hierarchical_structure
def _compute_hierarchical_structure(mats):
    """
    Computes the hierarchical structure of a set of correlation matrices.

    :param mats: (np.array) Correlation matrices. Has shape (n_samples, dim, dim).
    :return: (np.array) Hierarchical structure. Has shape (n_samples, dim, dim).
    """

    # Get number of samples
    n_samples = mats.shape[0]

    # Get hierarchical structure
    hierarchical_structure = np.zeros((n_samples, mats.shape[1], mats.shape[2]))
    for i in range(n_samples):
        hierarchical_structure[i] = np.linalg.matrix_power(mats[i], 2)

    return hierarchical_structure


def plot_hierarchical_structure(emp_mats, gen_mats):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Hierarchical structure of correlations.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
       Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
       Has shape (n_samples_b, dim_b, dim_b)
    :return: (tuple) Figures' axes.
    """

    # Get hierarchical structure
    emp_hier = _compute_hierarchical_structure(emp_mats)
    gen_hier = _compute_hierarchical_structure(gen_mats)

    # Plot hierarchical structure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(emp_hier, bins=100, density=True, label='Empirical')
    ax[0].hist(gen_hier, bins=100, density=True, label='Generated')
    ax[0].set_title('Hierarchical Structure')
    ax[0].set_xlabel('Hierarchical Structure')
    ax[0].set_ylabel('Density')
    ax[0].legend()

    # Plot hierarchical structure (zoomed in)
    ax[1].hist(emp_hier, bins=100, density=True, label='Empirical')
    ax[1].hist(gen_hier, bins=100, density=True, label='Generated')
    ax[1].set_title('Hierarchical Structure (Zoomed In)')
    ax[1].set_xlabel('Hierarchical Structure')
    ax[1].set_ylabel('Density')
    ax[1].legend()
    ax[1].set_xlim(0, 0.1)

    return ax


def _compute_mst_degree_counts(mats):
    """
    Computes the degree counts of the minimum spanning tree of a set of correlation matrices.

    :param mats: (np.array) Correlation matrices. Has shape (n_samples, dim, dim).
    :return: (np.array) Degree counts. Has shape (n_samples, dim).
    """

    # Get number of samples
    n_samples = mats.shape[0]

    # Get degree counts
    degree_counts = np.zeros((n_samples, mats.shape[1]))
    for i in range(n_samples):
        mst = minimum_spanning_tree(mats[i])
        degree_counts[i] = np.bincount(mst.degree())

    return degree_counts


def _compute_mst_degree_count(mats):
    """
    Computes the degree count of the minimum spanning tree of a set of correlation matrices.

    :param mats: (np.array) Correlation matrices. Has shape (n_samples, dim, dim).
    :return: (np.array) Degree count. Has shape (n_samples,).
    """

    # Get degree counts
    degree_counts = _compute_mst_degree_counts(mats)

    # Get degree count
    degree_count = np.sum(degree_counts, axis=1)

    return degree_count


def plot_mst_degree_count(emp_mats, gen_mats):
    """
    Plots all the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Scale-free property of the corresponding Minimum Spanning Tree (MST).

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
       Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
       Has shape (n_samples_b, dim_b, dim_b)
    :return: (plt.Axes) Figure's axes.
    """

    # Get MST degree count
    emp_mst_degree_count = _compute_mst_degree_count(emp_mats)
    gen_mst_degree_count = _compute_mst_degree_count(gen_mats)

    # Plot MST degree count
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(emp_mst_degree_count, bins=100, density=True, label='Empirical')
    ax[0].hist(gen_mst_degree_count, bins=100, density=True, label='Generated')
    ax[0].set_title('MST Degree Count')
    ax[0].set_xlabel('MST Degree Count')
    ax[0].set_ylabel('Density')
    ax[0].legend()

    # Plot MST degree count (zoomed in)
    ax[1].hist(emp_mst_degree_count, bins=100, density=True, label='Empirical')
    ax[1].hist(gen_mst_degree_count, bins=100, density=True, label='Generated')
    ax[1].set_title('MST Degree Count (Zoomed In)')
    ax[1].set_xlabel('MST Degree Count')
    ax[1].set_ylabel('Density')
    ax[1].legend()
    ax[1].set_xlim(0, 0.1)

    return ax


def _compute_pairwise_correlations(mats):
    """
    Computes the pairwise correlations of a set of correlation matrices.

    :param mats: (np.array) Correlation matrices. Has shape (n_samples, dim, dim).
    :return: (np.array) Pairwise correlations. Has shape (n_samples, dim, dim).
    """

    # Get number of samples
    n_samples = mats.shape[0]

    # Get pairwise correlations
    pairwise_correlations = np.zeros((n_samples, mats.shape[1], mats.shape[1]))
    for i in range(n_samples):
        pairwise_correlations[i] = np.corrcoef(mats[i])

    return pairwise_correlations


def plot_pairwise_correlations(emp_mats, gen_mats, n_hist):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - Pairwise correlations.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
       Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
       Has shape (n_samples_b, dim_b, dim_b)
    :param n_hist: (int) Number of histograms to plot.
    :return: (tuple) Figures' axes.
    """

    # Get pairwise correlations
    emp_pairwise_corrs = _compute_pairwise_correlations(emp_mats)
    gen_pairwise_corrs = _compute_pairwise_correlations(gen_mats)

    # Plot pairwise correlations
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(emp_pairwise_corrs, bins=100, density=True, label='Empirical')
    ax[0].hist(gen_pairwise_corrs, bins=100, density=True, label='Generated')
    ax[0].set_title('Pairwise Correlations')
    ax[0].set_xlabel('Pairwise Correlations')
    ax[0].set_ylabel('Density')
    ax[0].legend()

    # Plot pairwise correlations (zoomed in)
    ax[1].hist(emp_pairwise_corrs, bins=100, density=True, label='Empirical')
    ax[1].hist(gen_pairwise_corrs, bins=100, density=True, label='Generated')
    ax[1].set_title('Pairwise Correlations (Zoomed In)')
    ax[1].set_xlabel('Pairwise Correlations')
    ax[1].set_ylabel('Density')
    ax[1].legend()
    ax[1].set_xlim(0, 0.1)

    return ax


def _compute_first_eigenvector(mats):
    """
    Computes the first eigenvector of a set of correlation matrices.

    :param mats: (np.array) Correlation matrices. Has shape (n_samples, dim, dim).
    :return: (np.array) First eigenvectors. Has shape (n_samples, dim).
    """

    # Get number of samples
    n_samples = mats.shape[0]

    # Get first eigenvector
    first_eigenvector = np.zeros((n_samples, mats.shape[1]))
    for i in range(n_samples):
        first_eigenvector[i] = np.linalg.eig(mats[i])[1][:, 0]

    return first_eigenvector


def plot_first_eigenvector(emp_mats, gen_mats):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    - First eigenvector.

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
       Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
       Has shape (n_samples_b, dim_b, dim_b)
    :return: (plt.Axes) Figure's axes.
    """

    # Get first eigenvectors
    emp_first_eigenvector = _compute_first_eigenvector(emp_mats)
    gen_first_eigenvector = _compute_first_eigenvector(gen_mats)

    # Plot first eigenvector
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(emp_first_eigenvector, bins=100, density=True, label='Empirical')
    ax[0].hist(gen_first_eigenvector, bins=100, density=True, label='Generated')
    ax[0].set_title('First Eigenvector')
    ax[0].set_xlabel('First Eigenvector')
    ax[0].set_ylabel('Density')
    ax[0].legend()

    # Plot first eigenvector (zoomed in)
    ax[1].hist(emp_first_eigenvector, bins=100, density=True, label='Empirical')
    ax[1].hist(gen_first_eigenvector, bins=100, density=True, label='Generated')
    ax[1].set_title('First Eigenvector (Zoomed In)')
    ax[1].set_xlabel('First Eigenvector')
    ax[1].set_ylabel('Density')
    ax[1].legend()
    ax[1].set_xlim(0, 0.1)

    return ax


def plot_stylized_facts(emp_mats, gen_mats, n_hist=100):
    """
    Plots the following stylized facts for comparison between empirical and generated
    correlation matrices:

    1. Distribution of pairwise correlations is significantly shifted to the positive.

    2. Eigenvalues follow the Marchenko-Pastur distribution, but for a very large first
    eigenvalue (the market).

    3. Eigenvalues follow the Marchenko-Pastur distribution, but for a couple of other
    large eigenvalues (industries).

    4. Perron-Frobenius property (first eigenvector has positive entries).

    5. Hierarchical structure of correlations.

    6. Scale-free property of the corresponding Minimum Spanning Tree (MST).

    It is reproduced with modifications from the following blog post:
    `Marti, G. (2020) TF 2.0 DCGAN for 100x100 financial correlation matrices [Online].
    Available at: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html.
    (Accessed: 17 Aug 2020)
    <https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html>`_

    :param emp_mats: (np.array) Empirical correlation matrices.
        Has shape (n_samples_a, dim_a, dim_a)
    :param gen_mats: (np.array) Generated correlation matrices.
        Has shape (n_samples_b, dim_b, dim_b)
    :param n_hist: (int) Number of bins for histogram plots. (100 by default).
    """

    # Plot pairwise correlations
    plot_pairwise_correlations(emp_mats, gen_mats, n_hist)

    # Plot eigenvalues
    plot_eigenvalues(emp_mats, gen_mats, n_hist)

    # Plot first eigenvector
    plot_first_eigenvector(emp_mats, gen_mats)

    # Plot hierarchical structure
    plot_hierarchical_structure(emp_mats, gen_mats)

    # Plot MST degree count
    plot_mst_degree_count(emp_mats, gen_mats)

    return


def plot_optimal_hierarchical_cluster(mat, method="ward"):
    """
    Calculates and plots the optimal clustering of a correlation matrix.

    It uses the `optimal_hierarchical_cluster` function in the clustering module to calculate
    the optimal hierarchy cluster matrix.

    :param mat: (np.array/pd.DataFrame) Correlation matrix.
    :param method: (str) Method to calculate the hierarchy clusters. Can take the values
        ["single", "complete", "average", "weighted", "centroid", "median", "ward"].
    :return: (plt.Axes) Figure's axes.
    """

    # Calculate optimal hierarchical cluster
    cluster = optimal_hierarchical_cluster(mat, method)

    # Plot optimal hierarchical cluster
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax = sns.heatmap(cluster, cmap="Blues", ax=ax)
    ax.set_title("Optimal Hierarchical Cluster")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Asset")

    return ax
