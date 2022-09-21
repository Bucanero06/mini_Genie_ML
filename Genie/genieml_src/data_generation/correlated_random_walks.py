"""
Contains methods for generating correlated random walks.
"""

import numpy as np
import pandas as pd


def generate_cluster_time_series(n_series, t_samples=100, k_corr_clusters=1,
                                 d_dist_clusters=1, rho_main=0.1, rho_corr=0.3, price_start=100.0,
                                 dists_clusters=("normal", "normal", "student-t", "normal", "student-t")):
    """
    Generates a synthetic time series of correlation and distribution clusters.
    It is reproduced with modifications from the following paper:
    `Donnat, P., Marti, G. and Very, P., 2016. Toward a generic representation of random
    variables for machine learning. Pattern Recognition Letters, 70, pp.24-31.
    <https://www.sciencedirect.com/science/article/pii/S0167865515003906>`_
    `www.datagrapple.com. (n.d.). DataGrapple - Tech: A GNPR tutorial: How to cluster random walks.
    [online] Available at:  [Accessed 26 Aug. 2020].
    <https://www.datagrapple.com/Tech/GNPR-tutorial-How-to-cluster-random-walks.html>`_
    This method creates `n_series` time series of length `t_samples`. Each time series is divided
    into `k_corr_clusters` correlation clusters. Each correlation cluster is subdivided into
    `d_dist_clusters` distribution clusters.
    A main distribution is sampled from a normal distribution with mean = 0 and stdev = 1, adjusted
    by a `rho_main` factor. The correlation clusters are sampled from a given distribution, are generated
    once, and adjusted by a `rho_corr` factor. The distribution clusters are sampled from other
    given distributions, and adjusted by (1 - `rho_main` - `rho_corr`). They are sampled for each time series.
    These three series are added together to form a time series of returns. The final time series
    is the cumulative sum of the returns, with a start price given by `price_start`.
    :param n_series: (int) Number of time series to generate.
    :param t_samples: (int) Number of samples in each time series.
    :param k_corr_clusters: (int) Number of correlation clusters in each time series.
    :param d_dist_clusters: (int) Number of distribution clusters in each time series.
    :param rho_main: (float): Strength of main time series distribution.
    :param rho_corr: (float): Strength of correlation cluster distribution.
    :param price_start: (float) Starting price of the time series.
    :param dists_clusters: (list) List containing the names of the distributions to sample from.
        The following numpy distributions are available: "normal" = normal(0, 1), "normal_2" = normal(0, 2),
        "student-t" = standard_t(3)/sqrt(3), "laplace" = laplace(1/sqrt(2)). The first disitribution
        is used to sample for the correlation clusters (k_corr_clusters), the remaining ones are used
        to sample for the distribution clusters (d_dist_clusters).
    :return: (pd.DataFrame) Generated time series. Has size (t_samples, n_series).
    """
    # Check input
    if not isinstance(n_series, int):
        raise TypeError("n_series must be an integer.")
    if not isinstance(t_samples, int):
        raise TypeError("t_samples must be an integer.")
    if not isinstance(k_corr_clusters, int):
        raise TypeError("k_corr_clusters must be an integer.")
    if not isinstance(d_dist_clusters, int):
        raise TypeError("d_dist_clusters must be an integer.")
    if not isinstance(rho_main, float):
        raise TypeError("rho_main must be a float.")
    if not isinstance(rho_corr, float):
        raise TypeError("rho_corr must be a float.")
    if not isinstance(price_start, float):
        raise TypeError("price_start must be a float.")
    if not isinstance(dists_clusters, (list, tuple)):
        raise TypeError("dists_clusters must be a list or tuple.")
    if len(dists_clusters) != k_corr_clusters + d_dist_clusters:
        raise ValueError("dists_clusters must have length k_corr_clusters + d_dist_clusters.")
    if not all(isinstance(dist, str) for dist in dists_clusters):
        raise TypeError("dists_clusters must contain only strings.")
    if not all(dist in ("normal", "normal_2", "student-t", "laplace") for dist in dists_clusters):
        raise ValueError("dists_clusters must contain only the following strings: "
                         "'normal', 'normal_2', 'student-t', 'laplace'.")
    if rho_main + rho_corr >= 1.0:
        raise ValueError("rho_main + rho_corr must be less than 1.0.")

    # Generate correlation clusters
    corr_clusters = np.zeros((k_corr_clusters, t_samples))
    for i in range(k_corr_clusters):
        if dists_clusters[i] == "normal":
            corr_clusters[i, :] = np.random.normal(0, 1, t_samples)
        elif dists_clusters[i] == "normal_2":
            corr_clusters[i, :] = np.random.normal(0, 2, t_samples)
        elif dists_clusters[i] == "student-t":
            corr_clusters[i, :] = np.random.standard_t(3) / np.sqrt(3)
        elif dists_clusters[i] == "laplace":
            corr_clusters[i, :] = np.random.laplace(0, 1 / np.sqrt(2))
        else:
            raise ValueError("dists_clusters must contain only the following strings: "
                             "'normal', 'normal_2', 'student-t', 'laplace'.")
    corr_clusters = corr_clusters * rho_corr

    # Generate distribution clusters
    dist_clusters = np.zeros((d_dist_clusters, t_samples))
    for i in range(d_dist_clusters):
        if dists_clusters[k_corr_clusters + i] == "normal":
            dist_clusters[i, :] = np.random.normal(0, 1, t_samples)
        elif dists_clusters[k_corr_clusters + i] == "normal_2":
            dist_clusters[i, :] = np.random.normal(0, 2, t_samples)
        elif dists_clusters[k_corr_clusters + i] == "student-t":
            dist_clusters[i, :] = np.random.standard_t(3) / np.sqrt(3)
        elif dists_clusters[k_corr_clusters + i] == "laplace":
            dist_clusters[i, :] = np.random.laplace(0, 1 / np.sqrt(2))
        else:
            raise ValueError("dists_clusters must contain only the following strings: "
                             "'normal', 'normal_2', 'student-t', 'laplace'.")
    dist_clusters = dist_clusters * (1 - rho_main - rho_corr)

    # Generate main time series
    main_series = np.random.normal(0, 1, t_samples) * rho_main

    # Generate time series
    time_series = np.zeros((n_series, t_samples))
    for i in range(n_series):
        time_series[i, :] = main_series + corr_clusters[i % k_corr_clusters, :] + \
                            dist_clusters[i % d_dist_clusters, :]

    # Generate price series
    price_series = np.cumsum(time_series, axis=1) + price_start

    return pd.DataFrame(price_series)


def generate_correlated_random_walks(n_series, t_samples=100, rho=0.5, price_start=100.0):
    """
    Generates a synthetic time series of correlated random walks.
    It is reproduced with modifications from the following paper:
    `Donnat, P., Marti, G. and Very, P., 2016. Toward a generic representation of random
    variables for machine learning. Pattern Recognition Letters, 70, pp.24-31.
    <https://www.sciencedirect.com/science/article/pii/S0167865515003906>`_
    `www.datagrapple.com. (n.d.). DataGrapple - Tech: A GNPR tutorial: How to cluster random walks.
    [online] Available at:  [Accessed 26 Aug. 2020].
    <https://www.datagrapple.com/Tech/GNPR-tutorial-How-to-cluster-random-walks.html>`_
    This method creates `n_series` time series of length `t_samples`. Each time series is divided
    into `k_corr_clusters` correlation clusters. Each correlation cluster is subdivided into
    `d_dist_clusters` distribution clusters.
    A main distribution is sampled from a normal distribution with mean = 0 and stdev = 1, adjusted
    by a `rho_main` factor. The correlation clusters are sampled from a given distribution, are generated
    once, and adjusted by a `rho_corr` factor. The distribution clusters are sampled from other
    given distributions, and adjusted by (1 - `rho_main` - `rho_corr`). They are sampled for each time series.
    These three series are added together to form a time series of returns. The final time series
    is the cumulative sum of the returns, with a start price given by `price_start`.
    :param n_series: (int) Number of time series to generate.
    :param t_samples: (int) Number of samples in each time series.
    :param rho: (float): Strength of correlation between time series.
    :param price_start: (float) Starting price of the time series.
    :return: (pd.DataFrame) Generated time series. Has size (t_samples, n_series).
    """
    # Check input
    if not isinstance(n_series, int):
        raise TypeError("n_series must be an integer.")
    if not isinstance(t_samples, int):
        raise TypeError("t_samples must be an integer.")
    if not isinstance(rho, float):
        raise TypeError("rho must be a float.")
    if not isinstance(price_start, float):
        raise TypeError("price_start must be a float.")

    # Generate correlation matrix
    corr_matrix = np.zeros((n_series, n_series))
    for i in range(n_series):
        for j in range(n_series):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr_matrix[i, j] = rho

    # Generate time series
    time_series = np.random.multivariate_normal(np.zeros(n_series), corr_matrix, t_samples)

    # Generate price series
    price_series = np.cumsum(time_series, axis=0) + price_start

    return pd.DataFrame(price_series)

if __name__ == "__main__":
    # Generate correlated random walks
    n_series = 10
    t_samples = 100
    rho = 0.5
    price_start = 100.0
    price_series = generate_correlated_random_walks(n_series, t_samples, rho, price_start)
    print(price_series)

    # Generate correlation and distribution clusters
    n_series = 10
    t_samples = 100
    k_corr_clusters = 2
    d_dist_clusters = 3
    rho_main = 0.1
    rho_corr = 0.3
    price_start = 100.0
    dists_clusters = ("normal", "normal", "student-t", "normal", "student-t")
    price_series = generate_cluster_time_series(n_series, t_samples, k_corr_clusters,
                                                d_dist_clusters, rho_main, rho_corr, price_start,
                                                dists_clusters)
    print(price_series)