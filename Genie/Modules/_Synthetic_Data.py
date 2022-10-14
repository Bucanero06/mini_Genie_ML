import os

import pandas as pd

from Modules.Utils import dict_to_namedtuple

os.environ['MLFINLAB_API_KEY'] = "0800b4ea410a702acddefdec86f93523"
from mlfinlab.data_generation.hcbm import generate_hcmb_mat, time_series_from_dist
from mlfinlab.data_generation.data_verification import plot_optimal_hierarchical_cluster
from mlfinlab.data_generation.correlated_random_walks import generate_cluster_time_series
from mlfinlab.data_generation.data_verification import plot_time_series_dependencies
from mlfinlab.data_generation.corrgan import sample_from_corrgan  # noqa: F401

# Import packages
import numpy as np
import matplotlib.pyplot as plt


def HCBM_Synthetic_TS_Data(
        # Initialize parameters
        n_assets=1,  # Number of assets to generate (default=1)
        n_paths=200,  # Number of paths to generate (default=200)
        rho_low=0.1,  # minimum correlation between assets (default=0.1)
        rho_high=0.9,  # maximum correlation between assets (default=0.9)
        n_bars=1000,  # number of bars in each time series (default=1000)
        distribution="normal",  # distribution of returns (default="normal") "normal", "student_t"
        method="ward",  # method for hierarchical clustering (default="ward")
        # “single”, “complete”, “average”, “weighted”, “centroid”, “median”, Default: “ward”
        permute=False,  # Whether to permute the final HCBM matrix (default=False)
        plot=False,  # Whether to plot the HCBM matrices and TS distributions (default=False)
        # **kwargs
):
    """
    Generate synthetic time series data from HCBM matrix. This function is a wrapper for the mlfinlab functions

    """
    subplot_shape = (int(np.ceil(np.sqrt(n_assets))), int(np.ceil(np.sqrt(n_assets))))

    # hcbm_dict__= dict_to_namedtuple(dict())
    hcbm_dict__ = dict(
        n_assets=1,
        n_paths=200,
        rho_low=0.1,
        rho_high=0.9,
        n_bars=1000,
        dist=distribution,
        method=method,
        permute=permute
    )

    # Generate time series from HCBM matrix
    hcbm_mats = generate_hcmb_mat(t_samples=n_assets,
                                  n_size=n_paths,
                                  rho_low=rho_low,
                                  rho_high=rho_high,
                                  permute=permute)
    hcbm_dict__["hcbm_mats"] = hcbm_mats

    # TODO Carefully check from here on please :)
    if plot:
        # Plot HCBM matrices
        for i in range(len(hcbm_mats)):
            plt.subplot(*subplot_shape, i + 1)
            plt.pcolormesh(hcbm_mats[i], cmap='viridis')
            plt.title(f"Permuted HCBM Matrix {i + 1}") if permute else plt.title(f"HCBM Matrix {i + 1}")
        plt.colorbar()
        plt.show()
    hcbm_dict__[f"df"] = pd.DataFrame()
    for i in range(len(hcbm_mats)):
        # Generate time series
        series_df = time_series_from_dist(hcbm_mats[i], dist=distribution, t_samples=n_bars)
        series_df = series_df.cumsum()

        if isinstance(series_df, pd.DataFrame):
            series_df.columns = [f"asset_{i}_{col}" for col in series_df.columns]
        else:
            series_df.name = f"asset_{i}"
        hcbm_dict__[f"df"] = pd.concat([hcbm_dict__[f"df"], series_df], axis=1)

        if plot:
            # Plot time series
            series_df.plot(legend=None, title=f"Time Series from Permuted HCBM Matrix {i + 1}")
            plt.show()
            #
            # Plot recovered HCBM matrix
            plot_optimal_hierarchical_cluster(hcbm_dict__[f"df"][f"ts_{i}"].corr(), method=method)
            plt.title(f"Recovered HCBM Matrix {i + 1}")
            plt.show()

    return dict_to_namedtuple(hcbm_dict__)

def SP500_CorrGAN(dimensions=None, n_samples=None):
    """
    Generate synthetic time series data from HCBM matrix. This function is a wrapper for the mlfinlab functions

    """

    return sample_from_corrgan(model_loc="/home/ruben/PycharmProjects/mini_Genie_ML/corrgan_models",
                                       dim=dimensions,
                                       n_samples=n_samples
                                       # n_samples=2
                                       )

def Synthetic_Clustered_TS_Generation(n_series=200,
                                      t_samples=5000,
                                      k_clusters=5,
                                      d_clusters=2,
                                      rho_corrs=0.3,
                                      dists_clusters=["normal", "normal", "student-t", "normal", "student-t"]):
    # Initialize the example parameters for each example time series

    def plot_time_series(dataset, title, theta=0.5):
        """
        Plotting time series.
        """

        dataset.plot(legend=None, title="Time Series for {} Example".format(title))

        plot_time_series_dependencies(dataset, dependence_method='gpr_distance', theta=theta)
        plot_time_series_dependencies(dataset, dependence_method='gnpr_distance', theta=theta)

    # Plot the time series and codependence matrix for each example
    dataset = generate_cluster_time_series(n_series=n_series, t_samples=t_samples, k_corr_clusters=k_clusters,
                                           d_dist_clusters=d_clusters, rho_corr=rho_corrs,
                                           dists_clusters=dists_clusters)
    plot_time_series(dataset, title="Distribution", theta=0)

    plt.show()

    exit()


Synthetic_Clustered_TS_Generation(n_series=200,
                                  t_samples=5000,
                                  k_clusters=5,
                                  d_clusters=2,
                                  rho_corrs=0.3,
                                  dists_clusters=["normal", "normal", "student-t", "normal", "student-t"])
