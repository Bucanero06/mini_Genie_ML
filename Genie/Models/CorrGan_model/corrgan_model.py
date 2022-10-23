# Import packages
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy

# Import MlFinLab tools
os.environ['MLFINLAB_API_KEY'] = "0800b4ea410a702acddefdec86f93523"
from mlfinlab.data_generation.corrgan import sample_from_corrgan  # noqa: F401
from mlfinlab.data_generation.data_verification import (plot_pairwise_dist, plot_eigenvalues,  # noqa: F401
                                                        plot_eigenvectors, plot_hierarchical_structure,  # noqa: F401
                                                        plot_mst_degree_count, plot_stylized_facts)  # noqa: F401

from mlfinlab.data_generation.hcbm import generate_hcmb_mat, time_series_from_dist  # noqa: F401
from mlfinlab.data_generation.data_verification import plot_optimal_hierarchical_cluster  # noqa: F401


def daily_rolling_correlation__(prices, window=252):
    prices = prices.pct_change()
    result=prices.rolling(window, min_periods=window // 2).corr()
    return result


from Modules._Data_Manager import Data_Manager

warnings.filterwarnings('ignore')

# Setting seeds
# random.seed(2814)
# np.random.seed(2814)
# tf.random.set_seed(2814)

# Tickes to use
SP_ASSETS = ["AAPL",
             "MSFT",
             "AMZN",
             "META",
             "GOOG", "GOOGL", "JNJ", "PG", "V", "JPM",
             "UNH", "HD", "MA", "NVDA", "VZ", "DIS", "PYPL", "ADBE", "T", "NFLX",
             "PFE", "MRK", "INTC", "BAC", "CMCSA", "PEP", "WMT", "KO", "XOM", "CSCO",
             # "^GSPC"
             ]

# Adapted from: https://marti.ai/ml/2019/10/13/tf-dcgan-financial-correlation-matrices.html
dimensions = len(SP_ASSETS)
print(f"Number of assets: {dimensions}")
# Download stock returns and compute correlations
# asset_prices = yf.download(tickers=" ".join(SP_ASSETS), start="2017-08-01", end="2018-08-01")['Close']


from datetime import datetime, timedelta
import vectorbtpro as vbt

DAYS_TO_DOWNLOAD = 365 * 2
end_time = datetime.now()
start_time = end_time - timedelta(days=DAYS_TO_DOWNLOAD)
data = vbt.YFData.fetch(
    SP_ASSETS,
    start=start_time,
    end=end_time,
    timeframe="1d",
    missing_index='drop'  # Creates problems with missing the index
)
prices = data.get("Close")
# asset_prices.pct_change().to_csv('sp30_rolling_corr.csv')

# Upper triangle indices (without diagonal) of correlation matrix (n, n) -> (n*(n-1)/2, 2)
tri_rows, tri_cols = np.triu_indices(dimensions, k=1)

# Plot a few correlation matrices and get empirical eigenvalues
plt.figure(figsize=(12, 8))
plt.suptitle("Empirical Correlation Matrices of dimension = {}".format(dimensions))
rolling_corr = daily_rolling_correlation__(prices, window=252).dropna()
iter_keys = rolling_corr.groupby(level=0)
empirical_mats = []
#
for i, (date, corr_mat) in enumerate(iter_keys):
    corr_mat = corr_mat.values
    empirical_mats.append(corr_mat)
    #
    # Arrange with hierarchical clustering by maximizing the sum of the similarities between adjacent leaves
    dist = 1 - corr_mat
    linkage_mat = hierarchy.linkage(dist[tri_rows, tri_cols], method="ward")
    optimal_leaves = hierarchy.optimal_leaf_ordering(linkage_mat, dist[tri_rows, tri_cols])
    optimal_ordering = hierarchy.leaves_list(optimal_leaves)
    ordered_corr = corr_mat[optimal_ordering, :][:, optimal_ordering]
    #
    # Plot it
    plt.subplot(*(int(np.ceil(np.sqrt(len(iter_keys)))), int(np.ceil(np.sqrt(len(iter_keys))))), i + 1)
    plt.pcolormesh(ordered_corr, cmap='viridis')
    plt.title(f'{date}', fontsize=0.1)
#
empirical_mats = np.array(empirical_mats)
plt.colorbar()
plt.show()

# Sample from CorrGAN.
corrgan_mats = sample_from_corrgan(model_loc="/home/ruben/PycharmProjects/mini_Genie_ML/corrgan_models",
                                   dim=dimensions,
                                   n_samples=len(rolling_corr.index.get_level_values(0).unique())
                                   # n_samples=2
                                   )

data_manager = Data_Manager()
# corrmat_to_ts(ordered_corr, "Ordered", 1000)
# corrmat_to_ts(empirical_mats, "Empirical", 1000)
data_manager.corrmat_to_ts(corrgan_mats, "CorrGAN", 1000,
                           starting_prices=None,
                           plot=True)

# Plot a few samples
plt.figure(figsize=(12, 8))
plt.suptitle("Generated Correlation Matrices of dimension = {}".format(dimensions))

for i in range(len(corrgan_mats)):
    plt.subplot(*(int(np.ceil(np.sqrt(len(corrgan_mats)))), int(np.ceil(np.sqrt(len(corrgan_mats))))), i + 1)
    plt.pcolormesh(corrgan_mats[i], cmap='viridis')
    plt.colorbar()
plt.show()

plot_stylized_facts(empirical_mats, corrgan_mats)

# plot_pairwise_dist(empirical_mats, corrgan_mats)
# plot_eigenvalues(empirical_mats, corrgan_mats)
# plot_eigenvectors(empirical_mats, corrgan_mats)
# plot_hierarchical_structure(empirical_mats, corrgan_mats)
# plot_mst_degree_count(empirical_mats, corrgan_mats)
