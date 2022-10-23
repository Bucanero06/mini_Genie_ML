import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy.cluster import hierarchy

# Download stock returns and compute correlations
prices = yf.download(tickers=" ".join(SP_ASSETS), start="2017-08-01", end="2018-08-01")['Close']
rolling_corr = daily_rolling_correlation__(prices, window=252).dropna()

# Upper triangle indices (without diagonal) of correlation matrix (n, n) -> (n*(n-1)/2, 2)
tri_rows, tri_cols = np.triu_indices(dimensions, k=1)
# Plot a few correlation matrices
plt.figure(figsize=(12, 8))
plt.suptitle("Empirical Correlation Matrices of dimension = {}".format(dimensions))
# for i, date in enumerate(random.sample(rolling_corr.groupby(level=0).indices.keys(), 4)):

iter_keys = list(rolling_corr.groupby(level=0).indices.keys())[:4]

for i, date in enumerate(iter_keys):
    corr_mat = rolling_corr.loc[date].values

    # Arrange with hierarchical clustering by maximizing the sum of the
    # similarities between adjacent leaves
    dist = 1 - corr_mat
    linkage_mat = hierarchy.linkage(dist[tri_rows, tri_cols], method="ward")
    optimal_leaves = hierarchy.optimal_leaf_ordering(linkage_mat, dist[tri_rows, tri_cols])
    optimal_ordering = hierarchy.leaves_list(optimal_leaves)
    ordered_corr = corr_mat[optimal_ordering, :][:, optimal_ordering]

    # Plot it
    # plt.subplot(2, 2, i + 1)
    plt.subplot(*(int(np.ceil(np.sqrt(len(iter_keys)))), int(np.ceil(np.sqrt(len(iter_keys))))), i + 1)
    plt.pcolormesh(ordered_corr, cmap='viridis')
    plt.colorbar()
    # plt.title(date)
    plt.title(f'ordered_corr {date}')

plt.show()

# Convert pandas dataframe to numpy array
empirical_mats = []
for date, corr_mat in rolling_corr.groupby(level=0):
    empirical_mats.append(corr_mat.values)
empirical_mats = np.array(empirical_mats)
