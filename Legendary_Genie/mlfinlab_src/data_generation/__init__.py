"""
Tools for synthetic data generation.
"""

from mlfinlab_src.data_generation.corrgan import sample_from_corrgan
from mlfinlab_src.data_generation.data_verification import (
    plot_pairwise_dist,
    plot_eigenvalues,
    plot_eigenvectors,
    plot_hierarchical_structure,
    plot_mst_degree_count,
    plot_stylized_facts,
    plot_time_series_dependencies,
    plot_optimal_hierarchical_cluster)
from mlfinlab_src.data_generation.vines import (
    sample_from_cvine,
    sample_from_dvine,
    sample_from_ext_onion)
from mlfinlab_src.data_generation.correlated_random_walks import generate_cluster_time_series
from mlfinlab_src.data_generation.hcbm import (
    time_series_from_dist,
    generate_hcmb_mat)
from mlfinlab_src.data_generation.bootstrap import (
    row_bootstrap,
    pair_bootstrap,
    block_bootstrap)
