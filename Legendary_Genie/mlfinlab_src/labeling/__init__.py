"""
Labeling techniques used in financial machine learning.
"""

from mlfinlab_src.labeling.labeling import (add_vertical_barrier, apply_pt_sl_on_t1, barrier_touched, drop_labels,
                                        get_bins, get_events)
from mlfinlab_src.labeling.trend_scanning import trend_scanning_labels
from mlfinlab_src.labeling.tail_sets import TailSetLabels
from mlfinlab_src.labeling.fixed_time_horizon import fixed_time_horizon
from mlfinlab_src.labeling.matrix_flags import MatrixFlagLabels
from mlfinlab_src.labeling.excess_over_median import excess_over_median
from mlfinlab_src.labeling.raw_return import raw_return
from mlfinlab_src.labeling.return_vs_benchmark import return_over_benchmark
from mlfinlab_src.labeling.excess_over_mean import excess_over_mean
