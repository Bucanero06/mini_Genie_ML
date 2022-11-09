"""
Labeling techniques used in financial machine learning.
"""

from Modules.labeling_algorythms.labeling import (add_vertical_barrier, apply_pt_sl_on_t1, barrier_touched, drop_labels,
                                                  get_bins, get_events)
from Modules.labeling_algorythms.trend_scanning import trend_scanning_labels
from Modules.labeling_algorythms.tail_sets import TailSetLabels
from Modules.labeling_algorythms.fixed_time_horizon import fixed_time_horizon
from Modules.labeling_algorythms.matrix_flags import MatrixFlagLabels
from Modules.labeling_algorythms.excess_over_median import excess_over_median
from Modules.labeling_algorythms.raw_return import raw_return
from Modules.labeling_algorythms.return_vs_benchmark import return_over_benchmark
from Modules.labeling_algorythms.excess_over_mean import excess_over_mean
