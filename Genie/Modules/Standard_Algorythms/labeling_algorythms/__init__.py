"""
Labeling techniques used in financial machine learning.
"""

from Modules.Standard_Algorythms.labeling_algorythms.labeling import (add_vertical_barrier, apply_pt_sl_on_t1, barrier_touched, drop_labels,
                                                  get_bins, get_events)
from Modules.Standard_Algorythms.labeling_algorythms.trend_scanning import trend_scanning_labels
from Modules.Standard_Algorythms.labeling_algorythms.tail_sets import TailSetLabels
from Modules.Standard_Algorythms.labeling_algorythms.fixed_time_horizon import fixed_time_horizon
from Modules.Standard_Algorythms.labeling_algorythms.matrix_flags import MatrixFlagLabels
from Modules.Standard_Algorythms.labeling_algorythms.excess_over_median import excess_over_median
from Modules.Standard_Algorythms.labeling_algorythms.raw_return import raw_return
from Modules.Standard_Algorythms.labeling_algorythms.return_vs_benchmark import return_over_benchmark
from Modules.Standard_Algorythms.labeling_algorythms.excess_over_mean import excess_over_mean
