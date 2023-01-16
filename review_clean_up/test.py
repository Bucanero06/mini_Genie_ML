
exit()
from multiprocessing import cpu_count

import pandas as pd

# import mlfinlab as mlf
from Modules._Data_Manager import Data_Manager
from _old_pipeline_scripts_examples.fml.Utility_Functions import get_daily_volatility

# FILE_NAME = "USA100_bid_ask_price_sample.csv"
file_name = "Backtest_Data.csv"
file_id = file_name.split(".")[0]


import os
os.environ['MLFINLAB_API_KEY'] = "0800b4ea410a702acddefdec86f93523"
import mlfinlab as ml
from mlfinlab.data_generation.hcbm import generate_hcmb_mat, time_series_from_dist
from mlfinlab.data_generation.data_verification import plot_optimal_hierarchical_cluster

import matplotlib.pyplot as plt
import numpy as np


from Genie.Modules._Synthetic_Data import HCBM_Synthetic_TS_Data

df = HCBM_Synthetic_TS_Data(
    # Initialize parameters
    samples=1,
    dim=200,
    rho_low=0.1,
    rho_high=0.9,
    t_samples=1000,
    starting_price=100,
    dist="student_t",
    method="ward",
    plot=False,
    # **kwargs
)

print(df)





exit()

# Read in data
data = Data_Manager().fetch_csv_data_dask(data_file_name=file_name,
                                          search_in=[".", "Datas", "Datas/Sample-Data", "Datas/USDJPY_Tick_Data"])

exit()
# Compute daily volatility
daily_vol = ml.util.get_daily_vol(close=data['ask'], lookback=50)

# Apply Symmetric CUSUM Filter and get timestamps for events
# Note: Only the CUSUM filter needs a point estimate for volatility
cusum_events = ml.filters.cusum_filter(data['close'],
                                       threshold=daily_vol['2011-09-01':'2018-01-01'].mean())

# Compute vertical barrier using timedelta
vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events,
                                                     close=data['close'],
                                                     num_days=1)

# Another option is to compute the vertical bars after a fixed number of samples
vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events,
                                                     close=data['close'],
                                                     num_bars=20)

exit()
ask = tick_bars['ask'].copy()
bid = tick_bars['bid'].copy()
closing = (ask + bid) / 2

volatility = get_daily_volatility(closing)
times = mlf.filters.cusum_filter(closing, volatility.mean() * .1)

vertical_barriers = mlf.labeling.add_vertical_barrier(times, closing, num_days=1)
pt_sl = [1, 1]
min_ret = 0.004

threads = cpu_count() - 1

triple_barrier_events = mlf.labeling.get_events(closing,
                                                times,
                                                pt_sl,
                                                volatility,
                                                min_ret,
                                                threads,
                                                vertical_barriers)

labels_one = mlf.labeling.get_bins(triple_barrier_events, closing)

t1 = triple_barrier_events['t1'].copy()

full_df = pd.DataFrame(tick_bars.loc[labels_one['bin'].index], index=labels_one['bin'].index)
full_df.drop(columns=['bid', 'ask'], inplace=True)
full_df['labels'] = labels_one['bin'].copy()
print(full_df.head())
full_df.to_csv(f'Datas/{file_id}_labels.csv')

exit()

# from mlfinlab.data_structures import (get_ema_dollar_run_bars)
#
# # EMA Run Bars
# dollar_imbalance_ema = get_ema_dollar_run_bars(df, num_prev_bars=3,
#                                                exp_num_ticks_init=100000,
#                                                exp_num_ticks_constraints=[100, 1000],
#                                                expected_imbalance_window=10000,
#                                                # to_csv=True,
#                                                # output_path=f"Datas/{FILE_ID}_dollar_imbalance_run_bars_ema.csv"
#                                                )
#
# print(dollar_imbalance_ema)
