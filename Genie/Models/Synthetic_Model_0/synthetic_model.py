# import os
#
# import pandas as pd
#
# from Models.Synthetic_Model_0.__utils import __genie_strategy__
# from Modules._Data_Manager import Data_Manager
#
# # Asjust the datetime index of the HCBM data by stretching or compressing it in order for the synthetic data to
# # match the real data's daily volatility
#
# os.environ['MLFINLAB_API_KEY'] = "0800b4ea410a702acddefdec86f93523"
#
# import numpy as np
#
# from Genie.Modules._Synthetic_Data import HCBM_Synthetic_TS_Data
import pandas as pd

from Modules._Data_Manager import Data_Manager
from Modules._Synthetic_Data import HCBM_Synthetic_TS_Data

# FILE_NAME = "USA100_bid_ask_price_sample.csv"
# FILE_ID = FILE_NAME.split(".")[0]



hcbm_namedtuple = HCBM_Synthetic_TS_Data(
    # Initialize parameters
    n_assets=1,  # Number of assets to generate (default=1)
    n_paths=10,  # Number of paths to generate (default=200)
    rho_low=0.1,  # minimum correlation between assets (default=0.1)
    rho_high=0.9,  # maximum correlation between assets (default=0.9)
    n_bars=10,  # len(real_strategy_namedtuple.daily_vol),  # number of bars in each time series (default=1000)
    distribution="student",  # distribution of returns (default="normal") "normal", "student"
    method="ward",  # method for hierarchical clustering (default="ward")
    # “single”, “complete”, “average”, “weighted”, “centroid”, “median”, Default: “ward”
    permute=True,  # Whether to permute the final HCBM matrix (default=False)
    plot=False,  # Whether to plot the HCBM matrices and TS distributions (default=False)
    # **kwargs
    # Returns:namedtuple
    # hcbm_namedtuple__ .<['n_assets', 'n_paths', 'rho_low', 'rho_high', 'n_bars', 'dist', 'method', 'permute',
    #                      'hcbm_mats': (n_assets,n_paths * 2),
    #                      'ts_1', 'ts_2', ..., n_assets]>
)

# Todo for all assets
# for asset in range(synthetic_namedtuple.n_assets):
# for asset_index in range(synthetic_namedtuple.n_assets):
#     synthetic_ts_name = f"ts_{asset_index + 1}"
#     synthetic_ts_i = eval(f"synthetic_namedtuple.{synthetic_ts_name}")


# # Adjust the synthetic data to match the real data's volatility # todo
# hcbm_namedtuple = __adjust_stat_properties__(real_price_series=seed_asset_closing_price,
#                                         synthetic_prices_df=hcbm_namedtuple.ts_1)


synthetic_price_series = pd.Series(hcbm_namedtuple.df.values.ravel('F')).add(100)

synthetic_strategy_namedtuple = __genie_strategy__(synthetic_price_series,
                                                   volatility_window=2)
print(synthetic_strategy_namedtuple.df.to_csv("test_synth_data.csv"))
exit()

parameter_data = dict(
    Trend_filter_1_timeframes=['5 min', '1d'],
    Trend_filter_atr_windows=np.linspace(start=5, stop=45, num=number_of_suggestions, dtype=int),
    Trend_filter_1_data_lookback_windows=[2, 9, 14, 16],
    #
    PEAK_and_ATR_timeframes=['5 min', '1d'],
    atr_windows=np.linspace(start=5, stop=45, num=number_of_suggestions, dtype=int),
    data_lookback_windows=np.linspace(start=5, stop=45, num=number_of_suggestions, dtype=int),
    #
    EMAs_timeframes=['5 min', '1d'],
    ema_1_windows=np.linspace(start=5, stop=45, num=number_of_suggestions, dtype=int),
    ema_2_windows=np.arange(20, 60, step=20, dtype=int),
    #
    take_profit_points=np.linspace(start=lower_bound_tp, stop=upper_bound_tp, num=number_of_suggestions,
                                   dtype=type(lower_bound_tp)),
    #
    stop_loss_points=np.linspace(start=lower_bound_tp, stop=upper_bound_tp, num=number_of_suggestions,
                                 dtype=type(upper_bound_tp)),
)

"""Calling the function directly"""
long_entries, long_exits, short_entries, short_exits, \
strategy_specific_kwargs = Strategies.MMT(
    open_data=open_data,
    low_data=low_data,
    high_data=high_data,
    close_data=close_data,
    parameter_data=parameter_data, ray_sim_n_cpus=28, param_product=True)
