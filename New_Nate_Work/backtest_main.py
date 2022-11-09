#!/usr/bin/env python3
import glob
from datetime import datetime, timedelta
import warnings
from os import cpu_count
import vectorbtpro as vbt
import numpy as np
from logger_tt import setup_logging, logger  # noqa: F401

from genie_api_handler import ApiHandler

setup_logging(full_context=1)

warnings.simplefilter(action='ignore', category=FutureWarning)

# exit()
EXAMPLE_INPUT_DICT = dict(
    Genie=dict(
        # study_name='US_Light_debugging_study',
        # study_name='XAUUSD_debugging_study',
        study_name='US_Brent_debugging_study',
        # study_name='ray_debugging_study',
        run_mode='genie_pick',
        # run_mode='user_pick',
        # Strategy='mini_genie_source/Strategies/RLGL_Strategy.py',
        Strategy='mini_genie_source/Strategies/MMT_RLGL_Strategy.py',
        data_files_names=[
            # 'US_Light',
            'US_Brent',
            # 'XAUUSD',
        ],
        # data_files_names=['OILUSD'],
        tick_size=[
            0.01,
            # 0.01,
            # 0.01,
        ],
        init_cash=1_000_000,
        size=100_000,
        start_date=datetime(month=1, day=3, year=2021),  # Give 90 days to [warmup candles]/[train]
        # start_date=datetime(month=4, day=3, year=2022),  # Give 90 days to [warmup candles]/[train]
        end_date=datetime(month=10, day=13, year=2022),
        #
        Continue=True,
        batch_size=100,
        timer_limit=None,
        stop_after_n_epoch=10,
        # max_initial_combinations=1_000_000_000,
        max_initial_combinations=500_000_000,
        # max_initial_combinations=1_000_000,
        # max_initial_combinations=1000,
        # trading_fees=0.00005,  # 0.00005 or 0.005%, $5 per $100_000
        trading_fees=0.0001,  # 0.00005 or 0.005%, $5 per $100_000
        max_orders=1,
    ),
    # Data_Manager=dict(
    #     report=True,
    # ),
    # Filters=dict(
    #     # fixme needs a path, currently as it is -> f'{Spaces_Program_Info["working_dir"]}/Studies/{self.study_name}'
    #     study_name='US_Light_debugging_study',
    #     Min_total_trades=200,
    #     Profit_factor=1.0,
    #     Expectancy=0.01,
    #     Daily_drawdown=0.05,
    #     Total_drawdown=0.1,
    #     Profit_for_month=0.1,
    #     Total_Win_Rate=0.03,
    #     quick_filters=True,
    #     # delete_loners=True,
    # ),
    # MIP=dict(
    #     agg=True,
    # ),
    # Neighbors=dict(
    #     n_neighbors=20,
    # ),
    # Data_Manager=dict(
    #     delete_first_month=True,
    # ),
    # Overfit=dict(
    #     cscv=dict(
    #         n_bins=10,
    #         objective='sharpe_ratio',
    #         PBO=True,
    #         PDes=True,
    #         SD=True,
    #         POvPNO=True,
    #     )
    # ),
    # Data_Manager_1=dict(  # todo add _{} parser
    #     n_split=2,
    # ),
    # Filters=dict(
    #     quick_filters=True,
    #     delete_loners=True,
    # ),
)

api_handler = ApiHandler(EXAMPLE_INPUT_DICT)
api_handler.parse()
# print(api_handler.df[['Template_Code', 'Variable_Value']])
api_handler.run()




# split_kwargs = dict(
#             columns=['open', 'low', 'high', 'close'],
#             num_splits=10,
#             n_bars=10,
#         )
#
#         from Modules.Utils import range_split_ohlc
#         prices_columns = range_split_ohlc(symbols_data, ['open', 'low', 'high', 'close'], 10, 10)