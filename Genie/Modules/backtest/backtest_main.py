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
        study_name='MMT_RLGL_OILS_study_backtest',
        # study_name='MMT_RLGL_XAUUSD_study',
        # run_mode='genie_pick',
        run_mode='user_pick',
        # Strategy='mini_genie_source/Strategies/RLGL_Strategy.py',
        Strategy='mini_genie_source/Strategies/MMT_RLGL_Strategy.py',
        data_files_names=[
            # 'US_Light',
            'US_Brent',
            # 'XAUUSD',

        ],
        # data_files_names=['OILUSD'],
        tick_size=[
            # 0.01,
            0.01,
            # 0.0001,

        ],
        init_cash=1_000_000,
        size=100_000,
        # 2019.01.02
        start_date=datetime(month=4, day=3, year=2019),  # Give 90 days to [warmup candles]/[train]
        end_date=datetime(month=10, day=13, year=2022),
        #
        Continue=False,
        batch_size=100,
        timer_limit=None,
        stop_after_n_epoch=3,
        # max_initial_combinations=1_000_000_000,
        # max_initial_combinations=500_000_000,
        # max_initial_combinations=1_000_000,
        max_initial_combinations=1000,
        trading_fees=0.00005,  # 0.00005 or 0.005%, $5 per $100_000
        max_orders=5,
    ),
    # Data_Manager=dict(
    #     report=True,
    # ),
    # Filters=dict(
    #     # fixme needs a path, currently as it is -> f'{Spaces_Program_Info["working_dir"]}/Studies/{self.study_name}'
    #     study_name='MMT_RLGL_OILS_study',
    #
    #
    #     # study_name='Test_Study',
    #     # study_name='Study_OILUSD',
    #     Min_total_trades=1,
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

exit()
#
# config_template = dict(
#     # Data Settings
#     Data_Settings=dict(
#         load_CSV_from_pickle=True,  # momentary
#         data_files_dir='Datas',  # momentary
#         data_files_names=['AMZN'],  # momentary
#
#         delocalize_data=True,
#         drop_nan=False,
#         ffill=False,
#         fill_dates=False,
#         saved_data_file='SymbolData',
#         tick_size=[0.01],
#         minute_data_input_format="%m.%d.%Y %H:%M:%S",
#         minute_data_output_format="%m.%d.%Y %H:%M:%S",
#         #
#         accompanying_tick_data_input_format="%m.%d.%Y %H:%M:%S.%f",
#         accompanying_tick_data_output_format="%m.%d.%Y %H:%M:%S.%f",
#         #     2021-10-03 22:04:00
#     ),
#
#     Simulation_Settings=dict(
#         study_name='debug',
#         optimization_period=dict(
#             start_date=datetime(month=3, day=4, year=2022),
#             end_date=datetime(month=7, day=26, year=2022)
#             # end_date=datetime.datetime(month=10, day=1, year=2021)
#         ),
#         #
#         timer_limit=timedelta(days=365, hours=0, minutes=0, seconds=0),
#         Continue=True,
#         run_mode='ludicrous',
#         #
#         batch_size=100,
#         save_every_nth_chunk=1,
#         Initial_Search_Space=dict(
#             path_of_initial_metrics_record='saved_param_metrics.csv',
#             path_of_initial_params_record='saved_initial_params.csv',
#             max_initial_combinations=1000,
#             stop_after_n_epoch=200,
#             #
#             parameter_selection=dict(
#                 timeframes='all',  # todo: needs to add settings for how to reduce, these dont do anything
#                 windows='grid',  # todo: needs to add settings for how to reduce, these dont do anything
#                 tp_sl=dict(
#                     bar_atr_days=timedelta(days=90, hours=0, minutes=0, seconds=0),
#                     bar_atr_periods=[14],  # todo multiple inputs
#                     bar_atr_multiplier=[2],  # todo multiple inputs
#                     #
#                     n_ratios=[0.5, 1, 1.5],  # Scaling factor for \bar{ATR}
#                     gamma_ratios=[1, 1.5],  # Risk Reward Ratio
#                     number_of_bar_trends=1,
#                 ),
#             ),
#         ),
#
#         Loss_Function=dict(
#             metrics=[
#                 'Total Return [%]',
#                 # 'Benchmark Return [%]',
#                 # 'Max Gross Exposure [%]',
#                 # 'Total Fees Paid',
#                 # 'Max Drawdown [%]',
#                 'Expectancy',
#                 'Total Trades',
#                 # 'Win Rate [%]',
#                 # 'Best Trade [%]',
#                 # 'Worst Trade [%]',
#                 # 'Avg Winning Trade [%]',
#                 # 'Avg Losing Trade [%]',
#                 # 'Profit Factor',
#                 # 'Sharpe Ratio',
#                 # 'Omega Ratio',
#                 # 'Sortino Ratio',
#             ],
#         ),
#         #
#         Optuna_Study=dict(
#             sampler_name=None,
#             multi_objective_bool=None, )
#
#     ),
#     Portfolio_Settings=dict(
#         # Simulation Settings
#         Simulator=dict(
#             Strategy="mini_genie_source/Strategies/Money_Maker_Strategy.py.MMT_Strategy",
#             backtesting="mini_genie_source/Simulation_Handler/flexible_simulation.py.Flexible_Simulation_Backtest",
#             optimization="mini_genie_source/Simulation_Handler/flexible_simulation.py.Flexible_Simulation_Optimization",
#         ),
#         #
#         sim_timeframe='1m',
#         JustLoadpf=False,
#         slippage=0,  # 0.0001,
#         max_spread_allowed=np.inf,
#         trading_fees=0.00005,  # 0.00005 or 0.005%, $5 per $100_000
#         cash_sharing=False,
#         group_by=[],  # Leave blank
#         max_orders=1,
#         init_cash=1_000_000,
#         size_type='cash',  # 'shares',  # cash or shares
#         size=100_000,  # cash, else set size type to shares for share amount
#         type_percent=False,  # if true then take_profit and stop_loss are given in percentages, else cash amount
#     ),
#     # It faster when values given, if not pass 'auto' and I will do my best
#     RAY_SETTINGS=dict(
#         ray_init_num_cpus=cpu_count() - 4,
#         simulate_signals_num_cpus=cpu_count() - 4,
#     )
# )
#
# """
# # In chunks/batches:
# #    1.  Simulate N parameters' indicators
# #    2.  Simulate N parameters' events
# #    3.  Compute Metrics
# #    4.  Save Results to file
# """
# from mini_genie_source.mini_Genie_Object.mini_genie import mini_genie_trader
#
# # Initiate the genie object
# '''
# The genie object works as an operator, can act on itself through its methods, can be acted upon by other
# operators, and must always return the latest state of genie_operator.
# '''
#
# genie_object = mini_genie_trader(runtime_kwargs=config_template, args=arg_parser_values)
#
# exit()
#
# # Load symbols_data, open, low, high, close to genie object.
# genie_object.fetch_and_prepare_input_data()
#
# # todo update with new changes  in dev branch
# '''
#  List of Initial Params:
#       Product of:
#           All Categorical Params
#           Use a grid-like approach to windows for indicators
#           For TP and SL use the avg ATR for the 3 months prior to the optimization date window for every
#               timeframe (when possible do this separately for upwards, downwards and sideways, then use
#               these values separately during the strategy or average them for a single value) then:
#                   1.  Using \bar{ATR}(TF), define multiple TP_0 and SL_0 for and scale with n ratios [ 0.5, 1, 1.5, 2, 2.5]
#                   2.  Use (TP/SL) \gamma ratios like [ 1, 1.2, 1.5, 1.75, 2, 2.5, 3]
#                       (e.g. -> \bar{ATR}(TF='1h')=1000, n=2 and \gamma=1.5, -> R=\bar{ATR}(TF='1h') * n=500
#                           ==> TP=750 & SL=-500)
#                       (e.g. -> \bar{ATR}(TF='1h')=1000, n=2 and \gamma=1.0, -> R=\bar{ATR}(TF='1h') * n=500
#                           ==> TP=500 & SL=-500)
#                       (e.g. -> \bar{ATR}(TF='1h')=1000, n=2 and \gamma=0.5, -> R=\bar{ATR}(TF='1h') * n=500
#                           ==> TP=500 & SL=-750
#
#                       (e.g. -> \bar{ATR}(TF='1d')=2600, n=1 and \gamma=1, -> R=\bar{ATR}(TF='1h') * n=2600
#                           ==> TP=2600 & SL=-2600
#
#     Run product of unique param values in each category, and remove out of bound tp and sl combinations.
#   '''
#
# # Determine initial search space size and content
# #   Initiates:       genie_object._initiate_parameters_records
# #                    genie_object._initiate_metric_records
# #   Fills:           genie_object._initiate_parameters_records
# genie_object.suggest_parameters()
#
# # Run the optimization  and save the results to file
# genie_object.simulate()
# #
# # elif arg_parser_values.user_pick:
# #     genie_object.prepare_backtest()
# #     #
# #     genie_object.simulate()
# #
# # if __name__ == "__main__":
# #     from Run_Time_Handler.run_time_handler import run_time_handler
# #
# #     setup_logging(full_context=1)
# #
# #     #
# #     run_time_handler = run_time_handler(run_function=call_genie)
# #     run_time_handler.call_run_function()
