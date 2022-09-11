import numpy as np

from legendary_genie import walkfoward_report, Strategies, Data_Manager, vbt

"""Downloading/Loading Data"""
# > Downloads the last N bars of timeframe X data from YFData or Binance (CCXTData library)
# There are multiple methods of aquiring data, it can be downloaded or loaded from a file and the function you use
# will depend on the data you are using. Let's start by downloading some data from yahoo finance.

# DAYS_TO_DOWNLOAD = 1000
# end_time = datetime.now()
# start_time = end_time - timedelta(days=DAYS_TO_DOWNLOAD)
# data = vbt.YFData.fetch(
#     # ["BTC-USD", "ETH-USD", "XMR-USD", "ADA-USD"],
#     # ["BTC-USD", "ETH-USD"],
#     # "BTC-USD",
#     "ETH-USD",
#     start=start_time,
#     end=end_time,
#     timeframe="1d",
#     missing_index='drop'  # Creates problems with missing the index
# )
# print(f"YFData: ")
# print(data)
# #
# # > the same can be achieved by using the ccxt lib which gives you access to a lot more crypto data sources and brokers
# #
# data = vbt.CCXTData.fetch(
#     "ETHUSDT",
#     exchange="binance",
#     start=start_time,
#     end=end_time,
#     timeframe="1 day"
# )
# print(f"CCXTData: ")
# print(data)
# print(type(data))
#
# > You can also load data from a file
data_file_names = [
    "dollar_bars.csv",
]

# Searches for the data file names of csv type within the given directories
data = Data_Manager.fetch_data(data_file_names=data_file_names,
                               data_file_dirs=[".", "Datas", "Sample-Data"])
#
# This returns a vbt data object <class 'vectorbtpro.data.custom.<source>'> which essensially wraps the pandas
#   dataframe introducing new functionality and all of these can be saved and loaded as followed:
data.save("example_data")
data = vbt.Data.load("example_data")
# > Loading once again a <class 'vectorbtpro.data.base.Data'> object
print(type(data))

# > I will cover how to access the data in the next section however for now lets take a look at how to get the OHLC data
# Split the Data
try:
    open_data = data.get('Open')
    low_data = data.get('Low')
    high_data = data.get('High')
    close_data = data.get('Close')
except:
    open_data = data.get('open')
    low_data = data.get('low')
    high_data = data.get('high')
    close_data = data.get('close')

print(open_data.head())
# I know you are a pro and will pick this up in no time, to show you how easy it is to get started using the tools
#   gathered by Genie, lets start with a simple example beginning from raw tick data (in example 2 well learn how to
#   build user defined functions).


"""_________________________________________Example 1: Simple Walkthrough_________________________________________"""

"""Reading from CSV Data"""

# Let's start with a csv file containing tick data df of columns [date_time, price, volume]. Tick data files can get
#   quite large thus memory expensive, so lets load the data in chunks of N rows at a time and only save to memory the
#   data we want rather than loading the source data at the same time the resampled strategy is being filled for
#   example. When analyzing financial data, unstructured data sets, in this case tick data, are commonly transformed
#   into a structured format referred to as bars, where a bar represents a row in a table. MlFinLab implements tick,
#   volume, and dollar bars using traditional standard bar methods as well as the less common information driven bars.
#   Although it is useful for accuracy when every fraction of a second counts, tick data does not provide much
#   information about the price on its own arriving at inconsistent times so lets learn about how to use these data_structures
# data = Data_Manager.fetch_data(data_file_names='tick_data.csv',
#                                data_file_dirs=[".", "Datas", "Sample-Data"])
# print(data.wrapper.columns)  # Index(['Price', 'Volume'], dtype='object')
# price = data.get('Price')
# volume = data.get('Volume')
# print(price.head())




"""Preparing the Data"""
# Neither VBT nor Genie discriminate against the use of tick data at core, however more often than not you will be using an
# aggregated form of data such as minute OHLC or dollar bars (just to name a few) for use by the indicators (much faster), while the tick_data can be
# left as is, and used for the backtest. This can be computationally expensive but results in the most accurate
# simulation possible thanks to Genie's and VBT great teamwork. Let's start by aggregating the data into minute bars.

# > Aggregating the data into minute bars
# get_bars(data, out_bar_type='time', kwargs={})


print()
exit()

time_bars = Data_Manager.get_bars(
    data, out_bar_type='time',
    kwargs=dict(
        # For All Bar Types
        batch_size=1000000,
        verbose=False,
        #
        # For Time Bars
        file_path_or_df=None,  # (str, iterable of str, or pd.DataFrame) Path to
        # the csv file(s) or Pandas Data Frame containing raw tick data in the
        # format[date_time, price, volume]
        resolution='MIN',  # (str) Resolution type ('D', 'H', 'MIN', 'S')
        num_units=1,  # (int) Number of resolution units (3 days for example, 2 hours)
        # (int) The number of rows per batch. Less RAM = smaller batch size.
        to_csv=None,  # (bool) Save bars to csv after every batch run (True or False)
        output_path=None,  # (str) Path to csv file, if to_csv is True
        # returns (pd.DataFrame) Dataframe of time bars, if to_csv=True return None
        #
        # For Tick Bars
        threshold=10,
        # average=False,
    ))

print(time_bars)

exit()
# Parameter Data structure depends on your custom strategy. MMT will be used this example, it can accept a structured array or a dictionary
# e.g. (showing various methods of creating parameters to plug in to genie, up to you, is very flexible in the lengths of the arrays)
number_of_suggestions = 1
upper_bound_tp = 10000
lower_bound_tp = 100

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
print(long_exits)
###################################
exit()

# strategy = Strategies.STFA  # Strategies.RLGL_Strategy # Strategies.SuperTrendFastAF

# parameter_data=dict(
# period_windows = np.linspace(start=4, stop=45, num=2, dtype=int),
# multiplier_windows = np.linspace(start=2, stop=5, num=2, dtype=int)
# )

"""Walk Forward Optimization"""
walkfoward_kwargs = dict(
    # <class 'function'> Function to be used to initialize the strategy
    strategy=Strategies.MMT,
    # Strategy must accept any column you select here
    # e.g. MMT_Strategy(open_data, low_data, high_data, close_data, parameter_data, ray_sim_n_cpus)
    columns=['open', 'low', 'high', 'close'],
    # columns=['Open', 'Low', 'High', 'Close'],
    parameter_data=parameter_data,
    ray_sim_n_cpus=28,
    #
    # The following are optional, but it is how you adjust your windows for the walk forward optimization
    num_splits=10,
    n_bars=10,
)

result = walkfoward_report(data, **walkfoward_kwargs)

# print(result)

#
#
# #
# run_time_settings = {
#     'Data_Settings': {'load_CSV_from_pickle': True, 'data_files_dir': 'Datas', 'data_files_names': ['XAUUSD'],
#                       'delocalize_data': True, 'drop_nan': False, 'ffill': False, 'fill_dates': False,
#                       'saved_data_file': 'SymbolData', 'tick_size': [0.001],
#                       'minute_data_input_format': '%m.%d.%Y %H:%M:%S', 'minute_data_output_format': '%m.%d.%Y %H:%M:%S',
#                       'accompanying_tick_data_input_format': '%m.%d.%Y %H:%M:%S.%f',
#                       'accompanying_tick_data_output_format': '%m.%d.%Y %H:%M:%S.%f'},
#     'Simulation_Settings': {'study_name': 'Test_Study',
#                             'optimization_period': {'start_date': Timestamp('2022-03-04 00:00:00'),
#                                                     'end_date': Timestamp('2022-07-07 00:00:00')},
#                             'timer_limit': datetime.timedelta(days=365), 'Continue': False, 'run_mode': 'plaid_plus',
#                             'batch_size': 5000, 'save_every_nth_chunk': 1,
#                             'Initial_Search_Space': {'path_of_initial_metrics_record': 'saved_param_metrics.csv',
#                                                      'path_of_initial_params_record': 'saved_initial_params.csv',
#                                                      'max_initial_combinations': 1200000000, 'stop_after_n_epoch': 200,
#                                                      'parameter_selection': {'timeframes': 'all', 'windows': 'grid',
#                                                                              'tp_sl': {
#                                                                                  'bar_atr_days': datetime.timedelta(
#                                                                                      days=90), 'bar_atr_periods': [7],
#                                                                                  'bar_atr_multiplier': [3],
#                                                                                  'n_ratios': [0.5, 1, 1.5],
#                                                                                  'gamma_ratios': [1, 1.5],
#                                                                                  'number_of_bar_trends': 1}}},
#                             'Loss_Function': {'metrics': ['Total Return [%]', 'Expectancy', 'Total Trades']},
#                             'Optuna_Study': {'sampler_name': None, 'multi_objective_bool': None}},
#     'Portfolio_Settings': {'Simulator': {'Strategy': 'mini_genie_source/Strategies/RLGL_Strategy.py',
#                                          'backtesting': 'mini_genie_source/Simulation_Handler/flexible_simulation.py.Flexible_Simulation_Backtest',
#                                          'optimization': 'mini_genie_source/Simulation_Handler/flexible_simulation.py.Flexible_Simulation_Optimization'},
#                            'sim_timeframe': '1m', 'JustLoadpf': False, 'slippage': 0, 'max_spread_allowed': inf,
#                            'trading_fees': 5e-05, 'cash_sharing': False, 'group_by': [], 'max_orders': 10,
#                            'init_cash': 1000000.0, 'size_type': 'cash', 'size': 100000.0, 'type_percent': False},
#     'RAY_SETTINGS': {'ray_init_num_cpus': 28, 'simulate_signals_num_cpus': 28}}
# #
# # mini_genie(run_time_settings, )
# from Genie_API.genie_api import ApiHandler as Genie_API_Handler
#
# EXAMPLE_INPUT_DICT = dict(
#     Genie=dict(
#         study_name='RLGL_XAUUSD',
#         # run_mode='legendary',
#         run_mode='legendary_genie',
#         Strategy='mini_genie_source/Strategies/RLGL_Strategy.py',
#         data_files_names=['XAUUSD'],
#         # data_files_names=['OILUSD'],
#         tick_size=[0.001],
#         init_cash=1_000_000,
#         size=100_000,
#         start_date=datetime.datetime(month=3, day=4, year=2022),
#         end_date=datetime.datetime(month=7, day=7, year=2022),
#         #
#         Continue=False,
#         batch_size=2,
#         timer_limit=None,
#         stop_after_n_epoch=5,
#         # max_initial_combinations=1_000_000_000,
#         max_initial_combinations=1000,
#         trading_fees=0.00005,  # 0.00005 or 0.005%, $5 per $100_000
#         max_orders=1000,
#     ),
#     # Data_Manager=dict(
#     #     report=True,
#     # ),
#     # Filters=dict(
#     #     study_name='RLGL_AUDUSD',
#     #     # study_name='Test_Study',
#     #     # study_name='Study_OILUSD',
#     #     Min_total_trades=1,
#     #     Profit_factor=1.0,
#     #     Expectancy=0.01,
#     #     Daily_drawdown=0.05,
#     #     Total_drawdown=0.1,
#     #     Profit_for_month=0.1,
#     #     Total_Win_Rate=0.03,
#     #     quick_filters=True,
#     #     # delete_loners=True,
#     # ),
#     # MIP=dict(
#     #     agg=True,
#     # ),
#     # Neighbors=dict(
#     #     n_neighbors=20,
#     # ),
#     # Data_Manager=dict(
#     #     delete_first_month=True,
#     # ),
#     # Overfit=dict(
#     #     cscv=dict(
#     #         n_bins=10,
#     #         objective='sharpe_ratio',
#     #         PBO=True,
#     #         PDes=True,
#     #         SD=True,
#     #         POvPNO=True,
#     #     )
#     # ),
#     # Data_Manager_1=dict(  # todo add _{} parser
#     #     n_split=2,
#     # ),
#     # Filters=dict(
#     #     quick_filters=True,
#     #     delete_loners=True,
#     # ),
# )
#
# api_handler = Genie_API_Handler(EXAMPLE_INPUT_DICT)
# api_handler.parse()
# # print(api_handler.df[['Template_Code', 'Variable_Value']])
# api_handler.run()
