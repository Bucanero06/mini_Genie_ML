import datetime

from numpy import inf
from pandas import Timestamp

run_time_settings = {
    'Data_Settings': {'load_CSV_from_pickle': True, 'data_files_dir': 'Datas', 'data_files_names': ['XAUUSD'],
                      'delocalize_data': True, 'drop_nan': False, 'ffill': False, 'fill_dates': False,
                      'saved_data_file': 'SymbolData', 'tick_size': [0.001],
                      'minute_data_input_format': '%m.%d.%Y %H:%M:%S', 'minute_data_output_format': '%m.%d.%Y %H:%M:%S',
                      'accompanying_tick_data_input_format': '%m.%d.%Y %H:%M:%S.%f',
                      'accompanying_tick_data_output_format': '%m.%d.%Y %H:%M:%S.%f'},
    'Simulation_Settings': {'study_name': 'RLGL_XAUUSD',
                            'optimization_period': {'start_date': Timestamp('2022-03-04 00:00:00'),
                                                    'end_date': Timestamp('2022-07-07 00:00:00')},
                            'timer_limit': datetime.timedelta(days=365), 'Continue': False, 'run_mode': 'ludicrous',
                            'batch_size': 2, 'save_every_nth_chunk': 1,
                            'Initial_Search_Space': {'path_of_initial_metrics_record': 'saved_param_metrics.csv',
                                                     'path_of_initial_params_record': 'saved_initial_params.csv',
                                                     'max_initial_combinations': 1000, 'stop_after_n_epoch': 5,
                                                     'parameter_selection': {'timeframes': 'all', 'windows': 'grid',
                                                                             'tp_sl': {
                                                                                 'bar_atr_days': datetime.timedelta(
                                                                                     days=90), 'bar_atr_periods': [14],
                                                                                 'bar_atr_multiplier': [2],
                                                                                 'n_ratios': [0.5, 1, 1.5],
                                                                                 'gamma_ratios': [1, 1.5],
                                                                                 'number_of_bar_trends': 1}}},
                            'Loss_Function': {'metrics': ['Total Return [%]', 'Expectancy', 'Total Trades']},
                            'Optuna_Study': {'sampler_name': None, 'multi_objective_bool': None}},
    'Portfolio_Settings': {'Simulator': {'Strategy': 'mini_genie_source/Strategies/RLGL_Strategy.py',
                                         'backtesting': 'mini_genie_source/Simulation_Handler/flexible_simulation.py.Flexible_Simulation_Backtest',
                                         'optimization': 'mini_genie_source/Simulation_Handler/flexible_simulation.py.Flexible_Simulation_Optimization'},
                           'sim_timeframe': '1m', 'JustLoadpf': False, 'slippage': 0, 'max_spread_allowed': inf,
                           'trading_fees': 5e-05, 'cash_sharing': False, 'group_by': [], 'max_orders': 1000,
                           'init_cash': 1000000.0, 'size_type': 'cash', 'size': 100000.0, 'type_percent': False},
    'RAY_SETTINGS': {'ray_init_num_cpus': 28, 'simulate_signals_num_cpus': 28}}
