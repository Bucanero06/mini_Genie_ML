from os import cpu_count, environ

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

environ['MLFINLAB_API_KEY'] = "0800b4ea410a702acddefdec86f93523"
from Genie import genieml_src as ml


def __adjust_stat_properties__(real_price_series, synthetic_prices_df):
    ______starting_point = 82.1310
    print(f"synthetic_prices_df \n{synthetic_prices_df}\n")
    synthetic_prices_df.add(______starting_point)

    # print()
    # Remove columns with values that are negative
    print("(synthetic_prices_df > 0).all(axis=0).to_list()")
    print((synthetic_prices_df > 0).all(axis=0).sum())
    columns_to_keep = synthetic_prices_df.columns[(synthetic_prices_df > 0).all(axis=0).to_list()]
    print(f"columns_to_keep \n{columns_to_keep}\n")
    synthetic_prices_df = synthetic_prices_df[columns_to_keep]
    print(f"positive synthetic_prices_df \n{synthetic_prices_df}\n")

    # print(synthetic_prices_df)

    exit()
    target_var = real_price_series.var()
    synthetic_var_df = synthetic_prices_df.var()
    print("Initial_synthetic_var")
    print(synthetic_var_df)

    print("target_var")
    print(target_var)
    # (2)
    # Variance of synthetic time-series = (Variance of target time-series * Variance of synthetic time-series) / Variance of target time-series
    synthetic_var = (target_var * synthetic_var_df) / target_var

    # (3)
    synthetic_daily_volatility = np.sqrt(synthetic_var) / np.sqrt(len(synthetic_prices_df.index))
    print("synthetic_daily_volatility")
    print(synthetic_daily_volatility)

    real_daily_volatility = np.sqrt(target_var) / np.sqrt(252)
    print("real_daily_volatility")
    print(real_daily_volatility)

    ratio = (synthetic_daily_volatility / real_daily_volatility)
    print("ratio")
    print(ratio)

    print("original price_series")
    print(real_price_series)
    print("New price_series")
    print(synthetic_prices_df)
    print(synthetic_prices_df / ratio)

    print((synthetic_prices_df / ratio).add(______starting_point))
    exit()

    return synthetic_namedtuple


def random_date_df(start, end, n):
    """
    Generates a DataFrame with n rows, index is random date between start and end
    """
    start_u = start.value // 10 ** 9
    end_u = end.value // 10 ** 9
    return pd.DataFrame(
        {
            "Date":
                pd.to_datetime(
                    np.random.randint(start_u, end_u, n), unit='s'
                )
        }
    ).set_index('Date')


def smart_datetimeindex(start, end, freq):
    start = pd.to_datetime(start)  # convert to datetime
    end = pd.to_datetime(end)
    date_range = pd.date_range(start, end, freq=freq)
    date_range = date_range.sort_values()
    return date_range


def __genie_strategy__(
        # price_series is a pandas.Series object containing the price data for the strategy to be backtested on (e.g. close price)
        open,  # pandas.Series
        high,  # pandas.Series
        low,  # pandas.Series
        close,  # pandas.Series
        num_days=None,  # (int) Number of days to add for vertical barrier.
        num_hours=None,  # (int) Number of hours to add for vertical barrier.
        num_minutes=None,  # (int) Number of minutes to add for vertical barrier.
        num_seconds=None,  # (int) Number of seconds to add for vertical barrier.
        num_bars=None,  # (int) Number of bars (samples) after which to construct vertical barriers (None by default).
        #
        pt_sl=None,
        # min_ret is the minimum return value (e.g. 0.004)
        min_ret=None,
        # threads is the number of threads to be used for the backtest (e.g. None)
        threads=None,
        # lookback is the number of days to be used for the volatility calculation (e.g. 50)
        lookback=None,
        #
        volatility_window=None,
        threshold=None,
        REAL=False,
        fake_freq=None,
        **kwargs):
    """

    """

    if pt_sl is None:
        pt_sl = [1, 1]
    if min_ret is None:
        min_ret = 0.00005
    if threads is None:
        threads = cpu_count() - 2
    if volatility_window is None:
        volatility_window = 100

    if num_days is None:
        num_days = 0
    if num_hours is None:
        num_hours = 0
    if num_minutes is None:
        num_minutes = 0
    if num_seconds is None:
        num_seconds = 0
    if fake_freq is None:
        fake_freq = "1D"
    if REAL is None:
        REAL = False

    # Get_volatility
    # if not REAL:
    #     start = '1970-01-01 00:00:00.001'
    #     end = pd.to_datetime(start) + pd.Timedelta(seconds=len(price_series) - 1)
    #     price_series.index = smart_datetimeindex(start=start, end=end, freq=fake_freq)
    #     #
    #     # volatility = price_series.rolling(volatility_window).var()
    #
    #
    print("volatility")
    volatility = ml.util.get_yang_zhang_vol(open=open, high=high, low=low, close=close, window=volatility_window)
    vol_mean = volatility.mean()

    """
    Suppose we use a mean-reverting strategy as our primary model, giving each observation a label of -1 or 1. 
    We can then use meta-labeling to act as a filter for the bets of our primary model.

    Assuming we have a pandas series with the timestamps of our observations and their respective labels given by the 
    primary model, the process to generate meta-labels goes as follows.
    """

    # Apply Symmetric CUSUM Filter and get timestamps for events
    # Note: Only the CUSUM filter needs a point estimate for volatility
    print("cusum_filter")
    cusum_events = ml.filters.cusum_filter(close,
                                           threshold=vol_mean if not threshold else threshold)

    # Compute vertical barrier using timedelta or number of bars (num_bars) after event time (t1)
    #     Compute vertical barrier using timedelta after a fixed number of samples (ticks) have passed
    #     since the event time (t1) has been reached (t1 + num_bars)
    print("vertical_barriers")
    vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events,
                                                         close=close,
                                                         # num_days=num_days,
                                                         # num_hours=num_hours,
                                                         # num_minutes=num_minutes,
                                                         # num_seconds=num_seconds,
                                                         num_bars=volatility_window if not num_bars else num_bars
                                                         )

    """
    Once we have computed the daily volatility along with our vertical time barriers and have downsampled our series 
    using the CUSUM filter, we can use the triple-barrier method to compute our meta-labels by passing in the side 
    predicted by the primary model.
    """
    print("triple_barrier_events")
    # Get events and labels for each event using the triple barrier method
    triple_barrier_events = ml.labeling.get_events(close=close,
                                                   t_events=cusum_events,
                                                   pt_sl=pt_sl,
                                                   target=volatility,
                                                   min_ret=min_ret,
                                                   num_threads=threads,
                                                   vertical_barrier_times=vertical_barriers,
                                                   )
    print("triple_barrier_events ended")

    print(f'{len(cusum_events) = }')
    print(f'{len(volatility) = }')
    print(f'{len(vertical_barriers) = }')
    print(f'{len(triple_barrier_events) = }')

    """
    We scale our lower barrier by adjusting our minimum return via min_ret.

    Warning

    The biggest mistake often made here is that they change the daily targets and min_ret values to 
    get more observations, since ML models require a fair amount of data. This is the wrong approach!

    Please visit the Seven-Point Protocol under the Backtest Overfitting Tools section to learn more about 
    how to think about features and outcomes. (very important)

    Meta-labels can then be computed using the time that each observation touched its respective barrier.

    Note

    The triple barrier method is a very powerful tool for labeling data, but it is not without its drawbacks. 
    The biggest drawback is that it is computationally expensive. The triple barrier method is a brute force
    approach to labeling data, and as such, it is not very efficient. This is why we use the CUSUM filter
    to downsample our data before labeling it. The CUSUM filter is a very efficient way to downsample data,
    """
    print("get_bins")
    labels_one = ml.labeling.get_bins(triple_barrier_events, close)

    """Trend_scanning  """
    # exit()
    # """#############################################################################################################"""
    #
    # import matplotlib.pyplot as plt
    #
    # # Import MlFinLab tools
    # from mlfinlab.labeling import trend_scanning_labels
    #
    # # Loading data to use
    # eem_close = pd.read_csv('./test_data/stock_prices.csv', index_col=0, parse_dates=[0])
    #
    # # Choosing a period where trends would be seen (mid-2008  -  mid-2009)
    # eem_close = self.eem_close['EEM'].loc[pd.Timestamp(2008, 7, 1):pd.Timestamp(2009, 7, 1)]
    #
    # # Getting indexes that we want to label
    # t_events = self.eem_close.index
    #
    # # Fitting regressions to various windows up to 20 days back, using a minimum sample length of 5
    # tr_scan_labels = trend_scanning_labels(eem_close, t_events, observation_window=20,
    #                                        look_forward=False, min_sample_length=5)
    #
    # # Plotting the results
    # fig = plt.figure(figsize=(12, 7))
    # plt.scatter(x=eem_close.index, y=eem_close.values, c=tr_scan_labels["t_value"], s=200)
    # plt.show()
    #
    # exit()
    # """#############################################################################################################"""

    print("concat")

    labels_one_index = labels_one['bin'].index
    full_df = pd.concat([
        pd.DataFrame(open.loc[labels_one_index], index=labels_one_index),
        high.loc[labels_one_index],
        low.loc[labels_one_index],
        close.loc[labels_one_index],
        triple_barrier_events[['pt', 'sl']].loc[labels_one_index],
        labels_one[['ret', 'trgt', 'bin']].loc[labels_one_index]
    ], axis=1)

    print("printing")

    print(f"cusum_events {len(cusum_events)}")
    print(f"volatility {len(volatility)}")
    print(f"price_series {len(close)}")
    print(f"vertical_barriers {len(vertical_barriers)}")
    print(f"triple_barrier_events {len(triple_barrier_events)}")
    print(f"labels_one {len(labels_one)}")
    print(f"labels_one {len(full_df)}")
    # print(f"price_series: \n\n{price_series}\n\n")
    # print(f"triple_barrier_events: \n\n{triple_barrier_events}\n\n")
    # print(f"labels_one: \n\n{labels_one}\n\n")
    # print("writing")
    #
    # print(full_df.head(10))
    # print(full_df.tail(10))
    return full_df


from numba import njit


@njit
def corr_meta_nb(from_i, to_i, col, a, b):
    a_window = a[from_i:to_i, col]
    b_window = b[from_i:to_i, col]
    return np.corrcoef(a_window, b_window)[1, 0]


def perform_grid_search(X_data, y_data):
    """
    Function to perform a grid search.
    """
    '''Fit a model'''
    parameters = {'max_depth': [2, 3, 4, 5, 7],
                  'n_estimators': [1, 10, 25, 50, 100, 256, 512],
                  'random_state': [42]}
    rf = RandomForestClassifier(criterion='entropy')
    clf = GridSearchCV(rf, parameters, cv=4, scoring='roc_auc', n_jobs=3)
    clf.fit(X_data, y_data)
    print(clf.cv_results_['mean_test_score'])

    return clf.best_params_['n_estimators'], clf.best_params_['max_depth']


def __genie_strategy__1(data, num_threads=1):
    # Import packages
    import numpy as np
    import vectorbtpro as vbt

    # Import MlFinLab tools

    '''Primary Model'''
    # Compute moving averages
    fast_window = 20
    slow_window = 50

    # STRATEGY!!!!

    SUMCON_indicator = vbt.IF.from_techcon("SUMCON")
    indicator_bs = SUMCON_indicator.run(
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        volume=data["volume"],
        smooth=30
    )
    SUMCON_result = indicator_bs.buy - indicator_bs.sell

    Trend_filter_1_timeframes = np.array(["15 min"])
    Trend_filter_atr_windows = np.array([7])
    Trend_filter_1_data_lookback_windows = np.array([7])
    #
    PEAK_and_ATR_timeframes = np.array(["5 min"])
    atr_windows = np.array([7])
    data_lookback_windows = np.array([7])
    #
    EMAs_timeframes = np.array(["1 min"])
    ema_1_windows = np.array([30])
    ema_2_windows = np.array([33])
    #
    take_profit_points = np.array([100])
    #
    stop_loss_points = np.array([-100])

    '''Compile Structure and Run Master Indicator'''
    # from mini_genie_source.Strategies.Money_Maker_Strategy import apply_function
    # from mini_genie_source.Strategies.Money_Maker_Strategy import cache_func
    # Master_Indicator = vbt.IF(
    #     input_names=[
    #         'low_data', 'high_data', 'close_data',
    #         # 'datetime_index',
    #     ],
    #     param_names=[
    #         'Trend_filter_1_timeframes', 'Trend_filter_atr_windows', 'Trend_filter_1_data_lookback_windows',
    #         'PEAK_and_ATR_timeframes', 'atr_windows', 'data_lookback_windows',
    #         'EMAs_timeframes', 'ema_1_windows', 'ema_2_windows',
    #         'take_profit_points',
    #         'stop_loss_points'
    #     ],
    #     output_names=[
    #         'long_entries', 'long_exits', 'short_entries', 'short_exits',
    #         'take_profit_points', 'stop_loss_points'
    #     ]
    # ).with_apply_func(
    #     apply_func=apply_function,
    #     cache_func=cache_func,
    #     keep_pd=True,
    #     param_product=False,
    #     execute_kwargs=dict(
    #         engine='ray',
    #         init_kwargs={
    #             # 'address': 'auto',
    #             'num_cpus': 28,
    #             'ignore_reinit_error': True,
    #         },
    #         show_progress=True
    #     ),
    #     Trend_filter_1_timeframes='1d',
    #     Trend_filter_atr_windows=5,
    #     Trend_filter_1_data_lookback_windows=3,
    #     PEAK_and_ATR_timeframes='1d',
    #     atr_windows=5,
    #     data_lookback_windows=3,
    #     EMAs_timeframes='1h',
    #     ema_1_windows=13,
    #     ema_2_windows=50,
    #     take_profit_points=300,
    #     stop_loss_points=-600,
    # ).run(
    #     data["low"], data["high"], data["close"],
    #     Trend_filter_1_timeframes=Trend_filter_1_timeframes,
    #     Trend_filter_atr_windows=Trend_filter_atr_windows,
    #     Trend_filter_1_data_lookback_windows=Trend_filter_1_data_lookback_windows,
    #     PEAK_and_ATR_timeframes=PEAK_and_ATR_timeframes,
    #     atr_windows=atr_windows,
    #     data_lookback_windows=data_lookback_windows,
    #     EMAs_timeframes=EMAs_timeframes,
    #     ema_1_windows=ema_1_windows,
    #     ema_2_windows=ema_2_windows,
    #     take_profit_points=take_profit_points,
    #     stop_loss_points=stop_loss_points,
    # )
    # # Compute sides
    # data['side'] = np.nan
    #
    # long_signals = Master_Indicator.long_entries.values & (SUMCON_result > 0.05)
    # short_signals = Master_Indicator.short_entries.values & (SUMCON_result < -0.05)
    data['side'] = np.nan

    long_signals = (SUMCON_result > 0.05)
    short_signals = (SUMCON_result < -0.05)
    data.loc[long_signals, 'side'] = 1
    data.loc[short_signals, 'side'] = -1

    # Remove Look ahead biase by lagging the signal
    data['side'] = data['side'].shift(1)

    # Save the raw data
    raw_data = data.copy()

    # Drop the NaN values from our data set
    data.dropna(axis=0, how='any', inplace=True)

    # Compute daily volatility
    daily_vol = ml.util.get_daily_vol(close=data['close'], lookback=50)

    # Apply Symmetric CUSUM Filter and get timestamps for events
    # Note: Only the CUSUM filter needs a point estimate for volatility
    cusum_events = ml.filters.cusum_filter(data['close'], threshold=daily_vol.mean() * 0.5)

    # Compute vertical barrier
    vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events, close=data['close'], num_days=1)

    pt_sl = [1, 2]
    min_ret = 0.0005
    triple_barrier_events = ml.labeling.get_events(close=data['close'],
                                                   t_events=cusum_events,
                                                   pt_sl=pt_sl,
                                                   target=daily_vol,  # * 0.1,
                                                   min_ret=min_ret,
                                                   num_threads=num_threads,
                                                   vertical_barrier_times=vertical_barriers,
                                                   side_prediction=data['side'])

    labels = ml.labeling.get_bins(triple_barrier_events, data['close'])
    labels.side.value_counts()

    # primary_forecast = pd.DataFrame(labels['bin'])
    # primary_forecast['pred'] = 1
    # primary_forecast.columns = ['actual', 'pred']
    #
    # # Performance Metrics
    # actual = primary_forecast['actual']
    # pred = primary_forecast['pred']
    # print(classification_report(y_true=actual, y_pred=pred))
    #
    # print("Confusion Matrix")
    # print(confusion_matrix(actual, pred))
    #
    # print('')
    # print("Accuracy")
    # print(accuracy_score(actual, pred))

    '''Features'''
    # # Log Returns
    # raw_data['log_ret'] = np.log(raw_data['close']).diff()
    #
    # # Momentum
    # raw_data['mom1'] = raw_data['close'].pct_change(periods=1)
    # raw_data['mom2'] = raw_data['close'].pct_change(periods=2)
    # raw_data['mom3'] = raw_data['close'].pct_change(periods=3)
    # raw_data['mom4'] = raw_data['close'].pct_change(periods=4)
    # raw_data['mom5'] = raw_data['close'].pct_change(periods=5)
    #
    # # Volatility
    # raw_data['volatility_50'] = raw_data['log_ret'].rolling(window=50, min_periods=50, center=False).std()
    # raw_data['volatility_31'] = raw_data['log_ret'].rolling(window=31, min_periods=31, center=False).std()
    # raw_data['volatility_15'] = raw_data['log_ret'].rolling(window=15, min_periods=15, center=False).std()
    #
    # # Get the various log -t returns
    # raw_data['log_t1'] = raw_data['log_ret'].shift(1)
    # raw_data['log_t2'] = raw_data['log_ret'].shift(2)
    # raw_data['log_t3'] = raw_data['log_ret'].shift(3)
    # raw_data['log_t4'] = raw_data['log_ret'].shift(4)
    # raw_data['log_t5'] = raw_data['log_ret'].shift(5)
    #
    # # Serial Correlation
    # window_autocorr = 50
    # autocorr = vbt.pd_acc.rolling_apply(
    #     window_autocorr,
    #     corr_meta_nb,
    #     vbt.Rep('a'),
    #     vbt.Rep('b'),
    #     broadcast_named_args=dict(
    #         a=raw_data['log_ret'],
    #         b=pd.DataFrame({
    #             'b1': raw_data['log_t1'],
    #             'b2': raw_data['log_t2'],
    #             'b3': raw_data['log_t3'],
    #             'b4': raw_data['log_t4'],
    #             'b5': raw_data['log_t5']
    #         })
    #     ),
    #     chunked='ray',
    # )
    #
    # for i in range(1, autocorr.shape[1] + 1):
    #     raw_data[f'autocorr_{i}'] = autocorr[f'b{i}']

    '''...check this step '''  # todo
    # Re compute sides
    raw_data['side'] = 0

    raw_data.loc[long_signals, 'side'] = 1
    raw_data.loc[short_signals, 'side'] = -1

    # Remove look ahead bias
    raw_data = raw_data.shift(1)

    # Get features at event dates
    X = raw_data.loc[labels.index, :]



    # Drop unwanted columns
    # X.drop(['open', 'high', 'low', 'close', 'fast_mavg', 'slow_mavg', ],
    #        axis=1, inplace=True)

    y = labels['bin']


    # return pd.DataFrame({'ret': labels["ret"], 'target': X['side'] * y})
    return (X['side'] * y).astype(int)


def genie_strategy_wrapper(asset_ohlc, threads_per_worker=1):
    # return __genie_strategy__(
    #     # price_series is a pandas.Series object containing the price data for the strategy to be backtested on (e.g. close price)
    #     open=asset_ohlc['open'],  # pandas.Series
    #     high=asset_ohlc['high'],  # pandas.Series
    #     low=asset_ohlc['low'],  # pandas.Series
    #     close=asset_ohlc['close'],  # pandas.Series
    #     num_days=None,  # (int) Number of days to add for vertical barrier.
    #     num_hours=None,  # (int) Number of hours to add for vertical barrier.
    #     num_minutes=None,  # (int) Number of minutes to add for vertical barrier.
    #     num_seconds=None,  # (int) Number of seconds to add for vertical barrier.
    #     # (int) Number of bars (samples) after which to construct vertical barriers (None by default).
    #     num_bars=None,
    #     #
    #
    #     pt_sl=None,
    #     # min_ret is the minimum return value (e.g. 0.004)
    #     min_ret=None,
    #     # threads is the number of threads to be used for the backtest (e.g. cpu_count()-2)
    #     threads=threads_per_worker,
    #     # lookback is the number of days to be used for the volatility calculation (e.g. 50)
    #     lookback=None,
    #     #
    #     volatility_window=None,
    #     threshold=None,
    #     REAL=True,
    #     fake_freq=None,
    # )

    return __genie_strategy__1(
        data=asset_ohlc
    )


if __name__ == '__main__':

    """STEP __N__"""
    import vectorbtpro as vbt

    preped_forex_data = vbt.Data.load('preped_forex_data')

    """Pass Through a Strategy"""

    for asset_ohlc in preped_forex_data.data.keys():
        preped_forex_data.data[asset_ohlc] = preped_forex_data.data[asset_ohlc].dropna()
        preped_forex_data.data[asset_ohlc] = preped_forex_data.data[asset_ohlc].set_index("datetime")

    parametrized_genie_strategy_wrapper = vbt.parameterized(genie_strategy_wrapper,
                                                            # merge_func="concat",
                                                            # n_chunks=np.floor(param_combinations.shape[0]/4).astype(int),
                                                            # n_chunks=np.floor(param_combinations.shape[0]/4).astype(int),
                                                            chunk_len='auto',
                                                            engine='ray',
                                                            show_progress=True,
                                                            init_kwargs={
                                                                # 'address': 'auto',
                                                                'num_cpus': cpu_count() - 2,
                                                                # 'n_chunks':"auto",
                                                                # 'memory': 100 * 10 ** 9,
                                                                # 'object_store_memory': 100 * 10 ** 9,
                                                            },
                                                            )

    assets_list = [preped_forex_data.data[asset_ohlc] for asset_ohlc in preped_forex_data.data.keys()]
    result = parametrized_genie_strategy_wrapper(
        asset_ohlc=vbt.Param(
            assets_list
            , name='symbols'),
        threads_per_worker=np.floor((cpu_count() - 2) / len(assets_list)).astype(int),

    )
    concated_result = pd.concat(result, axis=0)
    print(concated_result.head(10))
    print(concated_result.tail(10))
    #
    # asset_ohlc = preped_forex_data.data["AUDJPY"]
    # asset_ohlc = asset_ohlc.dropna()
    # # asset_ohlc.info()
    # asset_ohlc = asset_ohlc.set_index("datetime")
    # exit()
    # # %%
    # __genie_strategy__(
    #     # price_series is a pandas.Series object containing the price data for the strategy to be backtested on (e.g. close price)
    #     open=asset_ohlc['open'],  # pandas.Series
    #     high=asset_ohlc['high'],  # pandas.Series
    #     low=asset_ohlc['low'],  # pandas.Series
    #     close=asset_ohlc['close'],  # pandas.Series
    #     num_days=None,  # (int) Number of days to add for vertical barrier.
    #     num_hours=None,  # (int) Number of hours to add for vertical barrier.
    #     num_minutes=None,  # (int) Number of minutes to add for vertical barrier.
    #     num_seconds=None,  # (int) Number of seconds to add for vertical barrier.
    #     num_bars=None,  # (int) Number of bars (samples) after which to construct vertical barriers (None by default).
    #     #
    #     pt_sl=None,
    #     # min_ret is the minimum return value (e.g. 0.004)
    #     min_ret=None,
    #     # threads is the number of threads to be used for the backtest (e.g. None)
    #     threads=None,
    #     # lookback is the number of days to be used for the volatility calculation (e.g. 50)
    #     lookback=None,
    #     #
    #     volatility_window=None,
    #     threshold=None,
    #     REAL=True,
    #     fake_freq=None,
    # )

    # %%

    # metrics_df = parametrized_metrics_qs_report(pf_or_pf_path=vbt.Param(
    #     np.random.choice(glob.glob(test_pf_paths), size=1, replace=False, p=None)
    #     , name='pf_path')
    # )
    #
    # # Change index to range index
    # metrics_df = metrics_df.reset_index(drop=False)
    # metrics_df = metrics_df.vbt.sort_index()
    #
    # # Drop pf_path column
    # metrics_df = metrics_df.drop(columns=['pf_path'])
    #
    # metrics_df.to_csv(f"temp_csv/pf_metrics_report_all.csv")


def tempscratchsave():
    #
    #
    #
    # only_n_asset = list(forex_dobj.data.keys())[:10]
    # assets_data_mdf = assets_data_mdf[only_n_asset]
    # print(assets_data_mdf)
    #
    # list=list(assets_data_mdf.keys())[:2]
    # print(list)
    # exit()
    # assets_data_mdf = assets_data_mdf[list]
    #

    # smalled_index = 0
    # largest_index = np.inf

    preprocessed_data_dict = dict()

    for asset in symbols_dict_obj:
        smalled_index = symbols_dict_obj[asset].first_valid_index()
        largest_index = symbols_dict_obj[asset].last_valid_index()

        preprocessed_data_dict[asset] = symbols_dict_obj[asset][smalled_index:largest_index]

        # preprocessed_data_dict[asset] = symbols_dict_obj[asset]
        #
        preprocessed_data_dict[asset].columns = ["id", "date", "time", "open", "high", "low", "close", "volume"]
        #
        # Change Date and Time to datetime format
        date = preprocessed_data_dict[asset]["date"].astype('Int64').astype(str)
        time = preprocessed_data_dict[asset]["time"].astype('Int64').astype(str).str.zfill(6)
        #
        # preprocessed_data_dict[asset]["datetime"] = date.str.cat(time, sep=' ')
        preprocessed_data_dict[asset].loc[:, "datetime"] = date.str.cat(time, sep=' ')
        preprocessed_data_dict[asset].drop(columns=["date", "time"], inplace=True)
        # #
        preprocessed_data_dict[asset]["datetime"] = pd.to_datetime(preprocessed_data_dict[asset]["datetime"],
                                                                   format="%Y%m%d %H%M%S")
        # preprocessed_data_dict[asset] = preprocessed_data_dict[asset][["datetime","id", "open", "high", "low", "close", "volume"]]
        #

        print(preprocessed_data_dict[asset].head())
        print(preprocessed_data_dict[asset].tail())
