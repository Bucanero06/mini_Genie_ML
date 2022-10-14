from os import cpu_count, environ

import numpy as np
import pandas as pd

from Modules.Utils import dict_to_namedtuple

environ['MLFINLAB_API_KEY'] = "0800b4ea410a702acddefdec86f93523"
import mlfinlab as ml


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
        price_series, # pandas.Series
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
        min_ret = 0.004
    if threads is None:
        threads = cpu_count() - 1
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

    # Get_volatility
    if not REAL:
        if fake_freq is None:
            fake_freq = "1D"

        start = '1970-01-01 00:00:00.001'
        end = pd.to_datetime(start) + pd.Timedelta(seconds=len(price_series) - 1)
        price_series.index = smart_datetimeindex(start=start, end=end, freq=fake_freq)
        #
        volatility = price_series.rolling(volatility_window).var()
        vol_mean = volatility.mean()
    else:
        volatility = ml.util.get_daily_vol(close=price_series, lookback=volatility_window)
        vol_mean = volatility.mean()

    """
    Suppose we use a mean-reverting strategy as our primary model, giving each observation a label of -1 or 1. 
    We can then use meta-labeling to act as a filter for the bets of our primary model.

    Assuming we have a pandas series with the timestamps of our observations and their respective labels given by the 
    primary model, the process to generate meta-labels goes as follows.
    """

    # Apply Symmetric CUSUM Filter and get timestamps for events
    # Note: Only the CUSUM filter needs a point estimate for volatility
    cusum_events = ml.filters.cusum_filter(price_series,
                                           threshold=vol_mean if not threshold else threshold)

    # Compute vertical barrier using timedelta or number of bars (num_bars) after event time (t1)
    #     Compute vertical barrier using timedelta after a fixed number of samples (ticks) have passed
    #     since the event time (t1) has been reached (t1 + num_bars)
    vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events,
                                                         close=price_series,
                                                         num_days=num_days,
                                                         num_hours=num_hours,
                                                         num_minutes=num_minutes,
                                                         num_seconds=num_seconds,
                                                         num_bars=volatility_window if not num_bars else num_bars
                                                         )

    print(f"cusum_events {len(cusum_events)}")
    print(f"volatility {len(volatility)}")
    print(f"price_series {len(price_series)}")
    print(f"vertical_barriers {len(vertical_barriers)}")

    exit()

    """
    Once we have computed the daily volatility along with our vertical time barriers and have downsampled our series 
    using the CUSUM filter, we can use the triple-barrier method to compute our meta-labels by passing in the side 
    predicted by the primary model.
    """

    # Get events and labels for each event using the triple barrier method
    triple_barrier_events = ml.labeling.get_events(close=price_series,
                                                   t_events=cusum_events,
                                                   pt_sl=pt_sl,
                                                   target=volatility,
                                                   min_ret=min_ret,
                                                   num_threads=threads,
                                                   vertical_barrier_times=vertical_barriers,
                                                   )

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

    labels_one = ml.labeling.get_bins(triple_barrier_events, price_series)

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

    full_df = pd.DataFrame(price_series.loc[labels_one['bin'].index], index=labels_one['bin'].index)
    full_df = pd.concat([
        full_df,
        triple_barrier_events[['pt', 'sl']].loc[labels_one.index],
        labels_one[['ret', 'trgt', 'bin']].loc[labels_one.index]
    ], axis=1)

    # print(f"price_series: \n\n{price_series}\n\n")
    # print(f"triple_barrier_events: \n\n{triple_barrier_events}\n\n")
    # print(f"labels_one: \n\n{labels_one}\n\n")

    return dict_to_namedtuple(dict(
        df=full_df,
        # daily_vol=daily_vol, cusum_events=cusum_events,
        # vertical_barriers=vertical_barriers,
        # triple_barrier_events=triple_barrier_events, labels_one=labels_one
    ), add_prefix=None)
