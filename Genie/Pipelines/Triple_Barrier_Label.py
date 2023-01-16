import pandas as pd

from Modules.Actors.genie_loader.genie_loader import Genie_Loader
from Modules.Standard_Algorythms import util
from Modules.Standard_Algorythms.labeling_algorythms import labeling
from Modules.Standard_Algorythms.timeseries_algorythms import timeseries_filters


def sample_triple_barrier_strat_labeling(data, num_threads=1, dates=None):
    """
    This function is a sample of how to use the triple barrier labeling algorythm
    Step 1: Create a Genie_Loader object
    Step 2: Use the fetch_data method to get the data
    Step 3: Use the triple_barrier_labeling method to get the labels
    Step 4: Use the timeseries_tilters method to get the filtered data
    Step 5: Use the labeling method to get the labels
    """
    # Import packages
    import numpy as np
    import vectorbtpro as vbt

    if data is None:
        raise ValueError("Data is None")
    if isinstance(data, vbt.Data):
        data = pd.DataFrame(
            {
                "close": data.close.tz_localize(None),
                "open": data.open.tz_localize(None),
                "high": data.high.tz_localize(None),
                "low": data.low.tz_localize(None),
                "volume": data.get("Tick volume").tz_localize(None),
            }
        )

    if dates is not None:
        data = data.loc[dates[0]: dates[1]]

    '''Primary Model'''
    # Compute moving averages
    fast_window = 20
    slow_window = 50

    # STRATEGY!!!!
    fast_mavg = data["close"].rolling(window=fast_window, min_periods=fast_window, center=False).mean()
    slow_mavg = data["close"].rolling(window=slow_window, min_periods=slow_window, center=False).mean()
    # SUMCON_indicator = vbt.IF.from_techcon("SUMCON")
    # indicator_bs = SUMCON_indicator.run(
    #     open=data["open"],
    #     high=data["high"],
    #     low=data["low"],
    #     close=data["close"],
    #     volume=data["volume"],
    #     smooth=30
    # )
    # SUMCON_result = indicator_bs.buy - indicator_bs.sell

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

    long_signals = fast_mavg >= slow_mavg
    short_signals = fast_mavg < slow_mavg
    # long_signals = (SUMCON_result > 0.05)
    # short_signals = (SUMCON_result < -0.05)
    data['side'] = 0
    data.loc[long_signals, 'side'] = 1
    data.loc[short_signals, 'side'] = -1

    # Remove Look ahead biase by lagging the signal
    data['side'] = data['side'].shift(1)

    # Save the raw data
    raw_data = data.copy()

    # Drop the NaN values from our data set
    data.dropna(axis=0, how='any', inplace=True)

    # Compute daily volatility
    daily_vol = util.get_daily_vol(close=data['close'], lookback=50)

    # Apply Symmetric CUSUM Filter and get timestamps for events
    # Note: Only the CUSUM filter needs a point estimate for volatility
    cusum_events = timeseries_tilters.cusum_filter(data['close'], threshold=daily_vol.mean() * 0.5)

    # Compute vertical barrier
    vertical_barriers = labeling.add_vertical_barrier(t_events=cusum_events, close=data['close'], num_days=1)

    pt_sl = [1, 2]
    min_ret = 0.0005
    triple_barrier_events = labeling.get_events(close=data['close'],
                                                t_events=cusum_events,
                                                pt_sl=pt_sl,
                                                target=daily_vol,  # * 0.1,
                                                min_ret=min_ret,
                                                num_threads=num_threads,
                                                vertical_barrier_times=vertical_barriers,
                                                side_prediction=data['side'])

    labels = labeling.get_bins(triple_barrier_events, data['close'])

    raw_data['side'] = 0
    raw_data.loc[long_signals, 'side'] = 1
    raw_data.loc[short_signals, 'side'] = -1


    raw_data.dropna(axis=0, how='any', inplace=True)
    labels.columns = ["ret", "trgt", "bin", "label_side"]

    return pd.concat([raw_data, labels], axis=1).fillna(0)


if __name__ == '__main__':
    import vectorbtpro as vbt

    symbols_data = vbt.Data.load("XAUUSD.pickle")

    resutls = sample_triple_barrier_strat_labeling(symbols_data, num_threads=28, dates=['2019-01-01', '2020-01-01'])

    # display all columns and rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(resutls[['side', 'bin', 'label_side']])
