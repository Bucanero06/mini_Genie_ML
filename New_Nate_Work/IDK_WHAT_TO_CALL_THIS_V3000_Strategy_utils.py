import vectorbtpro as vbt
import pandas as pd
import numpy as np
from logger_tt import logger

from Modules.Actors_Old.Utils import convert_to_seconds, comb_price_and_range_index


def cache_func(low, high, close,
               # datetime_index,
               # Trend_filter_1_timeframes, Trend_filter_atr_windows, Trend_filter_1_data_lookback_windows,
               rsi_timeframes, rsi_windows,
               sma_on_rsi_1_windows, sma_on_rsi_2_windows, sma_on_rsi_3_windows,
               PEAK_and_ATR_timeframes, atr_windows, data_lookback_windows,
               EMAs_timeframes, ema_1_windows, ema_2_windows,
               take_profit_points, stop_loss_points):
    """
    Cache function for MMT_RLGL strategy
    """

    cache = {
        # Data
        'Low': {},
        'High': {},
        'Close': {},
        # Resampler
        'Resamplers': {},
        # empty_df_like
        'Empty_df_like': pd.DataFrame().reindex_like(close),
    }

    # Create a set of all timeframes to resample data to
    timeframes = tuple(set(
        tuple(PEAK_and_ATR_timeframes) + tuple(EMAs_timeframes) + tuple(rsi_timeframes)
    ))
    #
    '''Pre-Resample Data'''
    #
    for timeframe in timeframes:
        # cache['Low'][timeframe] = resample_split_data(low,  timeframe=timeframe)
        # cache['High'][timeframe] = resample_split_data(high,  timeframe=timeframe)
        # cache['Close'][timeframe] = resample_split_data(close,  timeframe=timeframe)
        #
        # LOW
        cache['Low'][timeframe] = low.vbt.resample_apply(timeframe,
                                                         vbt.nb.min_reduce_nb).dropna() if timeframe != '1 min' else low
        # HIGH
        cache['High'][timeframe] = high.vbt.resample_apply(timeframe,
                                                           vbt.nb.max_reduce_nb).dropna() if timeframe != '1 min' else high
        # CLOSE
        cache['Close'][timeframe] = close.vbt.resample_apply(timeframe,
                                                             vbt.nb.last_reduce_nb).dropna() if timeframe != '1 min' else close

        '''Pre-Prepare Resampler'''
        cache['Resamplers'][timeframe] = vbt.Resampler(
            cache['Close'][timeframe].index,
            close.index,
            source_freq=timeframe,
            target_freq="1m") if timeframe != '1 min' else None
    return cache


def mmt_rlgl_post_cartesian_product_filter_function(parameters_record, **kwargs):
    '''RLGL'''
    logger.info(f'{parameters_record.shape = }')
    parameters_record = parameters_record[
        np.where(np.less_equal([convert_to_seconds(i) for i in parameters_record["EMAs_timeframes"]],
                               [convert_to_seconds(i) for i in parameters_record["rsi_timeframes"]]))[0]]


    logger.info(f'2{parameters_record.shape = }')
    parameters_record = parameters_record[
        np.where(parameters_record["sma_on_rsi_1_windows"] <= parameters_record["sma_on_rsi_2_windows"])[
            0]]

    logger.info(f'3{parameters_record.shape = }')
    parameters_record = parameters_record[
        np.where(parameters_record["sma_on_rsi_2_windows"] <= parameters_record["sma_on_rsi_3_windows"])[
            0]]

    '''MMT'''
    parameters_record = parameters_record[
        np.where(parameters_record["ema_1_windows"] <= parameters_record["ema_2_windows"])[0]]
    #
    # parameters_record = parameters_record[
    #     np.where(parameters_record["Trend_filter_1_timeframes"] != parameters_record[
    #         "PEAK_and_ATR_timeframes"])[0]]

    return parameters_record


def comb_split_price_n_datetime_index(price_cols, datetime_cols, n_splits):

    if np.ndim(price_cols) == 2:
        new_split_data = [int(0) for x in range(n_splits)]
        for j in range(n_splits):
            new_split_data[j] = comb_price_and_range_index(price_cols[j], datetime_cols[j])
        return new_split_data
    else:
        logger.exception("split price_cols is not 2D array")


def resample_split_data(split_data, timeframe):
    n_splits = len(split_data)
    print(split_data)
    exit()
    new_split_data = [int(0) for x in range(n_splits)]
    for j in range(n_splits):
        # print(split_data[j])

        new_split_data[j] = split_data[j].vbt.resample_apply(timeframe,
                                                             vbt.nb.min_reduce_nb).dropna() if timeframe != '1 min' else \
            split_data[j]

    return new_split_data
