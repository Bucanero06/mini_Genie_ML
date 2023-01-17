import numpy as np
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
    elif isinstance(data, pd.DataFrame):
        #make sure the data has the correct columns
        try:
            data = data[["close", "open", "high", "low", "volume"]]
        except:
            raise ValueError("Data is missing columns")

    else:
        raise ValueError("Data must be a DataFrame or vbt.Data")

    if dates is not None:
        data = data.loc[dates[0]: dates[1]]

    '''Primary Model'''
    # Compute moving averages
    fast_window = 20
    slow_window = 50

    # STRATEGY!!!!
    # fast_mavg = data["close"].rolling(window=fast_window, min_periods=fast_window, center=False).mean()
    # slow_mavg = data["close"].rolling(window=slow_window, min_periods=slow_window, center=False).mean()
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

    '''Compile Structure and Run Master Indicator'''
    # Compute sides
    # long_signals = Master_Indicator.long_entries.values & (SUMCON_result > 0.05)
    # short_signals = Master_Indicator.short_entries.values & (SUMCON_result < -0.05)
    data['side'] = np.nan

    # long_signals = fast_mavg >= slow_mavg
    # short_signals = fast_mavg < slow_mavg
    long_signals = (SUMCON_result > 0.05)
    short_signals = (SUMCON_result < -0.05)
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
    cusum_events = timeseries_filters.cusum_filter(data['close'], threshold=daily_vol.mean() * 0.5)

    # Compute vertical barrier
    vertical_barriers = labeling.add_vertical_barrier(t_events=cusum_events, close=data['close'], num_days=1)

    pt_sl = [0.5, 1]
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
    print(labels.head(100))
    raw_data.dropna(axis=0, how='any', inplace=True)
    # Change raw_data side name to direction
    raw_data.rename(columns={'side': 'direction'}, inplace=True)

    labels.columns = ["ret", "trgt", "bin", "label_side"]

    return pd.concat([raw_data, labels], axis=1).fillna(0)


def true_binary_label(y_pred, y_test):
    bin_label = np.zeros_like(y_pred)
    for i in range(y_pred.shape[0]):
        if y_pred[i] != 0 and y_pred[i] * y_test[i] > 0:
            bin_label[i] = 1  # true positive
    return bin_label


def live_execution_of_models(open, low, high, close, primary_model, secondary_model):
    X = np.array([open, low, high, close])
    y_pred = primary_model.predict(X)
    X_meta = np.hstack([y_pred[:, None], X])
    y_pred_meta = secondary_model.predict(X_meta)
    return y_pred * y_pred_meta


if __name__ == '__main__':
    import vectorbtpro as vbt

    symbols_data = vbt.Data.load("XAUUSD.pickle")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    retults = sample_triple_barrier_strat_labeling(symbols_data, num_threads=28,
                                                   dates=['2019-01-01', '2019-05-01']
                                                   )

    # display all columns and rows

    original_length = len(retults)
    train_length = int(original_length * 0.8)

    from imblearn.over_sampling import SMOTE

    X = retults[['open', 'close', 'high', 'low']].values
    bins = np.squeeze(retults[['bin']].values)
    y = np.squeeze(retults[['label_side']].values) * bins

    X_train, y_train = X[:train_length], y[:train_length]
    X_test, y_test = X[train_length:], y[train_length:]
    bins_train, bins_test = bins[:train_length], bins[train_length:]


    sm = SMOTE()
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    # First Model
    clf = LogisticRegression().fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)  # Predicted labels for test data to be used in meta labeling
    cm = confusion_matrix(true_binary_label(y_pred, y_test), y_pred != 0)
    print(cm)

    # Prep for meta labeling training
    # generate predictions for training set
    y_train_pred = clf.predict(X_train)
    # add the predictions to features
    X_train_meta = np.hstack([y_train_pred[:, None], X_train])
    X_test_meta = np.hstack([y_pred[:, None], X_test])

    # Meta Model Training
    # generate true meta-labels
    y_train_meta = true_binary_label(y_train_pred, y_train)
    # rebalance classes again
    sm = SMOTE()
    X_train_meta_res, y_train_meta_res = sm.fit_resample(X_train_meta, y_train_meta)
    model_secondary = LogisticRegression().fit(X_train_meta_res, y_train_meta_res)
    #
    # Meta Model Testing
    y_pred_meta = model_secondary.predict(X_test_meta)
    # use meta-predictions to filter primary predictions
    cm = confusion_matrix(true_binary_label(y_pred, y_test), (y_pred * y_pred_meta) != 0)
    print(cm)

    # for i in zip(retults['open'].values, retults['low'].values, retults['high'].values, retults['close'].values):
    #     print(live_execution_of_models(i[0], i[1], i[2], i[3], clf, model_secondary))
