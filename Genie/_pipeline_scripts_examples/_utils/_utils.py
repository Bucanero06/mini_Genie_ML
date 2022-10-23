import cudf
import numpy as np
# from dask import DataFrame as dd
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, GaussianNoise, Dropout, Dense


def Load_n_Prep_Data(file_name):
    # Prep Dta
    # data = pd.read_csv(FILE_NAME)
    data = cudf.read_csv(file_name)  # .to_pandas()

    data.columns = ["date", "open", "high", "low", "close", "volume"]
    data = data.set_index("date")
    # data = data.iloc[:10000]
    #
    # Compute Features
    data["returns"] = np.log(data["close"] / data["close"].shift(1))

    # Cudf capable metrics
    data["volatility"] = data["returns"].rolling(window=252, min_periods=252).std() * np.sqrt(252)
    data['volatility'] = data['returns'].rolling(window=252).std() * np.sqrt(252)
    data['mean'] = data['returns'].rolling(window=252).mean()
    data['std'] = data['returns'].rolling(window=252).std()
    data['var'] = data['returns'].rolling(window=252).var()
    data["sma"] = data["close"].rolling(window=20).mean()
    data["bollinger"] = (data["close"] - data["sma"]) / (2 * data["volatility"])
    data["momentum"] = data["close"] - data["close"].shift(20)
    data['close_pct_delta'] = data['close'].pct_change()

    # Convert cudf to pandas
    data = data.to_pandas()

    # Run some basic preprocessing without cudf
    data["direction"] = np.where(data["returns"] > 0, 1, 0)
    data['skewness'] = data['returns'].rolling(window=252).skew()
    data['kurtosis'] = data['returns'].rolling(window=252).kurt()
    data['median'] = data['returns'].rolling(window=252).median()
    # data['mode'] = data['returns'].rolling(window=252).apply(lambda x: x.mode()[0])

    # Compute mode of returns

    # Drop asset_prices
    data = data.drop(["low", "high", "close"], axis=1)
    # data = data.drop(["open", "low", "high", "close"], axis=1)

    # Clean Data
    data = data.dropna()
    print(f'Loaded {len(data)} rows from {file_name}')
    return data


def dataframe_to_structured_array(dataframe):
    """dataframe to structured record array (specially useful for when working with numba and you still want the easy access to the data)"""
    dtypes = [tuple(x) for x in dataframe.dtypes.iteritems()]
    dtypes = [(name, np.dtype(dtype)) for name, dtype in dtypes]
    return np.rec.fromrecords(dataframe.values, dtype=dtypes)


def Split_Bars_into_X_y(data, n_train_bars, n_predict_bars, shufle_bool=False):
    # Create Dataset
    X, y = [], []
    in_start = 0
    # step over the entire history one time step at a time
    data_len = len(data)
    for i in range(data_len):
        # define the end of the input sequence
        in_end = in_start + n_train_bars
        out_end = in_end + n_predict_bars
        # ensure we have enough data for this instance
        if out_end < data_len:
            data['open'][in_start:in_end] = data['open'][in_end]
            #
            _X = data.iloc[in_start:in_end]
            _y = data.iloc[in_end:out_end, 0]  # predicts opening price
            #
            X.append(_X.values)
            y.append(_y.values)
            #
        # move along one time step
        in_start += 1

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Shuffle
    if shufle_bool:
        c = list(zip(X, y))
        np.random.shuffle(c)
        X, y = zip(*c)
        X = np.array(X)
        y = np.array(y)

    # # Reshape
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    y = np.reshape(y, (y.shape[0], y.shape[1]))
    return X, y


def Load_model(_x_test, _y_test, input_shape, output_shape, model_name):
    # Create a basic model instance
    model = Create_Model(_x_train=_x_test, _y_train=_y_test, input_shape=input_shape, output_shape=output_shape)
    #
    # Evaluate the model
    loss, acc = model.evaluate(_x_test, _y_test, verbose=0)
    print(f"Untrained model, mse: {acc}")

    # # Load model
    model = tf.keras.models.load_model(f"models/{model_name}.h5")

    # Re-evaluate the model
    loss, acc = model.evaluate(_x_test, _y_test, verbose=0)
    print(f"Restored model, mse: {acc}")
    return model


def Create_Model(_x_train, _y_train, input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(GaussianNoise(0.1))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(GaussianNoise(0.1))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(output_shape))
    # Compile Model
    # model.compile(optimizer="rmsprop", loss="mae", metrics=["mse"])
    def custom_loss_function(y_true, y_pred):
        print(f'y_true: {y_true}')
        print(f'y_pred: {y_pred}')


        # Compute rolling metrics on y_true and y_pred


        # Compute vector for y_true metrics and y_pred metrics


        # Compute the loss based on the dot product of the two vectors


        # Return the loss
        return loss

    model.compile(optimizer='adam', loss=custom_loss_function)

    return model
















