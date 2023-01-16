import glob

import vectorbtpro as vbt
from Modules.Actors.genie_loader.genie_loader import Genie_Loader
from Modules.Standard_Algorythms import util
from Modules.Standard_Algorythms.labeling_algorythms import labeling
from Modules.Standard_Algorythms.timeseries_algorythms import timeseries_tilters

import numpy as np
import pandas as pd
import pyfolio as pf
import timeit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample, shuffle
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt

RELOAD_DATA = False
TRAINING_DATES = ['2019-01-01', '2021-01-01']
TESTING_DATES = ['2021-05-01', '2022.10.16']
C_RANDOM_STATE = 42
pd.set_option('display.max_columns', None)

# Wo4y3Foc2ExSYyTuMMV_
def perform_grid_search(X_data, y_data):
    """
    Function to perform a grid search.
    """

    rf = RandomForestClassifier(criterion='entropy')

    clf = GridSearchCV(rf, parameters, cv=4, scoring='roc_auc', n_jobs=3)

    clf.fit(X_data, y_data)

    print(clf.cv_results_['mean_test_score'])

    return clf.best_params_['n_estimators'], clf.best_params_['max_depth']
def get_daily_returns(intraday_returns):
    """
    This changes returns into daily returns that will work using pyfolio. Its not perfect...
    """

    cum_rets = ((intraday_returns + 1).cumprod())

    # Downsample to daily
    daily_rets = cum_rets.resample('B').last()

    # Forward fill, Percent Change, Drop NaN
    daily_rets = daily_rets.ffill().pct_change().dropna()

    return daily_rets
'''
Trend Following Strategy
This notebook answers question 3.4 form the text book Advances in Financial Machine Learning.

3.4 Develop a trend-following strategy based on a popular technical analysis statistic (e.g., crossing moving averages). For each observation, the model suggests a side, but not a size of the bet.

(a) Derive meta-labels for ptSl = [1, 2] and t1 where numDays = 1. Use as trgt the daily standard deviation as computed by Snippet 3.1.
(b) Train a random forest to decide whether to trade or not. Note: The decision is whether to trade or not, {0,1}, since the underlying model (the crossing moving average) has decided the side, {-1, 1}.
I took some liberties by extending the features to which I use to build the meta model. I also add some performance metrics at the end.

In conclusion: Meta Labeling works, SMA strategies suck.

Note: To re-run this Notebook one needs a S&P500 Emini futures contracts dataset that is not included in the repositry. It can be purchased from TickData LLC. Please check [our documentation](https://mlfinlab.readthedocs.io/en/latest/getting_started/barriers_to_entry.html) for more details.
'''


# Import MlFinLab tools
# Read in data
if RELOAD_DATA:
    data_file_names = [path.split("/")[-1] for path in
                           glob.glob("/home/ruben/PycharmProjects/mini_Genie_ML/Datas/XAUUSD.csv")]
    data_file_dirs = ["/home/ruben/PycharmProjects/mini_Genie_ML/Datas/"]
    symbols_data = Genie_Loader(
        data_file_names, data_file_dirs,
        # **kwargs
    ).fetch_data(data_file_names=data_file_names,data_file_dirs=data_file_dirs)

    symbols_data.save("XAUUSD")

else:
    symbols_data = vbt.Data.load("XAUUSD.pickle")


# Fit a Primary Model: Trend Following
# Based on the simple moving average cross-over strategy.
# Compute moving averages
fast_window = 20
slow_window = 50
data=pd.DataFrame(
    {
        "close":symbols_data.close.tz_localize(None),
        "open":symbols_data.open.tz_localize(None),
        "high":symbols_data.high.tz_localize(None),
        "low":symbols_data.low.tz_localize(None),
        "volume":symbols_data.get("Tick volume").tz_localize(None),
    }
)
data['fast_mavg'] = data["close"].rolling(window=fast_window, min_periods=fast_window, center=False).mean()
data['slow_mavg'] = data["close"].rolling(window=slow_window, min_periods=slow_window, center=False).mean()
data.head()

# Compute sides
data['side'] = np.nan

long_signals = data['fast_mavg'] >= data['slow_mavg']
short_signals = data['fast_mavg'] < data['slow_mavg']
data.loc[long_signals, 'side'] = 1
data.loc[short_signals, 'side'] = -1

# Remove Look ahead bias by lagging the signal
data['side'] = data['side'].shift(1)
# Save the raw data
raw_data = data.copy()

# Drop the NaN values from our data set
data.dropna(axis=0, how='any', inplace=True)
data['side'].value_counts()

'''
Filter Events: CUSUM Filter
Predict what will happen when a CUSUM event is triggered. Use the signal from the MAvg Strategy to determine the side 
of the bet.
'''
# Compute daily volatility for threshold and triple barrier even targets
daily_vol = util.get_daily_vol(close=data["close"], lookback=50)

# Apply Symmetric CUSUM Filter and get timestamps for events
# Note: Only the CUSUM filter needs a point estimate for volatility
# # Should only use training data to estimate volatility to avoid lookahead bias
cusum_events = timeseries_tilters.cusum_filter(data["close"], threshold=daily_vol.mean()*0.5)

# Compute vertical barrier
vertical_barriers = labeling.add_vertical_barrier(t_events=cusum_events, close=data["close"], num_days=1)
pt_sl = [1, 2]
min_ret = 0.005
triple_barrier_events = labeling.get_events(close=data["close"],
                                               t_events=cusum_events,
                                               pt_sl=pt_sl,
                                               target=daily_vol,
                                               min_ret=min_ret,
                                               num_threads=3,
                                               vertical_barrier_times=vertical_barriers,
                                               side_prediction=data['side'])


labels = labeling.get_bins(triple_barrier_events, data["close"])
print(f'{labels.side.value_counts() = }')

# Results of Primary Model:
# What is the accuracy of predictions from the primary model (i.e., if the sec- ondary model does not filter the bets)?
# What are the precision, recall, and F1-scores?
primary_forecast = pd.DataFrame(labels['bin'])
primary_forecast['pred'] = 1
primary_forecast.columns = ['actual', 'pred']

# Performance Metrics
actual = primary_forecast['actual']
pred = primary_forecast['pred']
print(classification_report(y_true=actual, y_pred=pred))

print("Confusion Matrix")
print(confusion_matrix(actual, pred))

print('')
print("Accuracy")
print(accuracy_score(actual, pred))

'''
A few takeaways
There is an imbalance in the classes - more are classified as "no trade"
Meta-labeling says that there are many false-positives
the sklearn's confusion matrix is [[TN, FP][FN, TP]]
'''

# Fit a Meta Model
# Train a random forest to decide whether to trade or not (i.e 1 or 0 respectively) since the earlier model has
# decided the side (-1 or 1)


# Features
# e.g.
#     Volatility
#     Serial Correlation
#     The returns at the different lags from the serial correlation
#     The sides from the SMavg Strategy
raw_data['log_ret'] = np.log(raw_data['close']).diff()

# Re compute sides
raw_data['side'] = np.nan

long_signals = raw_data['fast_mavg'] >= raw_data['slow_mavg']
short_signals = raw_data['fast_mavg'] < raw_data['slow_mavg']

raw_data.loc[long_signals, 'side'] = 1
raw_data.loc[short_signals, 'side'] = -1
# Remove look ahead bias
raw_data = raw_data.shift(1)

# Now get the data at the specified events
# Get features at event dates
X = raw_data.loc[labels.index, :]

# Drop unwanted columns # todo need a more dynamic way of doing this (easy)
X.drop(['open', 'high', 'low', 'close','volume',
        # 'cum_vol', 'cum_dollar', 'cum_ticks',
        'fast_mavg', 'slow_mavg'],
       axis=1, inplace=True)

y = labels['bin']
print(f'{y.value_counts() = }')

# Balance classes
# Split data into training, validation and test sets
X_training_validation = X[TRAINING_DATES[0]:TRAINING_DATES[1]]
y_training_validation = y[TRAINING_DATES[0]:TRAINING_DATES[1]]
X_train, X_validate, y_train, y_validate = train_test_split(X_training_validation, y_training_validation,
                                                            test_size=0.50, shuffle=False)


train_df = pd.concat([y_train, X_train], axis=1, join='inner')
print(f'{train_df["bin"].value_counts() = }')


# Upsample the training data to have a 50 - 50 split
majority = train_df[train_df['bin'] == 0]
minority = train_df[train_df['bin'] == 1]

new_minority = resample(minority,
                   replace=True,  # Sample with replacement
                   n_samples=majority.shape[0],  # To match majority class
                   random_state=42)

train_df = pd.concat([majority, new_minority])
train_df = shuffle(train_df, random_state=42)

print(f'{train_df["bin"].value_counts() = }')

# Create training data
y_train = train_df['bin']
X_train= train_df.loc[:, train_df.columns != 'bin']
# Fit a model
parameters = {'max_depth':[2, 3, 4, 5, 7],
              'n_estimators':[1, 10, 25, 50, 100, 256, 512],
              'random_state':[42]}


# Extract parameters
n_estimator, depth = perform_grid_search(X_train, y_train)

print(n_estimator, depth, C_RANDOM_STATE)

# Refit a new model with best params, so we can see feature importance
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
#             max_depth=7, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=512, n_jobs=None,
#             oob_score=False, random_state=42, verbose=0, warm_start=False)
rf = RandomForestClassifier(max_depth=depth, n_estimators=n_estimator,
                            criterion='entropy', random_state=C_RANDOM_STATE)

rf.fit(X_train, y_train.values.ravel())


# Training Metrics
# Performance Metrics
y_pred_rf = rf.predict_proba(X_train)[:, 1]
y_pred = rf.predict(X_train)
fpr_rf, tpr_rf, _ = roc_curve(y_train, y_pred_rf)
print(classification_report(y_train, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_train, y_pred))

print('')
print("Accuracy")
print(accuracy_score(y_train, y_pred))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# Meta-label
# Performance Metrics
y_pred_rf = rf.predict_proba(X_validate)[:, 1]
y_pred = rf.predict(X_validate)
fpr_rf, tpr_rf, _ = roc_curve(y_validate, y_pred_rf)
print(classification_report(y_validate, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_validate, y_pred))

print('')
print("Accuracy")
print(accuracy_score(y_validate, y_pred))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

print(X_validate.index.min())
print(X_validate.index.max())

# Primary model
primary_forecast = pd.DataFrame(labels['bin'])
primary_forecast['pred'] = 1
primary_forecast.columns = ['actual', 'pred']

start = primary_forecast.index.get_loc(X_validate.index.min())
end = primary_forecast.index.get_loc(X_validate.index.max()) + 1

subset_prim = primary_forecast[start:end]

# Performance Metrics
actual = subset_prim['actual']
pred = subset_prim['pred']
print(classification_report(y_true=actual, y_pred=pred))

print("Confusion Matrix")
print(confusion_matrix(actual, pred))

print('')
print("Accuracy")
print(accuracy_score(actual, pred))

# Feature Importance
title = 'Feature Importance:'
figsize = (15, 5)

feat_imp = pd.DataFrame({'Importance':rf.feature_importances_})
feat_imp['feature'] = X.columns
feat_imp.sort_values(by='Importance', ascending=False, inplace=True)
feat_imp = feat_imp

feat_imp.sort_values(by='Importance', inplace=True)
feat_imp = feat_imp.set_index('feature', drop=True)
feat_imp.plot.barh(title=title, figsize=figsize)
plt.xlabel('Feature Importance Score')
plt.show()

# Performance Tear Sheets (In-sample)
# Without Meta Labeling
valid_dates = X_validate.index
base_rets = labels.loc[valid_dates, 'ret']
primary_model_rets = get_daily_returns(base_rets)
# Set-up the function to extract the KPIs from pyfolio
perf_func = pf.timeseries.perf_stats
# Save the statistics in a dataframe
perf_stats_all = perf_func(returns=primary_model_rets,
                           factor_returns=None,
                           positions=None,
                           transactions=None,
                           turnover_denom="AGB")
perf_stats_df = pd.DataFrame(data=perf_stats_all, columns=['Primary Model'])

# pf.show_perf_stats(primary_model_rets)

meta_returns = labels.loc[valid_dates, 'ret'] * y_pred
daily_meta_rets = get_daily_returns(meta_returns)
# Save the KPIs in a dataframe
perf_stats_all = perf_func(returns=daily_meta_rets,
                           factor_returns=None,
                           positions=None,
                           transactions=None,
                           turnover_denom="AGB")

perf_stats_df['Meta Model'] = perf_stats_all

# pf.show_perf_stats(daily_meta_rets)

# Extarct data for out-of-sample (OOS)
X_oos = X[TESTING_DATES[0]:TESTING_DATES[1]]
y_oos = y[TESTING_DATES[0]:TESTING_DATES[1]]


print(X_oos)
# Performance Metrics
y_pred_rf = rf.predict_proba(X_oos)[:, 1]
y_pred = rf.predict(X_oos)
fpr_rf, tpr_rf, _ = roc_curve(y_oos, y_pred_rf)
print(classification_report(y_oos, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_oos, y_pred))

print('')
print("Accuracy")
print(accuracy_score(y_oos, y_pred))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# Primary model
primary_forecast = pd.DataFrame(labels['bin'])
primary_forecast['pred'] = 1
primary_forecast.columns = ['actual', 'pred']

subset_prim = primary_forecast[TESTING_DATES[0]:TESTING_DATES[1]]

# Performance Metrics
actual = subset_prim['actual']
pred = subset_prim['pred']
print(classification_report(y_true=actual, y_pred=pred))

print("Confusion Matrix")
print(confusion_matrix(actual, pred))

print('')
print("Accuracy")
print(accuracy_score(actual, pred))

# Primary Model (Test Data)
test_dates = X_oos.index

# Downsample to daily
prim_rets_test = labels.loc[test_dates, 'ret']
daily_rets_prim = get_daily_returns(prim_rets_test)

# Save the statistics in a dataframe
perf_stats_all = perf_func(returns=daily_rets_prim,
                           factor_returns=None,
                           positions=None,
                           transactions=None,
                           turnover_denom="AGB")

perf_stats_df['Primary Model OOS'] = perf_stats_all

# pf.create_returns_tear_sheet(labels.loc[test_dates, 'ret'], benchmark_rets=None)
# pf.show_perf_stats(daily_rets_prim)

# Meta Model (Test Data)
meta_returns = labels.loc[test_dates, 'ret'] * y_pred

daily_rets_meta = get_daily_returns(meta_returns)

# Save the KPIs in a dataframe
perf_stats_all = perf_func(returns=daily_rets_meta,
                           factor_returns=None,
                           positions=None,
                           transactions=None,
                           turnover_denom="AGB")

perf_stats_df['Meta Model OOS'] = perf_stats_all

print(f'{perf_stats_df = }')
print(f'{daily_rets_meta = }')
print(f'{daily_rets_prim = }')

# pf.create_returns_tear_sheet(daily_rets_meta, benchmark_rets=None)


