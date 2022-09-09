# Import packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Import MlFinLab tools
from _import_mlfinlab import *
from mlfinlab.ensemble.sb_bagging import SequentiallyBootstrappedBaggingClassifier

# Load data
X = pd.read_csv('X_FILE_PATH', index_col=0, parse_dates = [0])
y = pd.read_csv('y_FILE_PATH', index_col=0, parse_dates = [0])
triple_barrier_events = pd.read_csv('BARRIER_FILE_PATH', index_col=0, parse_dates = [0, 2])
price_bars = pd.read_csv('PRICE_BARS_FILE_PATH', index_col=0, parse_dates = [0, 2])

triple_barrier_events = triple_barrier_events.loc[X.index, :] # Take only train part
triple_barrier_events = triple_barrier_events[(triple_barrier_events.index >= X.index.min()) &
                                              (triple_barrier_events.index <= X.index.max())]

# Use tools
base_est = RandomForestClassifier(n_estimators=1, criterion='entropy', bootstrap=False,
                                  class_weight='balanced_subsample')
clf = SequentiallyBootstrappedBaggingClassifier(base_estimator=base_est,
                                                samples_info_sets=triple_barrier_events.t1,
                                                price_bars=price_bars, oob_score=True)
clf.fit(X, y)