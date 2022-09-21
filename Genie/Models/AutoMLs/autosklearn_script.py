import autosklearn.classification
# from autosklearn.estimators import AutoSklearnClassifier

import sklearn
import numpy as np
import pandas as pd

df = pd.read_csv("../../../Datas/XAUUSD_dollar_bars.csv")
df = df.drop(columns=["date_time"])
df = df.replace(to_replace='nan', value=np.nan)
df = df.dropna()
_y = df["close"]
_X = df.drop(columns=["close"])

y_max = _y.idxmax(axis=0)

import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

# X_train, X_test, y_train, y_test = \
# sklearn.model_selection.train_test_split(_X, _y, random_state=1)
# automl = autosklearn.classification.AutoSklearnClassifier()
# print(X_train.shape)
# print(y_train.shape)
# # exit()
# automl.fit(X_train, y_train)
# y_hat = automl.predict(X_test)
# print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

#
# for train, test in sklearn.model_selection.KFold(n_splits=2).split(_X):
#     X_train, X_test = _X.iloc[train], _X.iloc[test]
#     y_train, y_test = _y.iloc[train], _y.iloc[test]
#     automl = autosklearn.Ti()
#     automl.fit(X_train, y_train)
#     y_hat = automl.predict(X_test)
#     print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))


import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification


def main():
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=5,
        # tmp_folder='/tmp/autosklearn_holdout_example_tmp',
        disable_evaluator_output=False,
        # 'holdout' with 'train_size'=0.67 is the default argument setting
        # for AutoSklearnClassifier. It is explicitly specified in this example
        # for demonstrational purpose.
        # resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67},
    )
    automl.fit(X_train, y_train, dataset_name='digits')

    # Iterate all models used in the final ensemble
    for weight, model in automl.get_models_with_weights():
        # Obtain the step of the underlying scikit-learn pipeline
        print(model.steps[-2])
        # Obtain the scores of the current feature selector
        print(model.steps[-2][-1].choice.preprocessor.scores_)
        # Obtain the percentile configured by Auto-sklearn
        print(model.steps[-2][-1].choice.preprocessor.percentile)


if __name__ == '__main__':
    main()
