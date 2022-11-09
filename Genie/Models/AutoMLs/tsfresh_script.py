#
# exit()
#
from Modules.data_structures import get_dollar_bars

import pandas as pd
df = pd.read_csv("../../Modules/New_Nate_Work/Datas/XAUUSD.csv")

# df = get_dollar_bars(file_path_or_df=df, threshold=70000,
#                      batch_size=20000000, verbose=True, to_csv=True,
#                      output_path='../../../Datas/XAUUSD_dollar_bars.csv')








exit()
#
# print(df)
import pandas as pd
from tsfresh import extract_relevant_features
from tsfresh.examples import download_robot_execution_failures, load_robot_execution_failures

# df = pd.read_csv("../../Datas/XAUUSD_dollar_bars.csv")
# df = df.drop(columns=["date_time"])
# df = df.replace(to_replace='nan', value=np.nan)
# df = df.dropna()
# y = df.pop["close"]
#
# print(df)

df = pd.read_csv(
    "Datas/XAUUSD_dollar_bars.csv")
df["id"] = df.index
print(df.columns)
df = df.dropna(how='all')
y = df.pop("close")
df.pop("date_time")

from tsfresh.feature_extraction import ComprehensiveFCParameters
#
# # download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()

print(timeseries.head())
print(y.head())

exit()
#
# print(f"timeseries: {timeseries}")
# print(f"df: {df}")
#
# #print types
# print(f"timeseries type: {type(timeseries)}")
# print(f"df type: {type(df)}")
#
# exit()
# print(timeseries)

# timeseries = df.dropna()

extraction_settings = ComprehensiveFCParameters()

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification


# timeseries, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
# timeseries=pd.DataFrame(timeseries)
# timeseries["id"] = timeseries.index
# print(timeseries.head())
# y=pd.Series(y)
X_filtered_2 = extract_relevant_features(timeseries, y, column_id='id',
                                         # column_sort='date_time',
                                         default_fc_parameters=extraction_settings)
import vectorbtpro as vbt


# df= pd.concat([df, y], axis=1)
# df = pd.concat([df, X_filtered_2], axis=1)
# df = pd.concat([y, X_filtered_2], axis=1)
# df = df.dropna()
# set index
features = X_filtered_2
# df = df.drop(columns=['date_time'])
features.vbt.plot().show()

