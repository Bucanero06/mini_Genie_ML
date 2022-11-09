import h2o
import pandas as pd
from h2o.automl import H2OAutoML
forex_data=pd.read_csv('mmt_triplebarrier_h2o_test_dataset.csv.csv')
h2o.init()
h2o.remove_all()
# Identify predictors and response
x = list(forex_data.columns)
y = "EURUSD_target"
x.remove(y)

# pandas dataframe to h2o dataframe
h2o_forex_data = h2o.H2OFrame(forex_data)

# Split data into train and test sets,and  validation
train, valid, test = h2o_forex_data.split_frame([0.8, 0.1], seed=1234)



# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

exit()
"""STEP __N__"""
import glob

import numpy as np
import pandas as pd
# # Duka allows for download multiple assets at once however due to the limit imposed by the data source it tends
# # to cause perfoimary issues
# # ! duka EURUSD  -s 2020-09-29  --header && duka USDJPY -s 2020-09-29  --header &&
# #       duka GBPUSD  -s 2020-09-29  --header && duka AUDUSD  -s 2020-09-29  --header &&
# #       duka USDCAD -s 2020-09-29  --header && duka USDCNY  -s 2020-09-29  --header &&
# #       duka USDCHF  -s 2020-09-29  --header && duka EURGBP  -s 2020-09-29  --header &&
# #       duka USDKRW  -s 2020-09-29  --header
#
# full_paths=glob.glob("/home/ruben/PycharmProjects/mini_Genie_ML/Datas/Forex_Tick_Data/*.csv")
# file_names = [path.split("/")[-1] for path in full_paths]
# print(file_names)
# # data = Data_Manager().fetch_csv_data_dask(data_file_name="Backtest_Data.csv",
# #                                           search_in=[".", "Datas", "Datas/Sample-Data", "Datas/USDJPY_Tick_Data"])
# data = Data_Manager().fetch_data(data_file_names=file_names,
#                                data_file_dirs=["/home/ruben/PycharmProjects/mini_Genie_ML/Datas/Forex_Tick_Data/"],
#                                )
#
#
# print(data.data.keys())
# data.save("forex_tick_data")
#
# preped_forex_data = vbt.Data.load('preped_forex_data')
#
# forex_data_dict = dict()
# for asset_ohlc in tuple(preped_forex_data.data.keys())[:1]:
#     forex_data_dict[asset_ohlc] = preped_forex_data.data[asset_ohlc].dropna()
#     forex_data_dict[asset_ohlc] = forex_data_dict[asset_ohlc][:100000000]
#     # forex_data_dict[asset_ohlc] = forex_data_dict[asset_ohlc].set_index("datetime")
#     # forex_data_dict[asset_ohlc] = forex_data_dict[asset_ohlc].drop(columns=["id"])
#
import vectorbtpro as vbt
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame  # noqa

from Modules._Data_Manager import Data_Manager

# preped_forex_data = pd.read_pickle('forex_features.pickle')
file_paths = glob.glob("/home/ruben/PycharmProjects/mini_Genie_ML/Datas/Forex_Tick_Data/*_dollar_run_bars.pickle")[:2]
preped_forex_data = dict()
for path in file_paths:
    preped_forex_data[path.split("/")[-1][:6]] = pd.read_pickle(path)
    # print(preped_forex_data[path.split("/")[-1][:6]].info())

# forex_data_dict = mltidx_df_to_dict(preped_forex_data, first_n=100000000)

"""Pass Through a Strategy"""
from multiprocessing import cpu_count
from __utils import genie_strategy_wrapper

# %%
# assets_data_list = [preped_forex_data[asset_ohlc] for asset_ohlc in preped_forex_data.keys()]
#
# genie_strategy_wrapper(asset_ohlc=assets_data_list[0].set_index("date_time"),
#                        threads_per_worker=np.floor((cpu_count() - 2)).astype(int))
# exit()

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

assets_data_list = [preped_forex_data[asset_ohlc].set_index('date_time') for asset_ohlc in preped_forex_data.keys()]
result = parametrized_genie_strategy_wrapper(
    asset_ohlc=vbt.Param(
        assets_data_list
        # , name='symbols'
    ),
    threads_per_worker=np.floor((cpu_count() - 2) / len(assets_data_list)).astype(int),

)
# pd.set_option('display.max_columns', None)

first_index = preped_forex_data[list(preped_forex_data.keys())[0]].index[0]
last_index = preped_forex_data[list(preped_forex_data.keys())[0]].index[-1]


for index, asset_ohlc in enumerate(preped_forex_data.keys()):
    preped_forex_data[asset_ohlc]["target"] = result[index]
    # print(f'11{preped_forex_data[asset_ohlc] = }')

    preped_forex_data[asset_ohlc] = preped_forex_data[asset_ohlc].fillna(method="ffill")
    # first_index = preped_forex_data[asset_ohlc].index[0] if first_index > preped_forex_data[asset_ohlc].index[
    #     0] else first_index
    # last_index = preped_forex_data[asset_ohlc].index[-1] if last_index < preped_forex_data[asset_ohlc].index[
    #     -1] else last_index
    print(f'{preped_forex_data[asset_ohlc] = }')



# x, y = make_forecasting_frame(forex_data_dict[asset_ohlc]["close"], kind="price", max_timeshift=2, rolling_direction=1)
#
# # Replace id column with (asset name, date) tuple
# x["id"] = x["id"].apply(lambda x: (asset_ohlc, x[1]))
# #
# y = y.reset_index()
# y["id"] = y["index"].apply(lambda x: (asset_ohlc, x[1]))
# y = y.set_index("id", drop=True)
# print(x)


# %%
"""Compute alphas"""
# wqa_indicator = vbt.wqa101(53)
# value = wqa_indicator.run(open=asset_ohlc['open'], high=asset_ohlc['high'], low=asset_ohlc['low'],
#                           close=asset_ohlc['close'], volume=asset_ohlc['volume'])
# value.out
# %%

"""Prep for TSFRESH"""
forex_concated = pd.concat(forex_data_dict, axis=0)
flat_forex_dataframe = forex_concated.reset_index()
flat_forex_dataframe = flat_forex_dataframe.drop(columns=["level_0", "level_1"])
flat_forex_dataframe.to_pickle('flat_forex_dataframe.pickle')
# %%

"""TSFRESH Processes"""
import pandas as pd
from tsfresh.feature_extraction import extract_features

tsfresh_ready_data = pd.read_pickle("flat_forex_dataframe.pickle")
tsfresh_ready_data = tsfresh_ready_data.dropna()

# %%
from tsfresh.utilities.dataframe_functions import roll_time_series

tsfresh_ready_data_rolled = roll_time_series(tsfresh_ready_data, column_id="id", column_sort="datetime",
                                             max_timeshift=200, min_timeshift=200, rolling_direction=1)
# %%

# %%
from tsfresh.utilities.distribution import ClusterDaskDistributor, LocalDaskDistributor  # noqa: F401
from tsfresh.utilities.distribution import MultiprocessingDistributor  # noqa: F401
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, EfficientFCParameters  # noqa: F401

settings = EfficientFCParameters()
# print(settings)
# We construct a Distributor that will spawn the calculations
# over four threads on the local machine
# Distributor = MultiprocessingDistributor(n_workers=1,
#                                          disable_progressbar=False,
#                                          progressbar_title="Feature Extraction")

# Distributor = LocalDaskDistributor(n_workers=10)
# X_tsfresh contains the extracted tsfresh features
# just to pass the Distributor object to
# the feature extraction, along with the other parameters
X_tsfresh = extract_features(
    timeseries_container=tsfresh_ready_data_rolled,
    column_id='id',
    column_sort='datetime',
    # distributor=Distributor,
    n_jobs=1,
    default_fc_parameters=settings
)
# %%
print(X_tsfresh)
# %%
# which are now filtered to only contain relevant features
X_tsfresh_filtered = select_features(X_tsfresh, y)

# # we can easily construct the corresponding settings object
# kind_to_fc_parameters = tsf.feature_extraction.settings.from_columns(X_tsfresh_filtered)
# %%


exit()

# %%
"""Alternative Data ... """

AD_dobj = Data_Manager().fetch_data(
    data_file_names="AD_df.csv",
    data_file_dirs=
    ["/home/ruben/PycharmProjects/mini_Genie_ML/Datas/Forex_Tick_Data", ".",
     "/home/ruben/PycharmProjects/mini_Genie_ML/Datas/Alternative_Data"],
)
AD_dobj.data
