import gc

import numpy as np

from Modules._Data_Manager import Data_Manager

# Load the data
prices_df = Data_Manager().fetch_csv_data_dask(
    data_file_name="XAUUSD_GMT+0_NO-DST_M1.csv",
    # n_rows=100,
    search_in=
    ["/home/ruben/Downloads", "."])
# Print Index
print(f'Prices: \n{prices_df.head(100)}\n{prices_df.shape}')

# Prepare Time Series for Machine Learning (ML)
# Comppute the returns
prices_df['open'] = prices_df['open'].pct_change()
prices_df['low'] = prices_df['low'].pct_change()
prices_df['high'] = prices_df['high'].pct_change()
prices_df['close'] = prices_df['close'].pct_change()
# Compute 0 if abs(returns) < 0.0001, 1 if returns > 0, -1 if returns < 0,
prices_df['direction'] = prices_df['close'].apply(lambda x: 0 if abs(x) < 0.0001 else 1 if x > 0 else -1)

print(f'Prices: \n{prices_df[["close","direction"]].head(100)}\n{prices_df.shape}')

print('saving')
prices_df.to_csv(f"XAUUSD_pricesh2o.csv")
print('saved')