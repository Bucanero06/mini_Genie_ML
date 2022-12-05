import vectorbtpro as vbt

# Inputs:
TrendTimeframe = 'H1'
TrendRsi_Window = 13
TrendGreen_Window = 2
TrendRed_Window = 7
TrendBand_Window = 34

EntryTimeframe = 'M5'
EntryRsi_Window = 13
EntryGreen_Window = 2
EntryRed_Window = 7
EntryBand_Window = 34

file_names = 'XAUUSD.csv'
file_dir_paths = "/home/ruben/PycharmProjects/mini_Genie_ML/New_Nate_Work/Datas/"


def fetch_csv_data_dask(data_file_name: object, data_file_dir: object = None,
                        search_in=(".", "Datas"), scheduler='threads',
                        n_rows=None, first_or_last='first',
                        ) -> object:
    """
    Loads data from a CSV file into a dask dataframe
    :param data_file_name: name of the data file
    :param data_file_dir: directory of the data file
    :param input_format: format of the date in the data file
    :param output_format: format of the date to be outputted
    :return: dask dataframe
    """
    first_or_last = first_or_last.lower()

    from dask import dataframe as dd
    print(f'Loading {data_file_name} from CSV file')

    from Modules.Actors_Old.Utils import find_file
    directory = data_file_dir if data_file_dir else find_file(data_file_name, *search_in)

    data_file_path = f'{directory}/{data_file_name}'
    # add .csv if not present
    if not data_file_path.endswith('.csv'):
        data_file_path += '.csv'

    # load the data into a dask dataframe
    # bar_data = dd.read_csv(f'{data_file_dir}/{data_file_name}.csv', parse_dates=True)
    if n_rows:
        if first_or_last == 'first':
            bar_data = dd.read_csv(data_file_path, parse_dates=True, sample=100000000).head(n_rows)
        elif first_or_last == 'last':
            bar_data = dd.read_csv(data_file_path, parse_dates=True, sample=100000000).tail(n_rows)
    else:
        bar_data = dd.read_csv(data_file_path, parse_dates=True, sample=100000000)

    # convert all column names to upper case
    bar_data.columns = bar_data.columns.str.lower()
    print(f'Finished Loading {data_file_name} from CSV file')
    print(f'Prepping {data_file_name} for use')
    # get the name of the datetime column
    datetime_col = bar_data.columns[0]
    # parse the datetime column

    bar_data[datetime_col] = dd.to_datetime(bar_data[datetime_col])
    # bar_data[datetime_col] = bar_data[datetime_col].dt.strftime(output_format)
    if not n_rows:
        # compute the dask dataframe
        bar_data = bar_data.compute(scheduler=scheduler)

    # set the datetime column as the index
    bar_data.index = bar_data[datetime_col]
    # delete the datetime column
    del bar_data[datetime_col]
    return bar_data


ohlcv_dataframe = fetch_csv_data_dask(data_file_name=file_names, data_file_dir=file_dir_paths,
                                      search_in=(".", "Datas"), scheduler='threads',
                                      n_rows=None, first_or_last='first',
                                      )
ohlcv_dataframe["volume"] = ohlcv_dataframe["tick volume"]
print(ohlcv_dataframe)

open = ohlcv_dataframe['open']
high = ohlcv_dataframe['high']
low = ohlcv_dataframe['low']
close = ohlcv_dataframe['close']
volume = ohlcv_dataframe['volume']

print(open)
print(high)
print(low)
print(close)

# Trend Computation (use TrendTimeframe):
Rsi_Trend = vbt.RSI.run(close, TrendRsi_Window).rsi
Green_Trend = vbt.MA.run(Rsi_Trend, TrendGreen_Window)
Red_Trend = vbt.MA.run(Rsi_Trend, TrendRed_Window)  # not used

# We calculate a Bollinger Band of the Rsi, only want the "Middle" line
Bands = vbt.talib('BBANDS').run(Rsi_Trend, TrendBand_Window, 1.6185, 1.6185, 0)
#
# Trend Signals (multiple):
# Signals are Masks, optimize by selecting either Long_1, Long_2...
Long_1_Trend = Bands.middleband_above(50)
Long_2_Trend = Green_Trend.ma_above(Bands.middleband)
Long_3_Trend = Long_1_Trend | Long_2_Trend
Long_4_Trend = Long_1_Trend & Long_2_Trend

#
Short_1_Trend = Bands.middleband_below(50)
Short_2_Trend = Green_Trend.ma_below(Bands.middleband)
Short_3_Trend = Short_1_Trend | Short_2_Trend
Short_4_Trend = Short_1_Trend & Short_2_Trend
#
# Entry Computation (use EntryTimeframe):
Rsi_Entry = vbt.RSI.run(close, EntryRsi_Window).rsi
Green_Entry = vbt.MA.run(Rsi_Entry, EntryGreen_Window)
Red_Entry = vbt.MA.run(Rsi_Entry, EntryRed_Window)
# We calculate a Bollinger Band of the Rsi, only want the "Middle" line
Bands = vbt.talib('BBANDS').run(Rsi_Entry, EntryBand_Window, 1.6185, 1.6185, 0)
#
# Entry Signals (multiple):
# Signals are masks
Long_1_Entry = Green_Entry.ma.vbt.crossed_above(Red_Entry.ma)
Long_2_Entry = Long_1_Entry & Bands.middleband_above(50)
# Long_3_Entry = Long_1_Entry & Green_Entry.ma.combine(Red_Entry.ma).intercept_above(Bands.middleband) # comb needs work

Short_1_Entry = Green_Entry.ma.vbt.crossed_below(Red_Entry.ma)
Short_2_Entry = Short_1_Entry & Bands.middleband_below(50)
# Short_3_Entry = Short_1_Entry & Green_Entry.ma.combine(Red_Entry.ma).intercept_below(Bands.middleband) # comb needs work


print(Long_1_Entry.head())
print(Long_1_Entry.sum())
print(Long_2_Entry.head())
print(Long_2_Entry.sum())
# print(Long_3_Entry.head())
print('\n\n\n')
print(Short_1_Entry.head())
print(Short_1_Entry.sum())
print(Short_2_Entry.head())
print(Short_2_Entry.sum())
# print(Short_3_Entry.head())
