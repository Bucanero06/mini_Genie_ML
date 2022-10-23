from datetime import datetime, timedelta

# print(pct_prices)
# pct_prices= pct_prices.to_csv('sp30_rolling_corr.csv')
import fredpy as fp
import numpy as np
import pandas as pd

from Modules._Data_Manager import Data_Manager

fp.api_key = "fcd3f2f8ace61a2705a03e20b13420d5"


def get_fred_series(series_id):
    fred_series = fp.series(series_id)
    print(fred_series.title)
    print(fred_series.units)
    print(fred_series.frequency)
    print(fred_series.date_range)
    print(fred_series.source + "\n")
    return fred_series


def compute_logrealGDPpercapita_returns_fredseries():
    gdp = get_fred_series("GDP")
    defl = get_fred_series("GDPDEF")
    # Make sure that all series have the same window of observation
    gdp, defl = fp.window_equalize([gdp, defl])
    # Deflate GDP series
    gdp = gdp.divide(defl)
    # Convert GDP to per capita terms
    gdp = gdp.per_capita()
    # Take log of GDP
    gdp = gdp.log()
    return gdp


def fetch_alternative_data(alternative_data_dir):
    AD_df = pd.DataFrame()  # Alternative Data
    #
    log_gdp = compute_logrealGDPpercapita_returns_fredseries()

    gdp_cycle, gdp_trend = log_gdp.hp_filter()
    AD_df["gdp_cycle"], AD_df["gdp_trend"] = gdp_cycle.data, gdp_trend.data

    AD_df["log_gdp"] = log_gdp.data

    # T-bill data
    AD_df["t-bill"] = get_fred_series("TB3MS").data.ffill()

    # Construct the inflation series
    inflation = get_fred_series("CPIAUCSL")
    inflation = inflation.pc(annualized=True)
    AD_df["inflation"] = inflation.ma(length=6, center=True).data.ffill()

    # Unemployment rate
    AD_df["unemployment"] = get_fred_series("LNS14000028").data.ffill()

    # Construct the real interest rate series
    AD_df["real_rate"] = AD_df["t-bill"] - AD_df["inflation"]
    print(AD_df)

    AD_df = AD_df.ffill().dropna()
    AD_df.to_csv(f'{alternative_data_dir}/AD_df.csv')
    print(AD_df)
    return AD_df


def fetch_index_asset_prices(days_to_fetch=2, last_day=None, index_name="SP500",
                             read_from_file=False, alternative_data_dir=None):
    import pandas as pd

    if last_day is None:
        last_day = datetime.today()

    index_ticker_dict = dict(
        SP500=dict(
            ticker="^GSPC",
            historic_file="/home/ruben/PycharmProjects/mini_Genie_ML/Datas/Alternative_Data/S&P_500_Historical_Components_n_Changes(08-12-2022).csv",
        ),
        # NASDAQ="^IXIC",
        # DOW="^DJI",
        # RUSSELL="^RUT",
        # FTSE="^FTSE",
        # DAX="^GDAXI",
        # CAC="^FCHI",
        # NIKKEI="^N225",
        # HANGSENG="^HSI",
    )

    metric = ...  # todo: add metric for now use the weights given by the database

    if not read_from_file:
        if index_name in index_ticker_dict.keys():
            index_ticker_name = index_ticker_dict[index_name]["ticker"]
            ticker_file_path = index_ticker_dict[index_name]["historic_file"]
        else:
            print("Index name not found, changing to SP500")
            index_ticker_name = index_ticker_dict["SP500"]["ticker"]
            ticker_file_path = index_ticker_dict["SP500"]["historic_file"]
        #
        index_tickers_hist_list = pd.read_csv(ticker_file_path, index_col='date', parse_dates=True)

        # Convert ticker column from csv to list, then sort.
        index_tickers_hist_list['tickers'] = index_tickers_hist_list['tickers'].apply(lambda x: sorted(x.split(',')))

        # Set the start date given the number of days to fetch
        start_time = last_day - timedelta(days=days_to_fetch)

        # Add the last_day date the index_tickers_hist_list dataframe
        index_tickers_hist_list.loc[last_day] = index_tickers_hist_list.iloc[-1]

        # Get dates within the range
        index_tickers_hist_list = Data_Manager.fetch_dates_from_df(index_tickers_hist_list, start_date=start_time,
                                                                   end_date=last_day)
        # Get unique tickers from the list
        unique_tickers = set()
        for tickers in index_tickers_hist_list['tickers']:
            # change . to - for yahoo finance
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            unique_tickers.update(tickers)
        unique_tickers = sorted(unique_tickers)

        unique_tickers.append(index_ticker_name)
        # Get the prices for the tickers
        assets_to_download = ...  # N assets with highest market cap during the dates prqovided (top_N != assets_to_download)
        print(f"Fetching {len(unique_tickers)} assets from {start_time} to {last_day} found in the S&P500\n"
              f"following the added and removed tickers from the index over time")

        #
        # data = vbt.YFData.fetch(
        #     unique_tickers,
        #     start=start_time,
        #     end=last_day,
        #     timeframe="1d",
        #     ignore_errors=True,
        #     # missing_index='drop'  # Creates problems with missing the index
        # )
        # # 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'
        #

        # Combine each asset columns by naming convention f"{asset}_{metric}"
        # Also compute any metric that is needed (or desired)
        from pandas_datareader import DataReader
        returning_df = pd.DataFrame()
        for asset in unique_tickers:
            print(f"Fetching {asset} data")
            # INDEX:Date, High, Low, Open, Close, Volume, Adj Close
            data = DataReader(asset, "yahoo", start=start_time, end=last_day)
            print(data.head())
            # for metric in data.wrapper.columns:
            for metric in data.columns:
                if metric not in ["Dividends"]:
                    print(f"{asset}_{metric}")
                    # returning_df[f"{asset}_{metric}"] = data.data[asset][metric]
                    returning_df[f"{asset}_{metric}"] = data[metric].copy()

            # Custom metrics
            # returning_df[f"{asset}_log_returns"] = np.log(data.data[asset]["Close"]).diff()
            returning_df[f"{asset}_log_returns"] = np.log(data["Close"].copy()).diff()

        # # Add the index prices
        # returning_df[f"{index_ticker_name}_Close"] = data.data[index_ticker_name]["Close"]

        print(returning_df.head())
        exit()
        returning_df = returning_df.ffill()
        returning_df.to_csv(
            f"{alternative_data_dir}/{index_name}_assets_historic_prices.csv")

        # exit()
    else:
        return Data_Manager().fetch_data(
            data_file_names=f"{index_name}_assets_historic_prices.csv",
            data_file_dirs=["../../../Datas",
                            ".", "Datas", "Datas/Sample-Data", alternative_data_dir],
        )


def fetch_all_prices():
    ...


def create_hierarchical_etf_prices(df, weights):
    ...


if __name__ == '__main__':
    # forex_data = vbt.Data.load("forex_tick_data")

    # Get the prices for the tickers found in the index
    assets_data = fetch_index_asset_prices(days_to_fetch=10, last_day=datetime.today() - timedelta(days=1000)
                                           , index_name="SP500", read_from_file=False)
    alternative_data_dir = "/Datas/Alternative_Data"

    print(assets_data.data)

    exit()
    # Label data
    AD_df = fetch_alternative_data(alternative_data_dir)

    print(forex_data.data)

    print(assets_data.data)

    print(AD_df.head())
    exit()
    # Combine to Head Data Frame
    Head_DF = pd.concat([asset_prices, AD_df], axis=1)
    asset_prices.columns = ["price", "unemployment_rate"]

    asset_prices = asset_prices.ffill().dropna()
    # plt.plot(asset_prices.index, asset_prices.values, '-', lw=3, alpha=0.65)
    # plt.grid()
    # plt.show()
    exit()
    data = Data_Manager().fetch_data(data_file_names=FILE_NAME,
                                     data_file_dirs=["../../../Datas",
                                                     ".", "Datas", "Datas/Sample-Data"],
                                     )

    # .legend(['First line', 'Second line'])
    exit()
    price = data["Ask"]
    synthetic_strategy_namedtuple = __genie_strategy__(price,
                                                       volatility_window=50,
                                                       return_namedtuple_volatility=True,
                                                       REAL=True)
    synthetic_strategy_namedtuple.df.to_csv("test_synth_data.csv")
    exit()

    ##########################
