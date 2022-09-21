# from _pipeline_scripts_examples.fml.Backtest import backtest_cv
#
# backtest_df = backtest_cv('Datas/USDJPY_Tick_Data/Research_Data.csv')



def get_dollar_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: Union[float, pd.Series] = 70000000,
                    batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the dollar bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
    it is suggested that using 1/50 of the average daily dollar value, would result in more desirable statistical
    properties.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param threshold: (float, or pd.Series) A cumulative value above this threshold triggers a sample to be taken.
                      If a series is given, then at each sampling time the closest previous threshold is used.
                      (Values in the series can only be at times when the threshold is changed, not for every observation)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) Dataframe of dollar bars
    """

    bars = StandardBars(metric='cum_dollar_value', threshold=threshold, batch_size=batch_size)
    dollar_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return dollar_bars