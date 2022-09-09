# noinspection PyUnresolvedReferences
from mlfinlab_scripts.src import _import_mlfinlab_env


class BarsAggregator:

    def __init__(self):
        from mlfinlab.data_structures import time_data_structures, standard_data_structures
        self.time_data_structures = time_data_structures
        self.standard_data_structures = standard_data_structures

    def get_bars(self, tick_bars, out_bar_type='time', kwargs={}):

        # TODO: Need to add a check to make sure input is in correct format, for now is up to the user

        assert tick_bars.index.name == None
        assert tick_bars.columns.tolist() == ['date_time', 'price', 'volume'] \
               or tick_bars.columns.tolist() == ['datetime', 'price', 'volume']

        if out_bar_type == 'time':
            return self.get_time_bars(tick_bars, **kwargs)
        elif out_bar_type == 'tick':
            return self.get_tick_bars(tick_bars, **kwargs)
        elif out_bar_type == 'volume':
            return self.get_volume_bars(tick_bars, **kwargs)
        else:
            raise ValueError(f'Invalid bar type: {out_bar_type}')

    def get_time_bars(self, tick_bars, resolution='MIN', verbose=False):
        """
        Creates Time Bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

        :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                                in the format[date_time, price, volume]
        :param resolution: (str) Resolution type ('D', 'H', 'MIN', 'S')
        :param num_units: (int) Number of resolution units (3 days for example, 2 hours)
        :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
        :param verbose: (int) Print out batch numbers (True or False)
        :param to_csv: (bool) Save bars to csv after every batch run (True or False)
        :param output_path: (str) Path to csv file, if to_csv is True
        :return: (pd.DataFrame) Dataframe of time bars, if to_csv=True return None
        """
        return self.time_data_structures.get_time_bars(tick_bars, resolution=resolution, verbose=verbose)

    def get_tick_bars(self, tick_bars, threshold=5500, batch_size=1000000, verbose=False):
        """
            Creates the tick bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

            :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                                     in the format[date_time, price, volume]
            :param threshold: (float, or pd.Series) A cumulative value above this threshold triggers a sample to be taken.
                              If a series is given, then at each sampling time the closest previous threshold is used.
                              (Values in the series can only be at times when the threshold is changed, not for every observation)
            :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
            :param verbose: (bool) Print out batch numbers (True or False)
            :param to_csv: (bool) Save bars to csv after every batch run (True or False)
            :param output_path: (str) Path to csv file, if to_csv is True
            :return: (pd.DataFrame) Dataframe of volume bars
            """
        return self.standard_data_structures.get_tick_bars(tick_bars, threshold=threshold,
                                                           batch_size=batch_size, verbose=verbose)

    def get_volume_bars(self, tick_bars, threshold=28000, batch_size=1000000, average=False, verbose=False):
        """
        Creates the volume bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

        Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
        it is suggested that using 1/50 of the average daily volume, would result in more desirable statistical properties.

        :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                                in the format[date_time, price, volume]
        :param threshold: (float, or pd.Series) A cumulative value above this threshold triggers a sample to be taken.
                          If a series is given, then at each sampling time the closest previous threshold is used.
                          (Values in the series can only be at times when the threshold is changed, not for every observation)
        :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
        :param verbose: (bool) Print out batch numbers (True or False)
        :param to_csv: (bool) Save bars to csv after every batch run (True or False)
        :param output_path: (str) Path to csv file, if to_csv is True
        :return: (pd.DataFrame) Dataframe of volume bars
        """
        return self.standard_data_structures.get_volume_bars(tick_bars, threshold=threshold,
                                                             batch_size=batch_size, verbose=verbose,
                                                             average=average)
