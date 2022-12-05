#!/usr/bin/env python3
import gc
import warnings
from os.path import exists

import numpy as np
import pandas as pd
import ray
import vectorbtpro as vbt
from dask import dataframe as dd
from logger_tt import logger

from mini_genie_source.Utilities.reduce_utilities import min_reduce_nb

warnings.simplefilter(action='ignore', category=FutureWarning)


class Data_Handler:
    """
    Handles Data, everything from loading to saving

    Args:
         ():
         ():

    Attributes:
         ():
         ():

    Note:
        This is crazy
    """

    def __init__(self, genie_object):
        """Constructor for Data_Handler"""
        self.genie_object = genie_object
        from Modules._Data_Manager import Data_Manager
        self.data_manager = Data_Manager()

        #

    def print_dict(self, optional_object: object = None) -> object:
        import pprint
        pprint.pprint(self.__dict__ if not optional_object else optional_object.__dict__)

    #
    # @staticmethod
    # def fetch_data(data_file_names, data_file_dirs):
    #     if not data_file_dirs:
    #         data_file_dirs = [".","Datas","Sample-Data"]
    #     if not isinstance(data_file_names, list):
    #         data_file_names = [data_file_names]
    #
    #     from Modules.Utils import find_file
    #     data_file_paths = []
    #
    #     for study_name in data_file_names:
    #         directory = find_file(study_name, *data_file_dirs)
    #         __path = f'{directory}/{study_name}'
    #         data_file_paths.append(__path)
    #
    #     return vbt.CSVData.fetch(data_file_paths, index_col=0,
    #                       parse_dates=True, infer_datetime_format=True)

    # @staticmethod
    # def fetch_csv_data(data: object, data_files_dir: object) -> object:
    #     """
    #
    #     Args:
    #         data:
    #         data_files_dir:
    #
    #     Returns:
    #         object:
    #
    #     """
    #     logger.info(f'Loading {data} from CSV file')
    #     bar_data = pd.read_csv(f'{data_files_dir}/{data}.csv',
    #                            index_col=0, parse_dates=True)
    #     bar_data.columns = bar_data.columns.str.upper()
    #
    #     logger.info(f'Finished Loading {data} from CSV file')
    #     return bar_data

    # @staticmethod

    # @staticmethod
    # def fetch_csv_data_dask(data_file_name: object, data_file_dir: object = None,
    #                         search_in=(".", "Datas"), scheduler='threads') -> object:
    #     """
    #     Loads data from a CSV file into a dask dataframe
    #     :param data_file_name: name of the data file
    #     :param data_file_dir: directory of the data file
    #     :param input_format: format of the date in the data file
    #     :param output_format: format of the date to be outputted
    #     :return: dask dataframe
    #     """
    #     logger.info(f'Loading {data_file_name} from CSV file')
    #
    #     data_file_path = f'{data_file_dir}/{data_file_name}.csv'
    #
    #     # load the data into a dask dataframe
    #     # bar_data = dd.read_csv(f'{data_file_dir}/{data_file_name}.csv', parse_dates=True)
    #     bar_data = dd.read_csv(data_file_path, parse_dates=True)
    #     # convert all column names to upper case
    #     bar_data.columns = bar_data.columns.str.upper()
    #     logger.info(f'Finished Loading {data_file_name} from CSV file')
    #     logger.info(f'Prepping {data_file_name} for use')
    #     logger.info(f'_parsing dates')
    #     # get the name of the datetime column
    #     datetime_col = bar_data.columns[0]
    #     # parse the datetime column
    #
    #     bar_data[datetime_col] = dd.to_datetime(bar_data[datetime_col])
    #     # bar_data[datetime_col] = bar_data[datetime_col].dt.strftime(output_format)
    #     logger.info(f'_dask_compute')
    #     # compute the dask dataframe
    #     bar_data = bar_data.compute(scheduler=scheduler)
    #     # set the datetime column as the index
    #     bar_data.index = bar_data[datetime_col]
    #     # delete the datetime column
    #     del bar_data[datetime_col]
    #     return bar_data

    def compute_spread_from_ask_bid_data(self, tick_data):
        """
        Compute the spread from the ask and bid data
        :param self:
        :param tick_data:
        :return:
        """
        logger.info(f'_computing_spread')
        # Compute the spread
        tick_size = self.genie_object.runtime_settings["Data_Settings.tick_size"]
        tick_data["SPREAD"] = ((tick_data["BID"] - tick_data["ASK"]).abs() / tick_size).round()
        return tick_data

    def fetch_csv_data_add_spread_(self, data_name: object, data_files_dir: object) -> object:
        """

        Args:
            data:
            data_files_dir:

        Returns:
            object:

        """

        bar_data = self.fetch_csv_data_dask(data_name, data_files_dir)
        # bar_data = pd.DataFrame()

        #
        logger.info(f'1{bar_data.head() = }')
        logger.info(f'_done')
        # exit()
        if "SPREAD" not in bar_data.columns and exists(f'{data_files_dir}/{data_name}_tick.csv'):
            bar_data_tick = self.fetch_csv_data_dask(f'{data_name}_tick', data_files_dir,
                                                     # input_format=self.genie_object.runtime_settings[
                                                     #     "Data_Settings.accompanying_tick_data_input_format"],
                                                     # output_format=self.genie_object.runtime_settings[
                                                     #     "Data_Settings.accompanying_tick_data_output_format"]
                                                     )
            # todo for now genie does not use tick data other than to compute the max spread
            if ("ASK" and "BID") in (bar_data_tick.columns):
                logger.info(
                    f'Loading {data_name}_tick.csv to compute, resample to minute, and add spread column to {data_name}.csv')
                bar_data_tick = bar_data_tick[["ASK", "BID"]]
                #
                bar_data_tick = self.compute_spread_from_ask_bid_data(bar_data_tick)
                logger.info(f'1{bar_data_tick.head() = }')
                #
                logger.info(f'_resampling spread')
                tick_spread_resampled = bar_data_tick["SPREAD"].vbt.resample_apply('1 min', min_reduce_nb,
                                                                                   # chunked=True
                                                                                   ).dropna()

                # tick_spread_resampled = bar_data_tick["SPREAD"].vbt.resample_apply('1 min', max_reduce_nb)
                # tick_spread_resampled = bar_data_tick["SPREAD"].groupby(['id', pd.Grouper(freq='D')])['value'].sum()

                # dd_tick_spread = dd.from_pandas(bar_data_tick["SPREAD"], npartitions=100)
                # logger.info(f'{dd_tick_spread.divisions= }')
                # tick_spread_resampled = dd_tick_spread.resample('1 min').agg(np.max)

                # tick_spread_resampled = bar_data_tick["SPREAD"].groupby(pd.Grouper(freq='1 min')).max().dropna()
                # logger.info(f'{tick_spread_resampled.head(10) = }')
                # tick_spread_resampled = dask_series.resample(freq, label=label).dropna()

                logger.info(f'2{tick_spread_resampled.head() = }')
                #
                # from Utilities.bars_utilities import resample_dask_series
                # tick_spread_resampled = resample_dask_series(bar_data_tick["SPREAD"], '1 min', label=None).sum().compute(scheduler='processes')
                #
                logger.info(f'{tick_spread_resampled.head()=  }')
                logger.info(f'{len(tick_spread_resampled) =  }')
                logger.info(f'{len(bar_data) =  }')
                assert len(tick_spread_resampled) == len(bar_data)
                bar_data["SPREAD"] = tick_spread_resampled
                logger.info(f'{bar_data["SPREAD"].head() =  }')
                bar_data.to_csv(f'{data_files_dir}/{data_name}_with_spread_column.csv')

                # bar_data = self.fetch_csv_data_dask(f'{data_name}_with_spread_column.csv', data_files_dir,
                #                                     input_format="%Y.%m.%d %H:%M:%S",
                #                                     output_format='%m.%d.%Y %H:%M:%S')
                # bar_data.to_csv(f'{data_files_dir}/{data_name}_with_spread_column.csv')

                ## bar_data = self.resample_ask_bid_data_to_minute()

        logger.info(f'{bar_data.head() = }')
        return bar_data

    def fetch_csv_data_add_spread(self, data_name: object, data_files_dir: object) -> object:
        """
        add spread column to minute data
        :param self:
        :param data_name:
        :param data_files_dir:
        :return:
        """

        bar_data = self.data_manager.fetch_csv_data_dask(data_name, data_files_dir,
                                            #                                 input_format=self.genie_object.runtime_settings[
                                            # "Data_Settings.minute_data_input_format"], output_format=self.genie_object.runtime_settings[
                                            # "Data_Settings.minute_data_output_format"]
                                            )
        print(bar_data)
        exit()
        if "SPREAD" not in bar_data.columns and exists(
                f'{data_files_dir}/{data_name}_tick.csv'):  # if spread column is not in minute data and tick data exists
            bar_data_tick = self.data_manager.fetch_csv_data_dask(f'{data_name}_tick', data_files_dir,
                                                     # input_format=self.genie_object.runtime_settings[
                                                     #     "Data_Settings.accompanying_tick_data_input_format"],
                                                     # output_format=self.genie_object.runtime_settings[
                                                     #     "Data_Settings.accompanying_tick_data_output_format"]
                                                     )
            if ("ASK" and "BID") in (bar_data_tick.columns):  # if ask and bid columns are in tick data
                logger.info(
                    f'Loading {data_name}_tick.csv to compute, resample to minute, and add spread column to {data_name}.csv')
                bar_data_tick = bar_data_tick[["ASK", "BID"]]  # select only ask and bid columns
                bar_data_tick = self.compute_spread_from_ask_bid_data(bar_data_tick)  # compute spread column
                logger.info(f'_resampling spread')
                tick_spread_resampled = bar_data_tick["SPREAD"].vbt.resample_apply('1 min', min_reduce_nb,
                                                                                   # resample spread column to minute
                                                                                   # chunked=True
                                                                                   ).dropna()
                assert len(tick_spread_resampled) == len(
                    bar_data)  # check if length of resampled spread column is equal to length of minute data
                bar_data["SPREAD"] = tick_spread_resampled  # add resampled spread column to minute data
                bar_data.to_csv(
                    f'{data_files_dir}/{data_name}_with_spread_column.csv')  # save minute data with spread column
        return bar_data

    def _fetch_data(self) -> object:
        """
        Returns:
            object:
        """
        logger.info(f'Fetching Data')
        #
        load_from_pickle = self.genie_object.runtime_settings['Data_Settings.load_CSV_from_pickle']
        continuing_study = self.genie_object.continuing
        data_file = f'{self.genie_object.study_dir_path}/{self.genie_object.runtime_settings["Data_Settings.saved_data_file"]}'
        #
        if load_from_pickle and continuing_study and exists(f'{data_file}.pickle'):
            logger.warning("Loading data from pickle not reading from CSV")
            symbols_data = vbt.Data.load(
                f'{data_file}')

        else:
            data_array = [self.data_manager.fetch_csv_data_dask(data_name, self.genie_object.runtime_settings[
            # data_array = [self.fetch_csv_data_add_spread(data_name, self.genie_object.runtime_settings[
                "Data_Settings.data_files_dir"]) for data_name in
                          self.genie_object.runtime_settings["Data_Settings.data_files_names"]]

            datas_dict = {}
            for data_name, data_bars in zip(self.genie_object.runtime_settings["Data_Settings.data_files_names"],
                                            data_array):
                datas_dict[data_name] = data_bars

            logger.info(f'Converting data to symbols_data obj')
            symbols_data = vbt.Data.from_data(datas_dict)
            symbols_data.save(
                f'{data_file}')
            logger.info(
                f'Saved Symbol\'s Data to {data_file}')

        setattr(self.genie_object, "symbols_data_id", ray.put(symbols_data))

        return self

    @staticmethod
    def fetch_dates_from_df(df, start_date=None, end_date=None):
        """Cut DF
        :param df: pandas.DataFrame
        :param start_date: str
        :param end_date: str
        :return: pandas.DataFrame
        """
        if start_date is None:
            start_date = df.index[0]
        if end_date is None:
            end_date = df.index[-1]
        df_index = df.index
        mask = (df_index >= start_date) & (df_index <= end_date)
        return df.loc[mask]

    def break_up_olhc_data_from_symbols_data(self) -> object:
        """Separates the OLHC input data into two (or three) chunks:
                        1.  Optimization Date-Range
                        2.  Date-Range prior to "(1)" to be used for \bar{ATR}
                        3.  The rest of the data that will not be utilized"""
        # if not self.genie_object.symbols_data_id:
        #     self._fetch_data()

        '''Get OLHC'''
        # symbols_data = ray.get(self.genie_object.symbols_data_id)
        symbols_data = ray.get(
            self.genie_object.symbols_data_id) if self.genie_object.symbols_data_id else self.genie_object.symbols_data

        # open_data = symbols_data.get('OPEN')
        open_data = symbols_data.get('open')

        idx = pd.date_range(open_data.index[0], open_data.index[len(open_data) - 1], freq='1 min')

        open_data = open_data.tz_localize(None) if self.genie_object.runtime_settings[
            "Data_Settings.delocalize_data"] else open_data
        open_data = open_data.dropna() if self.genie_object.runtime_settings["Data_Settings.drop_nan"] else open_data
        open_data = open_data.ffill() if self.genie_object.runtime_settings["Data_Settings.ffill"] else open_data
        open_data = open_data.reindex(idx) if self.genie_object.runtime_settings[
            "Data_Settings.fill_dates"] else open_data

        low_data = symbols_data.get('low')
        low_data = low_data.tz_localize(None) if self.genie_object.runtime_settings[
            "Data_Settings.delocalize_data"] else low_data
        low_data = low_data.dropna() if self.genie_object.runtime_settings["Data_Settings.drop_nan"] else low_data
        low_data = low_data.ffill() if self.genie_object.runtime_settings["Data_Settings.ffill"] else low_data
        low_data = low_data.reindex(idx) if self.genie_object.runtime_settings["Data_Settings.fill_dates"] else low_data

        high_data = symbols_data.get('high')
        high_data = high_data.tz_localize(None) if self.genie_object.runtime_settings[
            "Data_Settings.delocalize_data"] else high_data
        high_data = high_data.dropna() if self.genie_object.runtime_settings["Data_Settings.drop_nan"] else high_data
        high_data = high_data.ffill() if self.genie_object.runtime_settings["Data_Settings.ffill"] else high_data
        high_data = high_data.reindex(idx) if self.genie_object.runtime_settings[
            "Data_Settings.fill_dates"] else high_data

        close_data = symbols_data.get('close')
        close_data = close_data.tz_localize(None) if self.genie_object.runtime_settings[
            "Data_Settings.delocalize_data"] else close_data
        close_data = close_data.dropna() if self.genie_object.runtime_settings["Data_Settings.drop_nan"] else close_data
        close_data = close_data.ffill() if self.genie_object.runtime_settings["Data_Settings.ffill"] else close_data
        close_data = close_data.reindex(idx) if self.genie_object.runtime_settings[
            "Data_Settings.fill_dates"] else close_data

        # assert there is enough data for the optimization period and the warm up period
        data_start_date = close_data.index[0]
        data_end_date = close_data.index[len(close_data) - 1]
        print(f'Range of Data Source from {data_start_date} to {data_end_date}')
        #
        bar_atr_days = self.genie_object.tp_sl_selection_space["bar_atr_days"]
        from datetime import timedelta
        one_day = timedelta(days=1, hours=0, minutes=0, seconds=0)
        bar_atr_start_date = self.genie_object.optimization_start_date - bar_atr_days
        bar_atr_end_date = self.genie_object.optimization_start_date - one_day
        #
        optimization_start_date = self.genie_object.optimization_start_date
        optimization_end_date = self.genie_object.optimization_end_date
        #

        try:
            assert data_start_date < bar_atr_start_date
            assert data_end_date > optimization_end_date
        except:
            logger.error(f'Not enough data to meet Warm-up or Optimization Period')
            exit()

        #  Split
        optimization_open_data = self.fetch_dates_from_df(open_data, optimization_start_date, optimization_end_date)
        optimization_low_data = self.fetch_dates_from_df(low_data, optimization_start_date, optimization_end_date)
        optimization_high_data = self.fetch_dates_from_df(high_data, optimization_start_date, optimization_end_date)
        optimization_close_data = self.fetch_dates_from_df(close_data, optimization_start_date, optimization_end_date)
        #
        bar_atr_open_data = self.fetch_dates_from_df(open_data, bar_atr_start_date, bar_atr_end_date)
        bar_atr_low_data = self.fetch_dates_from_df(low_data, bar_atr_start_date, bar_atr_end_date)
        bar_atr_high_data = self.fetch_dates_from_df(high_data, bar_atr_start_date, bar_atr_end_date)
        bar_atr_close_data = self.fetch_dates_from_df(close_data, bar_atr_start_date, bar_atr_end_date)

        logger.info(
            f'\\bar_ATR Warm-Up Data -> From: {bar_atr_close_data.index[0]} to {bar_atr_close_data.index[len(bar_atr_close_data) - 1]}')
        #
        logger.info(f'\n')
        logger.info(
            f'Optimization Data -> From: {optimization_close_data.index[0]} to {optimization_close_data.index[len(optimization_close_data) - 1]}')

        # fixme Just for debugging what comes next because tick data is being a pain in the ass
        if "spread" in symbols_data.wrapper.columns:
            spread_data = symbols_data.get('spread')
            spread_data = spread_data.tz_localize(None) if self.genie_object.runtime_settings[
                "Data_Settings.delocalize_data"] else spread_data
            spread_data = spread_data.dropna() if self.genie_object.runtime_settings[
                "Data_Settings.drop_nan"] else spread_data
            spread_data = spread_data.ffill() if self.genie_object.runtime_settings[
                "Data_Settings.ffill"] else spread_data
            spread_data = close_data.reindex(idx) if self.genie_object.runtime_settings[
                "Data_Settings.fill_dates"] else spread_data
            optimization_spread_data = self.fetch_dates_from_df(spread_data, optimization_start_date,
                                                                optimization_end_date)
        else:
            print(f'{optimization_close_data.head() = }')
            # optimization_spread_data = pd.DataFrame().reindex_like(optimization_close_data).fillna(-np.inf)
            optimization_spread_data = -np.inf

        print(f'{optimization_close_data.head() = }')
        print(f'{optimization_close_data.shape = }')
        print(f'{optimization_high_data.head() = }')
        print(f'{optimization_high_data.shape = }')
        # EMA = vbt.IF.from_expr("""
        #                         ema=@talib_ema(close, window)
        #                         ema""")
        # ATR_EWM = vbt.IF.from_expr("""
        #                             ATR:
        #                             tr0 = abs(high - low)
        #                             tr1 = abs(high - fshift(close))
        #                             tr2 = abs(low - fshift(close))
        #                             tr = nanmax(column_stack((tr0, tr1, tr2)), axis=1)
        #                             atr = @talib_ema(tr, 2 * window - 1)  # Wilder's EMA
        #                             tr, atr
        #                             """)
        # print(ATR_EWM.run(high=optimization_high_data, low=optimization_low_data, close=optimization_close_data,
        #                   window=14).atr)
        # print(EMA.run(close=optimization_close_data, window=14).ema)
        # sys.exit()

        # Set \bar{ATR} Data Attr's  (ray.put)
        setattr(self.genie_object, "bar_atr_open_data", ray.put(bar_atr_open_data))
        setattr(self.genie_object, "bar_atr_low_data", ray.put(bar_atr_low_data))
        setattr(self.genie_object, "bar_atr_high_data", ray.put(bar_atr_high_data))
        setattr(self.genie_object, "bar_atr_close_data", ray.put(bar_atr_close_data))
        #
        # Set Optimization Data Attr's (ray.put)
        setattr(self.genie_object, "optimization_open_data", ray.put(optimization_open_data))
        setattr(self.genie_object, "optimization_low_data", ray.put(optimization_low_data))
        setattr(self.genie_object, "optimization_high_data", ray.put(optimization_high_data))
        setattr(self.genie_object, "optimization_close_data", ray.put(optimization_close_data))
        #
        setattr(self.genie_object, "optimization_spread_data", ray.put(optimization_spread_data))
        #
        gc.collect()
        return self

        ...

    # def set_to_m_d_y_h_m_s(self,):
    #     import pandas as pd
    #     from datetime import datetime
    #
    #     custom_date_parser = lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M:%S")
    #     df = pd.read_csv('Datas/DAX.csv',
    #                      parse_dates=['Datetime'],
    #                      date_parser=custom_date_parser)
    #
    #     df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.strftime("%m.%d.%Y %H:%M:%S")
    #     df.set_index("Datetime", inplace=True)
    #     df.to_csv('Datas/DAX.csv')
    #     print(df)
    #     exit()
