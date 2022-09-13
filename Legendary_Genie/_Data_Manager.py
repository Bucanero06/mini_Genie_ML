from logger_tt import logger

from mini_Genie.mini_genie_source.Data_Handler.data_handler import Data_Handler  # noqa: F401
from src._bars_aggregators import BarsAggregator  # noqa: F401


class Data_Manager:

    @staticmethod
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
        logger.info(f'Loading {data_file_name} from CSV file')

        from Utils import find_file
        directory = data_file_dir if data_file_dir else find_file(data_file_name, *search_in)

        data_file_path = f'{directory}/{data_file_name}'

        # load the data into a dask dataframe
        # bar_data = dd.read_csv(f'{data_file_dir}/{data_file_name}.csv', parse_dates=True)
        if n_rows:
            if first_or_last == 'first':
                bar_data = dd.read_csv(data_file_path, parse_dates=True).head(n_rows)
            elif first_or_last == 'last':
                bar_data = dd.read_csv(data_file_path, parse_dates=True).tail(n_rows)
        else:
            bar_data = dd.read_csv(data_file_path, parse_dates=True)

        # convert all column names to upper case
        bar_data.columns = bar_data.columns.str.upper()
        logger.info(f'Finished Loading {data_file_name} from CSV file')
        logger.info(f'Prepping {data_file_name} for use')
        logger.info(f'_parsing dates')
        # get the name of the datetime column
        datetime_col = bar_data.columns[0]
        # parse the datetime column

        bar_data[datetime_col] = dd.to_datetime(bar_data[datetime_col])
        # bar_data[datetime_col] = bar_data[datetime_col].dt.strftime(output_format)
        if not n_rows:
            logger.info(f'_dask_compute')
            # compute the dask dataframe
            bar_data = bar_data.compute(scheduler=scheduler)

        # set the datetime column as the index
        bar_data.index = bar_data[datetime_col]
        # delete the datetime column
        del bar_data[datetime_col]
        return bar_data

    def fetch_data(self, data_file_names, data_file_dirs, **kwargs):

        import vectorbtpro as vbt
        if not data_file_dirs:
            data_file_dirs = [".", "Datas", "Sample-Data"]
        if not isinstance(data_file_names, list):
            data_file_names = [data_file_names]

        from Legendary_Genie.Utils import find_file
        data_file_paths = []
        data_array = []

        for file_name in data_file_names:
            directory = find_file(file_name, *data_file_dirs)
            __path = f'{directory}/{file_name}'
            data_file_paths.append(__path)

            data = self.fetch_csv_data_dask(data_file_name=file_name, data_file_dir=directory,
                                            search_in=data_file_dirs,
                                            scheduler=kwargs.get('scheduler', 'threads'),
                                            n_rows=kwargs.get('n_rows',None),
                                            first_or_last=kwargs.get('first_or_last', 'first'))

            data_array.append(data)

        datas_dict = {}
        for data_name, data_bars in zip(data_file_names, data_array):
            data_name = data_name.split('.')[0]
            datas_dict[data_name] = data_bars

        logger.info(f'Converting data to symbols_data obj')
        return vbt.Data.from_data(datas_dict)

        # return vbt.CSVData.fetch(data_file_paths, index_col=0,
        #                          parse_dates=True, infer_datetime_format=True)

    @staticmethod
    def get_bars(data, out_bar_type='time', **kwargs):
        from mlfinlab_scripts.src._bars_aggregators import BarsAggregator
        bars_aggregator = BarsAggregator()

        return bars_aggregator.get_bars(data, out_bar_type=out_bar_type, kwargs=kwargs)
