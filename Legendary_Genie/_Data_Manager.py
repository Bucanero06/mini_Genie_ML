from mini_Genie.mini_genie_source.Data_Handler.data_handler import Data_Handler  # noqa: F401
from src._bars_aggregators import BarsAggregator  # noqa: F401


class Data_Manager:

    @staticmethod
    def fetch_data(data_file_names, data_file_dirs):
        import vectorbtpro as vbt
        if not data_file_dirs:
            data_file_dirs = [".", "Datas", "Sample-Data"]
        if not isinstance(data_file_names, list):
            data_file_names = [data_file_names]

        from Legendary_Genie.Utils import find_file
        data_file_paths = []

        for file_name in data_file_names:
            directory = find_file(file_name, *data_file_dirs)
            __path = f'{directory}/{file_name}'
            data_file_paths.append(__path)
        # data_file_dir = data_file_dir or find_file(f'{data_file_name}.csv', *search_in)
        # data_file_paths = [f'{data_file_dir}/{data_file_name}.csv' for data_file_name in data_file_names]
        #     # data= Data_Handler.fetch_csv_data_dask(data_file_name=data_file_names, data_file_dir=data_file_dir,
        #     #                                  scheduler=scheduler)
        #     #
        #     # return vbt.Data.from_data(data)
        #
        #
        # data_array = [Data_Handler.fetch_csv_data_dask(data_file_name=data_file_name, data_file_dir=data_file_dir,
        #                                                scheduler=scheduler) for data_file_name in data_file_names]
        #
        # datas_dict = {}
        # for data_name, data_bars in zip(data_file_names, data_array):
        #     datas_dict[data_name] = data_bars
        #
        # logger.info(f'Converting data to symbols_data obj')
        # return vbt.Data.from_data(datas_dict)
        return vbt.CSVData.fetch(data_file_paths, index_col=0,
                                 parse_dates=True, infer_datetime_format=True)

    @staticmethod
    def get_bars(data, out_bar_type='time', **kwargs):
        from mlfinlab_scripts.src._bars_aggregators import BarsAggregator
        bars_aggregator = BarsAggregator()

        return bars_aggregator.get_bars(data, out_bar_type=out_bar_type, kwargs=kwargs)

