from logger_tt import logger
from matplotlib import pyplot as plt



# from src._bars_aggregators import BarsAggregator  # noqa: F401


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

        from Modules.Actors_Old.Utils import find_file
        directory = data_file_dir if data_file_dir else find_file(data_file_name, *search_in)

        data_file_path = f'{directory}/{data_file_name}'
        #add .csv if not present
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
        logger.info(f'Finished Loading {data_file_name} from CSV file')
        logger.info(f'Prepping {data_file_name} for use')
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

    def fetch_data(self, data_file_names, data_file_dirs, **kwargs):

        import vectorbtpro as vbt
        if not data_file_dirs:
            data_file_dirs = [".", "Datas", "Sample-Data"]
        if not isinstance(data_file_names, list):
            data_file_names = [data_file_names]

        data_file_paths = []
        data_array = []

        for file_name in data_file_names:
            from Modules.Actors_Old.Utils import find_file
            directory = find_file(file_name, *data_file_dirs)
            __path = f'{directory}/{file_name}'
            data_file_paths.append(__path)

            try:
                data = self.fetch_csv_data_dask(data_file_name=file_name, data_file_dir=directory,
                                                search_in=data_file_dirs,
                                                scheduler=kwargs.get('scheduler', 'threads'),
                                                n_rows=kwargs.get('n_rows', None),
                                                first_or_last=kwargs.get('first_or_last', 'first'))
            except Exception as e:
                # logger.exception(f'{e = }')
                logger.warning(f'Could not load {file_name} as a timeseries thus is being loaded but not prepared')
                import dask.dataframe as dd
                directory = find_file(file_name, *data_file_dirs)
                data_file_path = f'{directory}/{file_name}'
                data = dd.read_csv(data_file_path, parse_dates=False).compute(
                    scheduler=kwargs.get('scheduler', 'threads'))

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
    def corrmat_to_ts(mats, mat_type, t_samples,
                      starting_prices,
                      plot=False
                      ):
        from mlfinlab.data_generation.hcbm import time_series_from_dist  # noqa: F401

        ts_dict__ = dict()
        for i in range(len(mats)):
            # Generate time series
            series_df = time_series_from_dist(mats[i], dist='student', t_samples=t_samples)
            series_df = series_df.cumsum()

            if starting_prices:
                # Add starting asset_prices
                for col in series_df.columns:
                    series_df[col] = series_df[col] + starting_prices[col]
            ts_dict__[f"ts_{i + 1}"] = series_df

            if plot:
                from mlfinlab.data_generation.data_verification import plot_optimal_hierarchical_cluster  # noqa: F401

                # Plot time series
                series_df.plot(legend=None, title=f"Recovered {mat_type} Time Series {i + 1} from distribution")
                plt.show()
                #
                # Plot recovered HCBM matrix
                plot_optimal_hierarchical_cluster(series_df.corr(), method="ward")
                plt.title(
                    f"Recovered {mat_type} Matrix {i + 1} from distribution of time series recovered from the matrix")
                plt.show()

        return ts_dict__

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



