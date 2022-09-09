# Import MlFinLab tools

a=[1., 2., 3., 4., 5., 6., 7., 8.]
print(a[4])
exit()
import dask.dataframe as dd
from logger_tt import logger, setup_logging

setup_logging(full_context=1)


def fetch_csv_data_dask(data_name: object, data_files_dir: object,
                        input_format='%m.%d.%Y %H:%M:%S',
                        output_format='%m.%d.%Y %H:%M:%S', scheduler='processes'):
    """
    Loads data from a CSV file into a dask dataframe
    :param data_name: name of the data file
    :param data_files_dir: directory of the data file
    :param input_format: format of the date in the data file
    :param output_format: format of the date to be outputted
    :return: dask dataframe
    """
    logger.info(f'Loading {data_name} from CSV file')
    # load the data into a dask dataframe
    bar_data = dd.read_csv(f'{data_files_dir}/{data_name}.csv', parse_dates=True)
    # convert all column names to upper case
    bar_data.columns = bar_data.columns.str.lower()
    logger.info(f'Finished Loading {data_name} from CSV file')
    logger.info(f'Prepping {data_name} for use')
    logger.info(f'_parsing dates')
    # get the name of the datetime column
    datetime_col = bar_data.columns[0]
    # parse the datetime column
    bar_data[datetime_col] = dd.to_datetime(bar_data[datetime_col], format=input_format)
    logger.info(f'_dask_compute')
    # compute the dask dataframe
    # [distributed, multiprocessing, processes, single-threaded, sync, synchronous, threading, threads]
    bar_data = bar_data.compute(scheduler=scheduler)  # scheduler='processes'
    # set the datetime column as the index
    # bar_data.index = bar_data[datetime_col]
    # # delete the datetime column
    # del bar_data[datetime_col]
    return bar_data


if __name__ == '__main__':
    # data_files_dir = "../Sample-Data"
    # data_name = "USA100_tick"
    data_files_dir = "."
    data_name = "tick_data"

    # [distributed, multiprocessing, processes, single-threaded, sync, synchronous, threading, threads]
    tick_bars = fetch_csv_data_dask(data_files_dir=data_files_dir, data_name=data_name,
                                    # input_format="%m.%d.%Y %H:%M:%S.%f", scheduler='threads')
                                    input_format="%Y.%m.%d %H:%M:%S.%f", scheduler='threads')
    # tick_bars=tick_bars[:10000]
    # tick_bars.to_csv("temp_data.csv")

    from src._bars_aggregators import BarsAggregator

    bars_aggregator = BarsAggregator()
    time_bars = bars_aggregator.get_bars(tick_bars, out_bar_type='volume',
                                         kwargs=dict(threshold=10,
                                                     batch_size=1000000,
                                                     # average=False,
                                                     verbose=False
                                                     ))

    logger.info(f'{time_bars.columns.to_list() = }')
    logger.info(f'{time_bars.head(10) = }')
