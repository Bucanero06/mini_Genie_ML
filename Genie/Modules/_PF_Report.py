import gc
import glob

import numpy as np
import vectorbtpro as vbt
from mini_genie_source.Equipment_Handler.equipment_handler import CHECKTEMPS
from mini_genie_source.Run_Time_Handler.equipment_settings import TEMP_DICT


def filter_unmasked(mask, pf_daily=None, pf_weekly=None, pf_monthly=None, metric_name=None):
    passed_pfs = []
    if pf_daily is not None:
        passed_pfs.append(pf_daily)
    if pf_weekly is not None:
        passed_pfs.append(pf_weekly)
    if pf_monthly is not None:
        passed_pfs.append(pf_monthly)

    if len(mask) < 1:
        logger.warning(f'Portfolio filtered completely out by {metric_name} filter')

        for index, pf in enumerate(passed_pfs):
            passed_pfs[index] = None
        return passed_pfs
    else:
        for index, pf in enumerate(passed_pfs):
            passed_pfs[index] = pf[mask]
            metric_name = f"{metric_name} " if metric_name else ' '
            logger.info(f'After {metric_name}filter -> {pf_monthly.wrapper.shape[1]} strategies')
        return passed_pfs


# @ray.remote


def compute_metrics_report(pf, unique_params=None):
    # Check temperatures
    # print(f'\n\n{unique_params}\n\n')
    CHECKTEMPS(TEMP_DICT)
    #
    param_metrics = pf[unique_params].qs.metrics_report(display=False)
    # cols = param_metrics.columns
    #
    # Save the benchmark metrics only once
    param_metrics = param_metrics.drop(columns=['Benchmark'])
    # param_metrics.columns = [unique_params]
    print(f'Done with {unique_params}')
    CHECKTEMPS(TEMP_DICT)

    return param_metrics


def metrics_qs_report(pf_or_pf_path, remove_non_returns=True,
                      ):
    from vectorbtpro import Portfolio
    CHECKTEMPS(TEMP_DICT)
    # Note: if you have multiple portfolios, you can use the same function in a loop or combine them prior
    # Check if pf_or_pf_path is a path or a pf
    if isinstance(pf_or_pf_path, str):
        pf = Portfolio.load(pf_or_pf_path)
    else:
        pf = pf_or_pf_path

    if remove_non_returns:
        # > Remove those combinations with zero trades< #
        total_trades = pf.get_trades(chunked=True).count()
        mask = total_trades[total_trades != 0].index

        pf = pf[mask] if len(mask) != 0 else pf

        if pf.wrapper.shape[1] == 0:
            logger.warning('Portfolio filtered completely out by total trades filter')
            return None
        else:
            logger.info(f'After total trades filter -> {pf.wrapper.shape[1]} strategies')

    # Get parameter combinations from the portfolio
    param_combinations = pf.wrapper.columns
    print(f'There are {len(param_combinations)} parameter combinations')
    # > Compute the metrics report for each parameter combination < #
    # ray.init(num_cpus=20)
    # pf_id = ray.put(pf)
    # results = ray.get(
    #     [parallel_compute_metrics_report.remote(pf_id, param_combination) for param_combination in param_combinations])
    # pf_metric_qs_report_df = pd.concat(results, axis=1)
    # pf_metric_qs_report_df.to_csv('pf_metric_qs_report_df.csv'))
    import os
    os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = '1'
    param_combinations = param_combinations
    CHECKTEMPS(TEMP_DICT)
    parametrized_compute_metrics_report = vbt.parameterized(compute_metrics_report,
                                                            merge_func='column_stack',
                                                            # n_chunks=np.floor(param_combinations.shape[0]/4).astype(int),
                                                            # n_chunks=5,
                                                            chunk_len='auto',
                                                            # chunk_len=1,
                                                            show_progress=True,
                                                            engine='ray',
                                                            init_kwargs={
                                                                # 'address': 'auto',
                                                                'num_cpus': 20,
                                                                # 'object_store_memory': 100 * 10 ** 9
                                                            }
                                                            )

    pf_metric_qs_report_df = parametrized_compute_metrics_report(
        pf=pf,
        unique_params=vbt.Param(param_combinations.values, name='unique_params')
    )
    CHECKTEMPS(TEMP_DICT)

    pf_metric_qs_report_df = pf_metric_qs_report_df.vbt.drop_redundant_levels(0)
    pf_metric_qs_report_df.columns = param_combinations
    pf_metric_qs_report_df = pf_metric_qs_report_df.transpose()
    pf_metric_qs_report_df = pf_metric_qs_report_df.vbt.sort_index()
    # pf_metric_qs_report_df.to_csv(f"temp_csv/pf_metrics_report_all.csv")

    # Clean up RAM (mainly big items)
    del pf_metric_qs_report_df
    del pf
    del param_combinations
    gc.collect()
    CHECKTEMPS(TEMP_DICT)

    return pf_metric_qs_report_df

    # exit()
    # import driverlessai
    # dai = driverlessai.Client(address="https://steam.cloud.h2o.ai/proxy/driverless/3903/", username='ruben@moonshot.codes', password='xCAD8#sJf!w@b2c')
    # ds = dai.datasets.create(
    #     data='s3://h2o-public-test-data/smalldata/iris/iris.csv',
    #     data_source='s3',
    #     name='iris-getting-started'
    # )
    # print(ds)
    # dai.server.gui()

    # exit()
    # metrics_report = pf.qs.metrics_report(column=1)
    # print(metrics_report)
    # exit()


if __name__ == '__main__':
    from logger_tt import setup_logging, logger

    setup_logging(full_context=1)
    # import dask.dataframe as dd

    # bar_data = dd.read_csv("/home/ruben/PycharmProjects/mini_Genie_ML/Genie/Modules/temp_csv/pf_metrics_report_all.csv",
    #                        parse_dates=False, sample=100000000)
    # print(bar_data)
    # exit()
    # data = Data_Manager().fetch_csv_data_dask(data_file_name='pf_metrics_report_all.csv',
    #                                           n_rows=100,
    #                                           search_in=
    #                                           ["/home/ruben/PycharmProjects/mini_Genie_ML/Genie/Modules/temp_csv", "."])

    # print(data.head())

    # data = data.drop(level=0, axis=1)
    # print(data.head())
    # exit()
    # test_pf_paths = "/home/ruben/PycharmProjects/mini_Genie/Studies/Study_OILUSD/Portfolio/pf_*.pickle"
    test_pf_paths = "/home/ruben/pycharm_projects/mini_Genie_ML/Genie/Modules/backtest/Studies/US_Light_debugging_study/Portfolio/pf_*.pickle"
    logger.info(f"Loading portfolios from {test_pf_paths}")
    parametrized_metrics_qs_report = vbt.parameterized(metrics_qs_report,
                                                       merge_func="concat",
                                                       # n_chunks=np.floor(param_combinations.shape[0]/4).astype(int),
                                                       # n_chunks=np.floor(param_combinations.shape[0]/4).astype(int),
                                                       # chunk_len=1,
                                                       chunk_len='auto',
                                                       show_progress=True,
                                                       # engine='ray',
                                                       # init_kwargs={
                                                       #     # 'address': 'auto',
                                                       #     'num_cpus': 20,
                                                       #     # 'object_store_memory': 100 * 10 ** 9
                                                       # }
                                                       )
    # pf = ray_portfolio_load(
    #     vbt.Param(glob.glob(test_pf_path), name='pf_path'))
    # pf = vbt.Portfolio.column_stack(pf)

    metrics_df = parametrized_metrics_qs_report(pf_or_pf_path=vbt.Param(
        # np.random.choice(glob.glob(test_pf_paths), size=1, replace=False, p=None)
        glob.glob(test_pf_paths)[:2]
        , name='pf_path')
    )

    # Change index to range index
    logger.info(f"Changing index to range index")
    metrics_df = metrics_df.reset_index(drop=False)
    metrics_df = metrics_df.vbt.sort_index()

    # Drop pf_path column
    logger.info(f"Dropping pf_path column")
    metrics_df = metrics_df.drop(columns=['pf_path'])
    logger.info(f"Saving metrics_df to csv")
    metrics_df.to_csv(f"debugging_study_pf_metrics_report_all.csv")
