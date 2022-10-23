import gc
import glob

import numpy as np
import pandas as pd
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


def get_returns_function(pf_or_pf_path, remove_non_returns=False,
                      resample_freq='1D'):
    gc.collect()
    from vectorbtpro import Portfolio
    CHECKTEMPS(TEMP_DICT)
    # Note: if you have multiple portfolios, you can use the same function in a loop or combine them prior
    # Check if pf_or_pf_path is a path or a pf
    if isinstance(pf_or_pf_path, str):
        pf = Portfolio.load(pf_or_pf_path)
    else:
        pf = pf_or_pf_path

    if remove_non_returns:
        # > Remove those combinations with zero/negative returns< #
        # pf_total_returns = pf.get_total_return(chunked=True)
        total_trades = pf.get_trades(chunked=True).count()
        mask = total_trades[total_trades != 0].index

        pf = pf[mask] if len(mask) != 0 else pf

        if pf.wrapper.shape[1] == 0:
            logger.warning('Portfolio filtered completely out by total returns filter')
            return None
        else:
            logger.info(f'After total returns filter -> {pf.wrapper.shape[1]} strategies')

    # Get parameter combinations from the portfolio
    param_combinations = pf.wrapper.columns[:3] #todo: remove this

    # > Compute the metrics report for each parameter combination < #
    CHECKTEMPS(TEMP_DICT)


    a=pf[param_combinations].resample(resample_freq).get_returns(chunked=True)
    return a

def get_stats_function(pf_or_pf_path, remove_non_returns=False,
                      ):
    gc.collect()
    from vectorbtpro import Portfolio
    CHECKTEMPS(TEMP_DICT)
    # Note: if you have multiple portfolios, you can use the same function in a loop or combine them prior
    # Check if pf_or_pf_path is a path or a pf
    if isinstance(pf_or_pf_path, str):
        pf = Portfolio.load(pf_or_pf_path)
    else:
        pf = pf_or_pf_path

    if remove_non_returns:
        # > Remove those combinations with zero/negative returns< #
        # pf_total_returns = pf.get_total_return(chunked=True)
        total_trades = pf.get_trades(chunked=True).count()
        mask = total_trades[total_trades != 0].index

        pf = pf[mask] if len(mask) != 0 else pf

        if pf.wrapper.shape[1] == 0:
            logger.warning('Portfolio filtered completely out by total returns filter')
            return None
        else:
            logger.info(f'After total returns filter -> {pf.wrapper.shape[1]} strategies')

    # Get parameter combinations from the portfolio
    param_combinations = pf.wrapper.columns#[:3]

    # > Compute the metrics report for each parameter combination < #
    CHECKTEMPS(TEMP_DICT)


    a=pf[param_combinations].resample('1D').stats(agg_func=None, group_by=False)
    a=a.replace([np.inf, -np.inf], 0, inplace=False)
    a=a.replace({pd.NaT: "0 days"}, inplace=False)
    return a



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
    test_pf_paths = "/home/ruben/pycharm_projects/mini_Genie_ML/Genie/Modules/backtest/Studies/MMT_RLGL_study/Portfolio/pf_*.pickle"

    parametrized_get_returns_stats_function = vbt.parameterized(get_stats_function,
                                                                merge_func="column_stack",
                                                                # n_chunks=np.floor(param_combinations.shape[0]/4).astype(int),
                                                                # n_chunks=np.floor(param_combinations.shape[0]/4).astype(int),
                                                                chunk_len=1,
                                                                # chunk_len='auto',
                                                                # engine='dask',
                                                                show_progress=True,
                                                                # init_kwargs={
                                                                #     # 'address': 'auto',
                                                                #     'num_cpus': 28,
                                                                #     # 'n_chunks':"auto",
                                                                #     # 'memory': 100 * 10 ** 9,
                                                                #     # 'object_store_memory': 100 * 10 ** 9,
                                                                # },
                                                                )
    # pf = ray_portfolio_load(
    #     vbt.Param(glob.glob(test_pf_path), name='pf_path'))
    # pf = vbt.Portfolio.column_stack(pf)

    metrics_df = parametrized_get_returns_stats_function(pf_or_pf_path=vbt.Param(
        # np.random.choice(glob.glob(test_pf_paths), size=2, replace=False, p=None)
        glob.glob(test_pf_paths)
        , name='pf_path')
    )

    # Change index to range index
    metrics_df = metrics_df.reset_index(drop=False)
    metrics_df = metrics_df.vbt.sort_index()

    # Drop pf_path column
    metrics_df = metrics_df.drop(columns=['pf_path'])

    # print all columns
    # pd.set_option('display.max_columns', None)
    gc.collect()

    print(f'metrics_df: \n{metrics_df.head(100)}\n{metrics_df.shape}')
    print('saving')
    metrics_df.to_csv(f"temp_pf_short_metrics_report_all.csv")
    print('saved')

    
    ##############################################################################################################

    exit()
    #todo!!!!!!!!!!remove [:3] from funciton get_returns_function!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    parametrized_get_returns_function = vbt.parameterized(get_returns_function,
                                                                merge_func="column_stack",
                                                                # n_chunks=np.floor(param_combinations.shape[0]/4).astype(int),
                                                                # n_chunks=np.floor(param_combinations.shape[0]/4).astype(int),
                                                                chunk_len=1,
                                                                # chunk_len='auto',
                                                                # engine='dask',
                                                                show_progress=True,
                                                                # init_kwargs={
                                                                #     # 'address': 'auto',
                                                                #     'num_cpus': 28,
                                                                #     # 'n_chunks':"auto",
                                                                #     # 'memory': 100 * 10 ** 9,
                                                                #     # 'object_store_memory': 100 * 10 ** 9,
                                                                # },
                                                                )
    # pf = ray_portfolio_load(
    #     vbt.Param(glob.glob(test_pf_path), name='pf_path'))
    # pf = vbt.Portfolio.column_stack(pf)

    returns_df = parametrized_get_returns_function(pf_or_pf_path=vbt.Param(
        np.random.choice(glob.glob(test_pf_paths), size=2, replace=False, p=None)
        # glob.glob(test_pf_paths)
        , name='pf_path')
    )


    returns_df = returns_df.vbt.drop_redundant_levels(0)

    print(returns_df)
    exit()

    # Change index to range index
    returns_df = returns_df.reset_index(drop=False)
    returns_df = returns_df.vbt.sort_index()

    # Drop pf_path column
    returns_df = returns_df.drop(columns=['pf_path'])

    # print all columns
    # pd.set_option('display.max_columns', None)
    gc.collect()

    print(f'metrics_df: \n{metrics_df.head(100)}\n{metrics_df.shape}')
    print('saving')
    metrics_df.to_csv(f"temp_pf_returns_report_all.csv")
    print('saved')