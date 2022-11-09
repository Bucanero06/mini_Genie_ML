import gc
import glob

import numpy as np
import pandas as pd
import vectorbtpro as vbt

from Modules._Data_Manager import Data_Manager
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


def get_returns_function(pf_or_pf_path, remove_non_returns=True,
                         resample_freq=None):
    gc.collect()
    from vectorbtpro import Portfolio
    CHECKTEMPS(TEMP_DICT)
    # Note: if you have multiple portfolios, you can use the same function in a loop or combine them prior
    # Check if pf_or_pf_path is a path or a pf
    if isinstance(pf_or_pf_path, str):
        pf = Portfolio.load(pf_or_pf_path)
    else:
        pf = pf_or_pf_path

    # if remove_non_returns:
    #     # > Remove those combinations with zero/negative returns< #
    #     # pf_total_returns = pf.get_total_return(chunked=True)
    #     total_trades = pf.get_trades(chunked=True).count()
    #     mask = total_trades[total_trades != 0].index
    #
    #     pf = pf[mask] if len(mask) != 0 else pf
    #
    #     if pf.wrapper.shape[1] == 0:
    #         logger.warning('Portfolio filtered completely out by total returns filter')
    #         return None
    #     else:
    #         logger.info(f'After total returns filter -> {pf.wrapper.shape[1]} strategies')

    # Get parameter combinations from the portfolio
    param_combinations = pf.wrapper.columns  # [:3]  # todo: remove this

    # > Compute the metrics report for each parameter combination < #
    CHECKTEMPS(TEMP_DICT)

    if resample_freq is None:
        a = pf[param_combinations].get_returns(chunked=True)
    else:
        a = pf[param_combinations].resample(resample_freq).get_returns(chunked=True)

    # Clean up RAM
    # del pf
    # gc.collect()

    from vectorbtpro import riskfolio_optimize
    a.columns = a.columns.values
    optimized_portfolio = riskfolio_optimize(a, risk_free=0.0, target_return=0.0, target_risk=0.0)
    optimized_portfolio = pd.DataFrame(optimized_portfolio, index=optimized_portfolio.keys())
    print(optimized_portfolio)

    # print(a)
    # pf_opt = vbt.PortfolioOptimizer.from_pypfopt(
    #     prices=a,
    #     every="W",
    #     target=vbt.Param([
    #         "max_sharpe",
    #         "min_volatility",
    #         "max_quadratic_utility"
    #     ])
    # )

    # pf_opt.plot(column="min_volatility").show()
    # print('pf_opt')
    exit()
    return a


def get_stats_function(pf_or_pf_path, remove_non_returns=True,
                       resample_freq=None):
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
    param_combinations = pf.wrapper.columns  # [:3]

    # > Compute the metrics report for each parameter combination < #
    CHECKTEMPS(TEMP_DICT)

    if resample_freq is None:
        a = pf[param_combinations].resample(resample_freq).stats(agg_func=None, group_by=False)
    else:
        a = pf[param_combinations].stats(agg_func=None, group_by=False)
    a = a.replace([np.inf, -np.inf], 0, inplace=False)
    a = a.replace({pd.NaT: "0 days"}, inplace=False)

    # # Clean up RAM
    # del pf
    # gc.collect()

    return a


def get_returns(pf_paths=None, study_name=None, data_file_name=None, remove_non_returns=True, timeframe=None,
                n_random=None):
    parametrized_get_returns_function = vbt.parameterized(
        get_returns_function,
        merge_func="column_stack",
        # n_chunks=np.floor(param_combinations.shape[0]/4).astype(int),
        # n_chunks=np.floor(param_combinations.shape[0]/4).astype(int),
        chunk_len=1,
        # chunk_len='auto',
        # engine='ray',
        # show_progress=True,
        # init_kwargs={
        #     # 'address': 'auto',
        #     'num_cpus': 28,
        #     # 'n_chunks':"auto",
        #     # 'memory': 100 * 10 ** 9,
        #     # 'object_store_memory': 100 * 10 ** 9,
        # },
    )

    returns_df = parametrized_get_returns_function(pf_or_pf_path=vbt.Param(
        np.random.choice(glob.glob(pf_paths), size=n_random, replace=False, p=None) if isinstance(n_random,
                                                                                                  int) else glob.glob(
            pf_paths)
        # glob.glob(brent_pf_paths) + glob.glob(light_pf_paths) #+ glob.glob(gold_pf_paths)
        , name='pf_path'),
        remove_non_returns=remove_non_returns,
        resample_freq=timeframe

    )

    # Remove pf_path row level multiindex
    multiindex = returns_df.columns.droplevel('pf_path')
    returns_df.columns = multiindex

    # Combine column levels into one level
    # returns_df.columns = returns_df.columns.map('_'.join) # fix me  sequence item 1: expected str instance, int found
    # correct method
    returns_df.columns = returns_df.columns.values

    print(returns_df.head())

    # Load and add to dataframe the prices of the assets in the portfolio and match the index of the returns dataframe

    # Load prices
    prices_df = Data_Manager().fetch_csv_data_dask(
        data_file_name=data_file_name,
        # n_rows=100,
        search_in=
        ["/home/ruben/pycharm_projects/mini_Genie_ML/Genie/Modules/backtest/Datas", "."]).resample(
        timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'tick volume': 'sum'
    })

    # Print Index
    print(f'prices_df.index: {prices_df.index}')
    print(f'returns_df.index: {returns_df.index}')

    # Match index
    prices_df = prices_df.loc[returns_df.index]

    # Add prices to returns dataframe
    returns_df = returns_df.join(prices_df)

    print(returns_df.head())
    print(returns_df.index)

    gc.collect()

    print(f'returns_df: \n{returns_df.head(100)}\n{returns_df.shape}')
    print('saving')
    returns_df.to_csv(f"{study_name}_returns_prices_df.csv")
    print('saved')
    return returns_df


def get_metrics(pf_paths=None, study_name=None, remove_non_returns=True, timeframe=None, n_random=None):
    parametrized_get_returns_stats_function = vbt.parameterized(
        get_stats_function,
        merge_func="row_stack",
        # n_chunks=np.floor(param_combinations.shape[0]/4).astype(int),
        # n_chunks=np.floor(param_combinations.shape[0]/4).astype(int),
        # chunk_len=4,
        chunk_len='auto',
        engine='ray',
        show_progress=True,
        init_kwargs={
            # 'address': 'auto',
            'num_cpus': 28,
            # 'n_chunks':"auto",
            # 'memory': 100 * 10 ** 9,
            'object_store_memory': 50 * 10 ** 9,
            'clear_cache': True,
            'collect_garbage': True
        },
    )

    metrics_df = parametrized_get_returns_stats_function(pf_or_pf_path=vbt.Param(
        np.random.choice(glob.glob(pf_paths), size=n_random, replace=False, p=None) if isinstance(n_random,
                                                                                                  int) else glob.glob(
            pf_paths)
        , name='pf_path',
    ),
        remove_non_returns=remove_non_returns,
        resample_freq=timeframe
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
    metrics_df.to_csv(f"{study_name}_metrics_report_all.csv")
    print('saved')


def run_main(
        study_names=None,
        data_file_names=None,
        timeframe=None,
        remove_non_returns=True,
        action_get_returns=False,
        action_get_metrics=True,
        n_random=None
):
    for study_name, data_file_name in zip(study_names, data_file_names):
        pf_paths = f"/home/ruben/pycharm_projects/mini_Genie_ML/Genie/Modules/backtest/Studies/{study_name}/Portfolio/pf_*.pickle"

        if action_get_returns:
            logger.info(f'Getting returns for {study_name}')
            get_returns(pf_paths=pf_paths, study_name=study_name, data_file_name=data_file_name,
                        remove_non_returns=remove_non_returns, timeframe=timeframe, n_random=2)

        if action_get_metrics:
            logger.info(f'Getting metrics for {study_name}')
            get_metrics(pf_paths=pf_paths, study_name=study_name, remove_non_returns=remove_non_returns,
                        timeframe=timeframe, n_random=n_random)

    # Combine all metrics reports
    metrics_reports = glob.glob(f"*_metrics_report_all.csv")
    metrics_reports_df = pd.concat([pd.read_csv(metrics_report) for metrics_report in metrics_reports])
    metrics_reports_df = metrics_reports_df.drop(columns=['Unnamed: 0'])
    metrics_reports_df = metrics_reports_df.reset_index(drop=False)
    metrics_reports_df = metrics_reports_df.vbt.sort_index()
    metrics_reports_df = metrics_reports_df.drop(columns=['index'])
    metrics_reports_df.to_csv(f"all_assets_metrics_reports_df.csv")

def run_arg_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_names", nargs='+', type=str, default=None)
    parser.add_argument("--data_file_names", nargs='+', type=str, default=None)
    parser.add_argument("--timeframe", type=str, default='1T')
    parser.add_argument("--remove_non_returns", type=bool, default=True)
    parser.add_argument("--action_get_returns", type=bool, default=False)
    parser.add_argument("--action_get_metrics", type=bool, default=True)
    parser.add_argument("--n_random", type=int, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    from logger_tt import setup_logging, logger

    setup_logging(full_context=1)

    args = run_arg_parser()

    run_main(
        study_names=args.study_names,
        data_file_names=args.data_file_names,
        timeframe=args.timeframe,
        remove_non_returns=args.remove_non_returns,
        action_get_returns=args.action_get_returns,
        action_get_metrics=args.action_get_metrics,
        n_random=args.n_random
    )


