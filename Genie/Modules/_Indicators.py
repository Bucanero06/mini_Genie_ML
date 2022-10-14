def _add_indicator_to_dataframe(df, indicator_series, prefix=None):
    """
    Add indicator series to dataframe. If prefix is provided, add it to the column name.
    :param df: (pd.DataFrame) Dataframe to add indicator to.
    :param indicator_series: (pd.Series) Indicator series to add to dataframe.
    :param prefix: (str) Prefix to add to column name.
    :return: (pd.DataFrame) Dataframe with indicator added.
    """
    if prefix is not None:
        indicator_series.name = f'{prefix}_{indicator_series.name}'
    return df.join(indicator_series)

#
# def _compute_wqa_n(price_series, n: int or list = 1):
#     import pandas as pd
#
#     if isinstance(n, int):
#         n = [n]
#     if isinstance(price_series, pd.Series):
#         price_series = price_series.to_frame()
#     price_series = price_series.copy()
#     import vectorbtpro as vbt
#
#     for i in price_series.columns:
#         for j in n:
#             price_series[f'wqa_{i}'] = vbt.wqa101(j).run(price_series[i]).out
#
#     return price_series
