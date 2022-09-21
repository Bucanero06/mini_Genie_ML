#!/usr/bin/env python3
import ast
import inspect

import numpy as np
import pandas as pd

from mini_genie_source.Utilities import _typing as tp

"""Expression Handler"""


def multiline_eval(expr: str, context: tp.KwargsLike = None) -> tp.Any:
    """Evaluate several lines of input, returning the result of the last line.

    Args:
        expr: The expression to evaluate.
        context: The context to evaluate the expression in.

    Returns:
        The result of the last line of the expression.

    Raises:
        SyntaxError: If the expression is not valid Python.
        ValueError: If the expression is not valid Python.
    """
    if context is None:
        context = {}
    tree = ast.parse(inspect.cleandoc(expr))
    eval_expr = ast.Expression(tree.body[-1].value)
    exec_expr = ast.Module(tree.body[:-1], type_ignores=[])
    exec(compile(exec_expr, "file", "exec"), context)
    return eval(compile(eval_expr, "file", "eval"), context)


"""Range Splitter Handler"""


def _clean_up_range_split_ohlc_nb(split_data,n_splits):
    # if len(split_data) == 1:
    #     new_split_data = np.empty(shape=n_splits)
    #     for j in range(n_splits):
    #         print(f'{j = }')
    #         print(split_data[0][0][0])
    #
    #         exit()
    #         new_split_data[j] = \
    #             comb_price_and_range_index(split_data[j], split_data[j])
    # #
    # else:

    new_split_data=[[int(0) for x in range(n_splits)] for y in range(len(split_data))]

    for i in range(len(split_data)):
        for j in range(n_splits):
            # print(f'{i = }')
            # print(f'{j = }')
            # split_data[i][0][j] = \
            #     comb_price_and_range_index(split_data[i][0][j], split_data[i][1][j])
            new_split_data[i][j] = comb_price_and_range_index(split_data[i][0][j], split_data[i][1][j])
    return new_split_data


def range_split_ohlc(data, columns, num_splits, range_len):
    """Split data into ranges of a given length.

    Args:
        data (pd.DataFrame): Data to split.
        n (int): Number of ranges to split into.
        range_len (int): Length of each range.

    Returns: (in a list of length columns each containing {see below})
        pd.DataFrame: Split data. (multi-indexed)
        np.ndarray: Indexes of each range.

        each of len(n_splits)
                      FEATURES   (prices,datetime_index)  SPLITS
    __prices_columns[feature_index](m in [0,1])[n in [0,n_splits]]


    """
    if isinstance(columns, str):
        columns = [columns]

    output = []
    for column in columns:
        output.append(data.get(column).vbt.range_split(n=num_splits, range_len=range_len))

    # return _clean_up_range_split_ohlc_nb(output,num_splits)
    return output


def comb_price_and_range_index(price_array, range_index):
    """Merge price and range index into a single dataframe.

    Args:
        price (pd.DataFrame): Price data.
        range_index (pd.DataFrame): Range index.

    Returns:
    Returns:
        pd.DataFrame: Merged data.
    """
    # Create Dataframe from price_array's data and range_index's index
    merged = pd.DataFrame(price_array).set_index(range_index)
    #
    return merged


"""Data Types Handling"""


def dict_to_namedtuple(d, add_prefix=None):
    """Convert a dictionary to a namedtuple."""
    # return namedtuple('kwargs', d.keys())(**d)
    from types import SimpleNamespace

    if add_prefix is not None:
        d = {add_prefix + k: v for k, v in d.items()}

    return SimpleNamespace(**d)


"""Directories Handling"""


def find_file(filename, *args):
    directories = [*args]
    foundfile = False
    from os import path
    for searchdirectory in directories:
        if path.exists(searchdirectory + "/" + filename):
            if searchdirectory == ".":
                print("Found " + str(filename) + " inside the current directory")
            else:
                print("Found " + str(filename) + " inside " + str(searchdirectory) + " directory")
            foundfile = True
            return searchdirectory

    # if not exited by now it means that the file was not found in any of the given directories thus rise error
    if foundfile != True:
        print(str(filename) + " not found inside " + str(directories) + "\n exiting...")
        exit()


def probe_all_subdirectories(directory):
    """Probe all subdirectories and return a list of all the directories found"""
    import os
    subdirectories = []

    def listdirs(rootdir):
        for file in os.listdir(rootdir):
            d = os.path.join(rootdir, file)
            if os.path.isdir(d):
                # print(d)
                subdirectories.append(d)
                listdirs(d)

    listdirs(directory)
    return subdirectories