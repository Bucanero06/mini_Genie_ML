#!/usr/bin/env python3
import ast
import gc
import inspect

import pandas as pd
import psutil

from mini_genie_source.Utilities import _typing as tp
from logger_tt import logger
import numpy as np

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

"""Array Handler"""


def randomly_sample_from_2d(arr, sample_size, replace=False):

    arr = np.array(arr)
    return arr[np.random.choice(arr.shape[0], sample_size, replace=replace)]

"""Data Handler"""


def mltidx_df_to_dict(df, first_n=None):
    return {k: v.droplevel(0)[:first_n if first_n else len(v)] for k, v in df.groupby(level=0)}


"""Range Splitter Handler"""


def _clean_up_range_split_ohlc_nb(split_data, n_splits):
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

    new_split_data = [[int(0) for x in range(n_splits)] for y in range(len(split_data))]

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
                      FEATURES   (asset_prices,datetime_index)  SPLITS
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


def convert_to_seconds(input_timeframe):
    '''
    Can accept any time frame as long as it is composed of a signle continuous interger and a timeframe recognized by pandas and convert the string timeframe into seconds.
    e.g. 1m,5 m,15min, 34 min, 1h, h4,etc ...
    '''
    seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 3600 * 24, "w": 3600 * 24 * 7}
    input_timeframe = input_timeframe.lower()
    # try to extract an integer string
    import re
    splits = re.split('(\d+)', input_timeframe)
    integer_str = splits[1]

    # try to extract the time intervel
    if splits[2].startswith('s'):
        timeframe_str = 's'
    elif splits[2].startswith('m'):
        timeframe_str = 'm'
    elif 'm' in splits[2]:
        timeframe_str = 'm'
    elif 'h' in splits[2]:
        timeframe_str = 'h'
    elif 'd' in splits[2]:
        timeframe_str = 'd'
    elif 'w' in splits[2]:
        timeframe_str = 'w'
    elif splits[2].startswith('h'):
        timeframe_str = 'h'
    elif splits[2].startswith('d'):
        timeframe_str = 'd'
    elif splits[2].startswith('w'):
        timeframe_str = 'w'
    else:
        print('unknown timeframe:', input_timeframe)
        print('please input a timeframe in the form of 1min, 5min, 1h, 60s, etc ....')
        return
    return int(integer_str) * seconds_per_unit[timeframe_str]


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


def next_path(path_pattern):
    import os

    """
    Finds the next free path in an sequentially named list of files

    e.g. path_pattern = 'file-%s.txt':

    file-1.txt
    file-2.txt
    file-3.txt

    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b


def fetch_pf_files_paths(study_dir):
    import glob

    targetPattern = r"%s/Portfolio/pf_*.pickle" % study_dir
    return glob.glob(targetPattern)


def load_n_combine_pf_objects_then_save(study_dir, target=None, new_pf_obj=None,
                                        output_pf_file_name='pf_combined',
                                        delete_old_pf_files=False,batch_size=10):
    """Load all pf objects from a study directory and combine them into a single pf object then save it to a new file.
    Does not modify the original pf objects and does not return any objects other than false if the operation failed
    and true if it succeeded. Has room for improvement, usefull but missing checks and error handling.

    Args:
        study_dir (str): Path to the study directory.
        target (str, optional): Path to the target file. Defaults to None/'all'.
        new_pf_obj (pf, optional): New pf object to be added to the combined pf object. Defaults to None.
        output_dir (str, optional): Name of the output file. Defaults to 'pf_combined'.
        delete_old_pf_files (bool, optional): Delete the old pf files after combining them. Defaults to False but it is
            recommended to set it to True when possible to save space.


        """
    from vectorbtpro import Portfolio
    import os
    import numpy as np

    if target is None or target == 'all':  # load all pf objects and combine them into a single object
        pf_paths = fetch_pf_files_paths(study_dir)  # .remove('pf_combined')

        try:
            logger.info(pf_paths)
            # remove if pf_combined is in the list of pf files
            old_target_path = os.path.join(study_dir, 'Portfolio', 'pf_combined.pickle')
            if old_target_path in pf_paths: pf_paths.remove(old_target_path)
            # todo this can be done in parallel as shown in other places in the code
            # pf_objects = [Portfolio.load(pf_path) for pf_path in pf_paths]

            # Get max n_chunks given max batch_size
            n_chunks = int(np.floor(len(pf_paths) / batch_size)) if batch_size < len(pf_paths) else 1
            # Split arrays into n_chunks
            chunked_pf_paths = np.array_split(pf_paths, n_chunks)
            #
            # from Utilities.general_utilities import put_objects_list_to_ray
            # chunks_ids_of_params_left_to_compute = put_objects_list_to_ray(chunks_of_params_left_to_compute)
            # for epoch_n, epoch_params_record_id in enumerate(chunks_of_params_left_to_compute):
            for chunk_id, pf_paths_chunk in enumerate(chunked_pf_paths):
                logger.info(f'Loading pf objects chunk {chunk_id + 1}/{n_chunks}')
                # Load this chunk's pf objects
                pf_objects_chunk = [Portfolio.load(pf_path) for pf_path in pf_paths_chunk]
                logger.info(f'Combining pf objects chunk {chunk_id + 1}/{n_chunks}')
                # Combine them into a single pf object
                pf_combined = Portfolio.column_stack(pf_objects_chunk)
                logger.info(f'Saving pf objects chunk {chunk_id + 1}/{n_chunks}')
                # Save the combined pf object
                pf_combined.save(os.path.join(study_dir, 'Portfolio', 'pf_combined_chunk_%s.pickle' % chunk_id))
                logger.info(f'pf_combined_chunk_{chunk_id} saved')

                # Clean RAM
                del pf_objects_chunk
                del pf_combined
                gc.collect()


            # Delete the old pf files
            if delete_old_pf_files:
                for pf_path in pf_paths:
                    os.remove(pf_path)

            return True

        except Exception as e:
            logger.error(e)
            return False
    elif new_pf_obj is not None:  # if new_pf_obj is not None then target must be a string
        try:
            # make sure target file exists and load it else save only the new_pf_obj as combined_pf
            target_pf_path = "%s/Portfolio/%s.pickle" % (study_dir, target)
            if os.path.exists(target_pf_path):
                pf_objects = [Portfolio.load(target_pf_path), new_pf_obj]
            else:
                pf_objects = [new_pf_obj]
        except Exception as e:
            logger.error(e)
            return False
    else:
        logger.error('please specify either target or new_pf_obj')
        return False

    '''Combine all pf objects into one'''
    try:
        if len(pf_objects) > 1:
            logger.info('combining pf objects')
            # combined_pf = Portfolio.combine(pf_objects)
            combined_pf = Portfolio.column_stack(pf_objects)
            logger.info('done combining pf objects')
        else:
            logger.info('only one pf object found, no need to combine')
            logger.info(f'{pf_objects  = }')
            combined_pf = pf_objects[0]

        '''Save the combined pf object'''
        combined_pf_path = "%s/Portfolio/%s" % (study_dir, output_pf_file_name)
        combined_pf.save(combined_pf_path + '.pickle')
    except Exception as e:
        logger.error(e)
        return False

    return True


def save_record_to_file(df, path_to_file, write_mode='w'):
    from os import path
    if path.exists(path_to_file) and write_mode == 'a':
        df.to_csv(path_to_file, mode=write_mode, header=False)
    else:
        df.to_csv(path_to_file)


'''Indicators Aid'''


def rsi_params_filter(params, low_rsi=40, high_rsi=60, **kwargs):
    # import numpy as np
    # a_1 = params["rsi_windows"][np.where(params["rsi_windows"] < kwargs.get('low_rsi', low_rsi))[0]]
    # a_2 = params["rsi_windows"][np.where(params["rsi_windows"] > kwargs.get('high_rsi', high_rsi))[0]]
    # a_3 = []
    # for a, b in zip(a_1, a_2):
    #     a_3.append(a)
    #     a_3.append(b)
    # params["rsi_windows"] = a_3
    logger.info(f'rsi_windows: {len(params["rsi_windows"])}')
    # del a_1, a_2, a_3
    return params


'''Equipment Aid'''


def auto_garbage_collect(pct=80.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return
