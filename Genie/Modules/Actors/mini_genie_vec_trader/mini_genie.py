#!/usr/bin/env python3.9
import numpy as np

DEFAULT_SETTINGS = dict(
    STUDY_DIRECTORY='Studies',
    DATA_DIRECTORY='Data',
    REPORTS_DIRECTORY='Reports',
    PORTFOLIO_DIRECTORY='Portfolios',
    MISC_DIRECTORY='Misc',
    LOGS_DIRECTORY='Logs',
    #
    #
    ACCEPTED_TF_TYPES=['timeframe', 'tf'],
    ACCEPTED_WINDOW_TYPES=['window', 'w'],
    ACCEPTED_TP_TYPES=['take_profit', 'tp'],
    ACCEPTED_SL_TYPES=['stop_loss', 'sl'],
    ACCEPTED_TP_SL_TYPES=['take_profit', 'tp'] + ['take_profit', 'tp'],

)


def _compute_n_initial_combinations_carefully(parameters, key_names, dictionary):  # fixme: naming is horrible
    """

    Args:
        dictionary:

    Returns:

    """
    n_reduced_lengths = [dictionary[f'{key_name}_length'] for key_name in key_names
                         if parameters[key_name][
                             'type'].lower() not in DEFAULT_SETTINGS['ACCEPTED_TP_SL_TYPES']]
    n_reduced_lengths.append(dictionary["tp_sl_length"])

    return np.product(n_reduced_lengths)


def _compute_windows_lengths_now(parameters, key_names, dictionary):
    """

    Args:
        dictionary:
        _keynames:

    Returns:
        object:

    """
    _keynames = key_names

    return np.product(
        [dictionary[f'{key_name}_length'] for key_name in _keynames if
         parameters[key_name]['type'].lower() in DEFAULT_SETTINGS['ACCEPTED_WINDOW_TYPES']])




class mini_genie_assistant:

    @staticmethod
    def new_study(
            study_name=None,
    ):
        # Get working directory aka study_dir_path
        study_dir_path = f'{DEFAULT_SETTINGS["STUDY_DIRECTORY"]}/{study_name}'

        # Set up Logging
        from logger_tt import setup_logging, logger
        setup_logging(full_context=1, log_path=f'{study_dir_path}/{study_name}_logs.txt')  # todo return logger
        logger.info(f'Creating new study: {study_name}')

        # Define Paths
        portfolio_dir_path = f'{study_dir_path}/{DEFAULT_SETTINGS["PORTFOLIO_DIRECTORY"]}'
        reports_dir_path = f'{study_dir_path}/{DEFAULT_SETTINGS["REPORTS_DIRECTORY"]}'
        data_dir_path = f'{study_dir_path}/{DEFAULT_SETTINGS["DATA_DIRECTORY"]}'
        misc_dir_path = f'{study_dir_path}/{DEFAULT_SETTINGS["MISC_DIRECTORY"]}'

        from Modules.Actors.mini_genie_vec_trader.Utilities.general_utilities import create_or_clean_directories
        create_or_clean_directories(study_dir_path, portfolio_dir_path, reports_dir_path, misc_dir_path, data_dir_path,
                                    delete_content=True)

        return logger, study_dir_path, portfolio_dir_path, reports_dir_path, data_dir_path, misc_dir_path

    @staticmethod
    def load_data_to_study(input_files_paths):
        if isinstance(input_files_paths, str):
            input_files_paths = [input_files_paths]

        file_names = [path.split("/")[-1] for path in input_files_paths]
        file_dir_paths = ["/".join(path.split("/")[:-1]) for path in input_files_paths]

        from Modules.Actors_Old._Data_Manager import Data_Manager
        data_obj = Data_Manager().fetch_data(data_file_names=file_names,
                                             data_file_dirs=file_dir_paths)

        return data_obj

    @staticmethod
    def save_data_to_study(study_name, data_obj, output_data_obj_name):
        data_obj.save(
            f'{DEFAULT_SETTINGS["STUDY_DIRECTORY"]}/{study_name}/{DEFAULT_SETTINGS["DATA_DIRECTORY"]}/{output_data_obj_name}')


    @staticmethod
    def load_strategy_to_study(strategy_params):


        return strategy_obj

    @staticmethod
    def suggest_parameters(
            strategy,
            data_obj,
            max_initial_combinations,
            ):
        from logger_tt import logger

        print("Suggesting parameters")
        print("strategy", strategy)
        print("ohlcv_dataframe", data_obj)
        print("max_initial_combinations", max_initial_combinations)

        #Load strategy
        from Modules.Actors.mini_genie_vec_trader.Utilities.general_utilities import load_strategy







