import sys

import numpy as np
from logger_tt import logger


def _initiate_parameters_records(self, add_ids=None, initial_params_size=None):
    def _total_possible_values_in_window(lower, upper, step):
        return int((upper - lower) / step)

    def _reduce_initial_parameter_space(lengths_dict, max_initial_combinations):
        """
        Reduce initial parameter space combinations by reducing the number of param suggestions:
            1. For TP and SL parameters
            2. For Windowed parameters
        Output:
            n_initial_combinations to be used in the construction of the initial parameters record

        Args:
            lengths_dict:
            max_initial_combinations:

        Returns:
            object:
        """

        def _compute_n_initial_combinations_carefully(dictionary):  # fixme: naming is horrible
            """

            Args:
                dictionary:

            Returns:

            """
            n_reduced_lengths = [dictionary[f'{key_name}_length'] for key_name in self.key_names
                                 if self.parameter_windows[key_name][
                                     'type'].lower() not in self.ACCEPTED_TP_SL_TYPES]
            n_reduced_lengths.append(dictionary["tp_sl_length"])

            return np.product(n_reduced_lengths)

        def _compute_windows_lengths_now(dictionary, _keynames=None):
            """

            Args:
                dictionary:
                _keynames:

            Returns:
                object:

            """
            _keynames = self.key_names if not _keynames else _keynames

            return np.product(
                [dictionary[f'{key_name}_length'] for key_name in _keynames if
                 self.parameter_windows[key_name]['type'].lower() in self.ACCEPTED_WINDOW_TYPES])

        # if n_initial_combinations > max_initial_combinations: reduce initial search space
        if lengths_dict["n_initial_combinations"] > max_initial_combinations:
            # First try the reduced TP and SL space

            lengths_dict['tp_sl_length'] = len(self.tp_sl_selection_space["n_ratios"]) * len(
                self.tp_sl_selection_space["gamma_ratios"]) * self.tp_sl_selection_space[
                                               "number_of_bar_trends"] * len(
                lengths_dict["all_tf_in_this_study"])
            #
            lengths_dict["n_initial_combinations"] = _compute_n_initial_combinations_carefully(lengths_dict)
            lengths_dict["using_reduced_tp_sl_space"] = True
        #

        if lengths_dict["n_initial_combinations"] > max_initial_combinations:
            # Second try the reduced windowed space
            temp_lengths_dict = dict(
                windows_product=1,
                window_keynames_to_be_reduced=[],
            )
            from copy import deepcopy
            for key_name in self.window_key_names:
                if lengths_dict[f'{key_name}_length'] > 1:
                    temp_lengths_dict[f'{key_name}_length'] = deepcopy(lengths_dict[f'{key_name}_length'])
                    #
                    temp_lengths_dict[f'windows_product'] = temp_lengths_dict[f'windows_product'] * \
                                                            temp_lengths_dict[f'{key_name}_length']
                    #
                    temp_lengths_dict["window_keynames_to_be_reduced"].append(key_name)

            temp_lengths_dict["big_r_scaling_factor"] = lengths_dict[
                                                            "n_initial_combinations"] / max_initial_combinations
            temp_lengths_dict[f'max_windows_product'] = temp_lengths_dict[
                                                            f'windows_product'] / temp_lengths_dict[
                                                            "big_r_scaling_factor"]

            temp_lengths_dict["small_r_scaling_factor"] = temp_lengths_dict["big_r_scaling_factor"] ** (
                    1 / len(temp_lengths_dict["window_keynames_to_be_reduced"]))

            # we_are_good = all(
            #     temp_lengths_dict[f'{key_name}_length'] > temp_lengths_dict["small_r_scaling_factor"] for
            #     key_name in
            #     temp_lengths_dict["window_keynames_to_be_reduced"])
            # logger.info(f'{temp_lengths_dict["window_keynames_to_be_reduced"] = }')
            #                 logger.info(f'{we_are_good = }')
            #                 logger.info(f'{_compute_n_initial_combinations_carefully(temp_lengths_dict) = }')
            #
            #
            #                 we_are_good=_compute_n_initial_combinations_carefully(temp_lengths_dict)

            # Refine small_r_scaling_factor
            we_are_good = False
            temp_lengths_dict = dict(
                big_r_scaling_factor=temp_lengths_dict["big_r_scaling_factor"],
                small_r_scaling_factor=temp_lengths_dict["small_r_scaling_factor"],
                max_windows_product=temp_lengths_dict[f'max_windows_product'],
                windows_product=1,
                n_windows_to_be_reduced=0,
                window_keynames_to_be_reduced=[],
                window_keynames_above_1=temp_lengths_dict["window_keynames_to_be_reduced"],
            )
            while not we_are_good:
                temp_lengths_dict = dict(
                    big_r_scaling_factor=temp_lengths_dict["big_r_scaling_factor"],
                    small_r_scaling_factor=temp_lengths_dict["small_r_scaling_factor"],
                    max_windows_product=temp_lengths_dict[f'max_windows_product'],
                    windows_product=1,
                    n_windows_to_be_reduced=0,
                    window_keynames_to_be_reduced=[],
                    window_keynames_above_1=temp_lengths_dict["window_keynames_above_1"],
                )
                #
                for key_name in temp_lengths_dict["window_keynames_above_1"]:
                    if lengths_dict[f'{key_name}_length'] > temp_lengths_dict["small_r_scaling_factor"]:
                        temp_lengths_dict[f'{key_name}_length'] = deepcopy(lengths_dict[f'{key_name}_length'])
                        #
                        temp_lengths_dict[f"windows_product"] = temp_lengths_dict[f'windows_product'] * \
                                                                temp_lengths_dict[f'{key_name}_length']
                        #
                        temp_lengths_dict["window_keynames_to_be_reduced"].append(key_name)
                    #
                if temp_lengths_dict["window_keynames_to_be_reduced"]:
                    #
                    temp_lengths_dict["small_r_scaling_factor"] = temp_lengths_dict["big_r_scaling_factor"] ** (
                            1 / len(temp_lengths_dict["window_keynames_to_be_reduced"]))
                    #
                    we_are_good = all(
                        temp_lengths_dict[f'{key_name}_length'] > temp_lengths_dict["small_r_scaling_factor"]
                        for key_name in
                        temp_lengths_dict["window_keynames_to_be_reduced"])
                    #
                else:
                    max_initial_combinations = max_initial_combinations + (max_initial_combinations * 0.01)
                    we_are_good = False
                    #
                    temp_lengths_dict["big_r_scaling_factor"] = lengths_dict[
                                                                    "n_initial_combinations"] / max_initial_combinations
                    temp_lengths_dict["small_r_scaling_factor"] = temp_lengths_dict["big_r_scaling_factor"] ** (
                            1 / len(temp_lengths_dict["window_keynames_above_1"]))
                    temp_lengths_dict[f'max_windows_product'] = temp_lengths_dict[f'windows_product'] / \
                                                                temp_lengths_dict["big_r_scaling_factor"]
                    #

            # Scale down lengths
            for key_name in temp_lengths_dict["window_keynames_to_be_reduced"]:
                #
                temp_value = lengths_dict[f'{key_name}_length'] / temp_lengths_dict["small_r_scaling_factor"]
                if temp_value < 1:
                    temp_lengths_dict[f'{key_name}_length'] = 1
                else:
                    temp_lengths_dict[f'{key_name}_length'] = temp_value
            #
            # Redefine window lengths in length_dict
            for key_name in temp_lengths_dict["window_keynames_to_be_reduced"]:
                # if self.parameter_windows[key_name]['type'].lower() in self.ACCEPTED_WINDOW_TYPES:
                lengths_dict[f'{key_name}_length'] = int(temp_lengths_dict[f'{key_name}_length'])

            lengths_dict["using_reduced_window_space"] = True
            temp_lengths_dict["windows_product"] = _compute_windows_lengths_now(lengths_dict,
                                                                                _keynames=temp_lengths_dict[
                                                                                    "window_keynames_to_be_reduced"])

        lengths_dict["n_initial_combinations"] = _compute_n_initial_combinations_carefully(lengths_dict)
        #
        return lengths_dict

    parameters_record_dtype = []

    # Keep track of miscellaneous settings for reducing the space
    parameters_lengths_dict = dict(
        using_reduced_tp_sl_space=False,
        using_reduced_window_space=False,
        n_total_combinations=0,
        n_initial_combinations=0,
        all_tf_in_this_study=[],
        tp_sl_length=1,
    )

    if add_ids:
        parameters_record_dtype.append(('trial_id', np.int_))
    for key_name in self.key_names:
        parameters_lengths_dict[f'{key_name}_length'] = 0

        if self.parameter_windows[key_name]["type"].lower() in self.ACCEPTED_TF_TYPES:
            tf_in_key_name = [tf.lower() for tf in self.parameter_windows[key_name]['values']]

            if set(tf_in_key_name).issubset(set(self.ACCEPTED_TIMEFRAMES)):
                parameters_record_dtype.append((key_name, 'U8'))
            else:
                erroneous_timeframes = [tf for tf in tf_in_key_name if tf not in self.ACCEPTED_TIMEFRAMES]
                logger.error(
                    f'These timeframes provided are not accepted or understood, please revise.\n'
                    f'      {erroneous_timeframes = }.\n'
                    f'      {self.ACCEPTED_TIMEFRAMES = }.\n'
                    f'      {tf_in_key_name = }.\n'
                )

            parameters_lengths_dict[f'{key_name}_length'] = len(tf_in_key_name)
            parameters_lengths_dict["all_tf_in_this_study"].extend(tf_in_key_name)


        elif self.parameter_windows[key_name]["type"].lower() in self.ACCEPTED_WINDOW_TYPES:
            if isinstance(self.parameter_windows[key_name]['min_step'], int):
                parameters_record_dtype.append((key_name, 'i4'))
            elif isinstance(self.parameter_windows[key_name]['min_step'], float):
                parameters_record_dtype.append((key_name, 'f4'))
            else:
                logger.error(f'Parameter {key_name} is defined as type window but inputs are inconsistent.\n'
                             f'     (e.g. -> Either lower_bound or upper_bound is a float => float)\n'
                             f'     (e.g. -> both lower_bound or upper_bound are integers => int)\n'
                             f' upper_bound type:   {type(self.parameter_windows[key_name]["upper_bound"])}'
                             f' lower_bound type:   {type(self.parameter_windows[key_name]["lower_bound"])}'
                             )
                sys.exit()
            parameters_lengths_dict[f'{key_name}_length'] = _total_possible_values_in_window(
                self.parameter_windows[key_name]["lower_bound"], self.parameter_windows[key_name]["upper_bound"],
                self.parameter_windows[key_name]["min_step"])

        elif self.parameter_windows[key_name]["type"].lower() in self.ACCEPTED_TP_SL_TYPES:
            if isinstance(self.parameter_windows[key_name]['min_step'], int):
                parameters_record_dtype.append((key_name, 'i4'))
            elif isinstance(self.parameter_windows[key_name]['min_step'], float):
                parameters_record_dtype.append((key_name, 'f4'))
            else:
                logger.error(f'Parameter {key_name} is defined as type window but inputs are inconsistent.\n'
                             f'     (e.g. -> Either lower_bound or upper_bound is a float => float)\n'
                             f'     (e.g. -> both lower_bound or upper_bound are integers => int)\n'
                             f' upper_bound type:   {type(self.parameter_windows[key_name]["upper_bound"])}'
                             f' lower_bound type:   {type(self.parameter_windows[key_name]["lower_bound"])}'
                             )
                sys.exit()
            #
            parameters_lengths_dict[f'{key_name}_length'] = _total_possible_values_in_window(
                self.parameter_windows[key_name]["lower_bound"], self.parameter_windows[key_name]["upper_bound"],
                self.parameter_windows[key_name]["min_step"])

            if not self.user_pick:
                parameters_lengths_dict[f'tp_sl_length'] = parameters_lengths_dict[f'tp_sl_length'] * \
                                                           parameters_lengths_dict[f'{key_name}_length'] if \
                    parameters_lengths_dict[f'tp_sl_length'] else parameters_lengths_dict[f'{key_name}_length']

        else:
            logger.error(f'Parameter {key_name} is defined as type {self.parameter_windows[key_name]["type"]}'
                         f'but that type is not accepted ... yet ;)')
            sys.exit()

    if not self.user_pick:
        parameters_lengths_dict["all_tf_in_this_study"] = list(set(parameters_lengths_dict["all_tf_in_this_study"]))
        # Determine size of complete parameter space combinations with settings given as well reduced space
        parameters_lengths_dict["n_total_combinations"] = parameters_lengths_dict["n_initial_combinations"] = \
            np.product([parameters_lengths_dict[f'{key_name}_length'] for key_name in self.key_names])

        '''Get record dimensions'''
        if not initial_params_size:
            self.parameters_lengths_dict = _reduce_initial_parameter_space(parameters_lengths_dict,
                                                                           self.runtime_settings[
                                                                               "Simulation_Settings.Initial_Search_Space.max_initial_combinations"])
        else:
            parameters_lengths_dict["n_initial_combinations"] = initial_params_size
            self.parameters_lengths_dict = parameters_lengths_dict

        if self.parameters_lengths_dict["n_initial_combinations"] > self.runtime_settings[
            "Simulation_Settings.Initial_Search_Space.max_initial_combinations"]:
            continuing_text = f' because we are continuing the study from file' if self.continuing else ' '
            logger.warning(
                f'I know max_initial_combinations was set to '
                f'{self.runtime_settings["Simulation_Settings.Initial_Search_Space.max_initial_combinations"]} '
                f'but, I needed at least {self.parameters_lengths_dict["n_initial_combinations"]} '
                f'initial combinations{continuing_text}'
                f"\N{smiling face with smiling eyes}"
            )

        self.parameter_data_dtype = np.dtype(parameters_record_dtype)
        # FIXME exploring initiating it after elminating some parameter to reduce memory usage
        if self.continuing:
            self.parameters_record = np.empty(self.parameters_lengths_dict["n_initial_combinations"],
                                              dtype=self.parameter_data_dtype)
        # #
    else:
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.parameter_windows = self.runtime_settings[
            f"Strategy_Settings.strategy_user_picked_params.parameter_windows"]
        #
        '''Prepare'''
        # Determine Parameter Combinations (returns either a list of tuple params or a df of params)
        n_initial_combinations, initial_param_combinations = self._define_backtest_parameters()
        #
        # Create empty parameter record
        self.parameter_data_dtype = np.dtype(parameters_record_dtype)
        self.parameters_record = np.empty(n_initial_combinations, dtype=self.parameter_data_dtype)
        #
        # Fill parameter records
        if not self.user_defined_param_file_bool:  # (List of param combination tuples)
            for index in range(len(initial_param_combinations)):
                value = ((index,) + tuple(initial_param_combinations[index]))
                self.parameters_record[index] = value
        else:
            # fixme: left here
            self.parameters_record["trial_id"] = np.arange(0, len(initial_param_combinations), 1)
            #
            for key_name, values in initial_param_combinations.items():
                self.parameters_record[key_name] = values
        #
        logger.info(f'Total number of combinations to run backtest on -->  {len(self.parameters_record)}\n'
                    f'  on {len(self.asset_names)} assets')

        from mini_genie_source.Utilities.general_utilities import delete_non_filled_elements
        self.parameters_record = delete_non_filled_elements(self.parameters_record)
