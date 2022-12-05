from Modules.Actors.mini_genie_vec_trader.mini_genie import mini_genie_assistant

def test_strategy(open,low,high,close,params):
    print("test_strategy")
    print("params", params)
    print("open", open.head())
    print("low", low.head())
    print("high", high.head())
    print("close", close.head())
    return True



if __name__ == "__main__":
    STUDY_NAME = "development_study"
    INPUT_DATA_PATHS = '/home/ruben/PycharmProjects/mini_Genie_ML/Datas/XAUUSD.csv'
    OUTPUT_DATA_OBJ_NAME = 'XAUUSD'
    Strategy_Settings = dict(
        strategy_path="/home/ruben/PycharmProjects/mini_Genie_ML/User_Strategies/RLGL_Strategy.py",
        strategy_function="RLGL_Strategy",
        strategy_input_params=['open', 'low', 'high', 'close', 'params'],
        # The order of parameter key_names should be honored across all files
        _pre_cartesian_product_filter=dict(
            function=rsi_params_filter,
            kwargs=dict(
                low_rsi=35,
                high_rsi=65
            )
        ),
        _post_cartesian_product_filter=dict(
            function=rlgl_post_cartesian_product_filter_function,
            kwargs=dict()
        ),

        parameter_settings=dict(
            rsi_timeframes=dict(type='timeframe', values=['5 min', '15 min', '30 min', '1h', '4h', '1d']),
            rsi_windows=dict(type='window', lower_bound=2, upper_bound=98, min_step=1),
            #
            sma_on_rsi_1_windows=dict(type='window', lower_bound=2, upper_bound=50, min_step=1),
            sma_on_rsi_2_windows=dict(type='window', lower_bound=5, upper_bound=70, min_step=1),
            sma_on_rsi_3_windows=dict(type='window', lower_bound=15, upper_bound=90, min_step=1),
            #
            T1_ema_timeframes=dict(type='timeframe', values=['1 min', '5 min', '15 min', '30 min', '1h', '4h']),
            T1_ema_1_windows=dict(type='window', lower_bound=2, upper_bound=65, min_step=1),
            T1_ema_2_windows=dict(type='window', lower_bound=15, upper_bound=70, min_step=1),
            #
            take_profit_points=dict(type='take_profit', lower_bound=1, upper_bound=10000000, min_step=100000),
            stop_loss_points=dict(type='stop_loss', lower_bound=1, upper_bound=10000000, min_step=100000),

        ),
    )
    MAX_INITIAL_COMBINATIONS= 1000

    # Prepate a New Study
    logger, study_dir_path, portfolio_dir_path, reports_dir_path, data_dir_path, misc_dir_path \
        = mini_genie_assistant.new_study(study_name=STUDY_NAME)

    # mini_genie_assistant.study_dir_path = study_dir_path

    # Load Data to Study
    data_obj = mini_genie_assistant.load_data_to_study(INPUT_DATA_PATHS)
    print(data_obj)

    # Save Data to Study
    mini_genie_assistant.save_data_to_study(
        study_name=STUDY_NAME,
        data_obj=data_obj,
        output_data_obj_name=OUTPUT_DATA_OBJ_NAME)

    # Load Strategy
    strategy = mini_genie_assistant.load_strategy(STRATEGY)


    exit()

    # Suggest Parameters
    mini_genie_assistant.suggest_parameters(
        strategy=strategy,
        data_obj=data_obj,
        max_initial_combinations=MAX_INITIAL_COMBINATIONS,
    )
