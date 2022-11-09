import datetime

from Genie_API.genie_api_handler import ApiHandler as Genie_API_Handler

EXAMPLE_INPUT_DICT = dict(
    Genie=dict(
        study_name='RLGL_XAUUSD',
        # run_mode='legendary',
        run_mode='ludicrous',
        Strategy='mini_genie_source/Strategies/RLGL_Strategy.py',
        data_files_names=['XAUUSD'],
        # data_files_names=['OILUSD'],
        tick_size=[0.001],
        init_cash=1_000_000,
        size=100_000,
        start_date=datetime.datetime(month=3, day=4, year=2022),
        end_date=datetime.datetime(month=7, day=7, year=2022),
        #
        Continue=False,
        batch_size=2,
        timer_limit=None,
        stop_after_n_epoch=5,
        # max_initial_combinations=1_000_000_000,
        max_initial_combinations=1000,
        trading_fees=0.00005,  # 0.00005 or 0.005%, $5 per $100_000
        max_orders=1000,
    ),
    # Data_Manager=dict(
    #     report=True,
    # ),
    # Filters=dict(
    #     study_name='RLGL_AUDUSD',
    #     # study_name='Test_Study',
    #     # study_name='Study_OILUSD',
    #     Min_total_trades=1,
    #     Profit_factor=1.0,
    #     Expectancy=0.01,
    #     Daily_drawdown=0.05,
    #     Total_drawdown=0.1,
    #     Profit_for_month=0.1,
    #     Total_Win_Rate=0.03,
    #     quick_filters=True,
    #     # delete_loners=True,
    # ),
    # MIP=dict(
    #     agg=True,
    # ),
    # Neighbors=dict(
    #     n_neighbors=20,
    # ),
    # Data_Manager=dict(
    #     delete_first_month=True,
    # ),
    # Overfit=dict(
    #     cscv=dict(
    #         n_bins=10,
    #         objective='sharpe_ratio',
    #         PBO=True,
    #         PDes=True,
    #         SD=True,
    #         POvPNO=True,
    #     )
    # ),
    # Data_Manager_1=dict(  # todo add _{} parser
    #     n_split=2,
    # ),
    # Filters=dict(
    #     quick_filters=True,
    #     delete_loners=True,
    # ),
)

api_handler = Genie_API_Handler(EXAMPLE_INPUT_DICT)
api_handler.parse()
# print(api_handler.df[['Template_Code ', 'Variable_Value']])
api_handler.run()
