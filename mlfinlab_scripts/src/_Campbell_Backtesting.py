# Import MlFinLab tools
from _import_mlfinlab import *
from mlfinlab.backtest_statistics.backtests import CampbellBacktesting


class _Campbell_BackTesting:
    """
    This class is used to run the Campbell backtesting.
    """

    def __init__(self):
        """
        Initialize the class.
        """
        backtesting = CampbellBacktesting(simulations=2000)










if __name__ == "__main__":
    _Campbell_BackTesting()


# # Specify the desired number of simulations
# backtesting = CampbellBacktesting(4000)
#
# # In this example, annualized Sharpe ratio of 1, not adjusted to autocorrelation of returns
# # at 0.1, calculated on monthly observations of returns for two years (24 total observations),
# # with 10 multiple testing and average correlation among returns of 0.4
# haircuts = backtesting.haircut_sharpe_ratios(sampling_frequency='M', num_obs=24,
#                                              sharpe_ratio=1, annualized=True,
#                                              autocorr_adjusted=False, rho_a=0.1,
#                                              num_mult_test=10, rho=0.4)
#
# # Adjsuted Sharpe ratios by method used
# sr_adj_bonferroni = haircuts[1][0]
# sr_adj_holm = haircuts[1][1]
# sr_adj_bhy = haircuts[1][2]
# sr_adj_average = haircuts[1][3]
#


