"""Example Strategies"""
from mini_Genie.mini_genie_source import Strategies  # noqa: F401 Contains a few pre-made strategies

"""Data Handling"""
from Legendary_Genie._Data_Manager import Data_Manager  # noqa: F401

"""Strategy Reports"""
from Overfitting import walkfoward_report  # noqa: F401

"""Utilities"""
import Utils  # noqa: F401

"""Third Party Libraries"""
# This was done to make the code more readable since the mlfinlab_src
# package is used a lot, revomes the need to import the KEY the biggest of the resons is that they do not allow their
# paid users to view the source code. Thus, we were using a combination of the source code and the documentation to
# create pass through functions which do contain docstring for every function and commented out what I believe their
# source code to actually be. This is a solution until we decide whether the enterprice licence is worth it
# or not. #todo might not be working yet
from Legendary_Genie.mlfinlab_src import *  # noqa: F401
# from Legendary_Genie import mlfinlab_src  # noqa: F401
import mlfinlab as mlf  # noqa: F401 # This is to directly access the mlfinlab_src package
from mlfinlab import *  # noqa: F401 # This is to directly access the mlfinlab_src package
#
import vectorbtpro as vbt  # noqa: F401 This is to directly access the vectorbtpro package
from vectorbtpro import *  # noqa: F401 Easy to use backtesting library

vbt.settings.set_theme("dark")
#
#
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
import datetime as datetime  # noqa: F401
