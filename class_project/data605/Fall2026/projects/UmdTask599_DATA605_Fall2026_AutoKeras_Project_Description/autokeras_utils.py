"""
template_utils.py

This file contains utility functions that support the tutorial notebooks.

- Notebooks should call these functions instead of writing raw logic inline.
- This helps keep the notebooks clean, modular, and easier to debug.
- Students should implement functions here for data preprocessing,
  model setup, evaluation, or any reusable logic.

Import as:

import class_project.project_template.template_utils as utils
"""

import logging
import os

from typing import Any, Optional, Tuple

import ipywidgets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats
import yfinance as yf
from IPython.display import clear_output, display

import helpers.hnotebook as hnotebo
import helpers.htutorial as htutori


_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    global _LOG
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)

# #############################################################################
# Function: download SPY data, compute volatility estimators, save to CSV
# #############################################################################

def build_volatility_timeseries(csv_path="data/spy_volatility_timeseries_2005_2025.csv"):
    """ Function performs the following tasks:
        - Downloads 2005-2025 SPY Open, High, Low, Close data as Pandas dataframe using yfinance
        - Computes Daily Log Return and adds corresponding column to dataframe
        - Computes Garman-Klass Variance Volatility estimator and adds column to dataframe
        - Creates "data" directory if it doesn't already exist
        - Saves file to default csv_path indicated
        Website References for this function (in order used):
        - makedirs: https://docs.python.org/3/library/os.html
        - yfinance download: 
          https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html#yfinance.download
        - Daily Log Return: https://365financialanalyst.com/knowledge-hub/corporate-finance/log-return/
        - Pandas .shift(): https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html
        - Garman-Klass variance equation: 
          https://portfoliooptimizer.io/blog/range-based-volatility-estimators-overview-and-examples-of-usage/#garman-klass-volatility-estimator

        
    """
    # Create target directory if it doesn't already exist
    parent_dir = os.path.dirname(csv_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Download 2005-2025 daily prices of SPY ETF; start set to 2004-12-31 so January has prior close date:
    raw_df = yf.download("SPY", 
                         start = "2004-12-31", 
                         end = "2026-01-01", 
                         auto_adjust=True, 
                         multi_level_index=False)

    # Copy open, high, low, and close values required for analysis:
    spy_df = raw_df[['Open', 'High', 'Low', 'Close']].copy()

    # Compute Daily Log Return and add as dataframe column
    spy_df['Log_return'] = np.log(spy_df['Close'] / spy_df['Close'].shift(1))

    # Compute Garman-Klass (GK) daily variance estimator and add as dataframe column
    ln_hl = np.log(spy_df['High'] / spy_df['Low'])
    ln_co = np.log(spy_df['Close'] / spy_df['Open'])
    spy_df['GK_variance'] = 0.5 * (ln_hl ** 2) - (2 * np.log(2) - 1) * (ln_co ** 2)

    # Drop initial 2004-12-31 row, where Log_return is NaN due the shift(1).
    spy_df = spy_df.dropna()

    # Save to CSV file in data_dir, and print summary statistics.
    spy_df.to_csv(csv_path)

    print(f"Saved {spy_df.shape[0]} rows and {spy_df.shape[1]} columns to: {csv_path}.\n") 
    print("\nSummary Statistics:")
    print(spy_df.describe())

    return spy_df

# #############################################################################
# Function: Load volatility timeseries data from CSV & print summary statistics.
# #############################################################################

def load_volatility_timeseries(csv_path="data/spy_volatility_timeseries_2005_2025.csv"):
    """ Loads 2005-2025 SPY volatility timeseries data and prints summary statistics. """
    # Check to ensure file exists at path.
    if not os.path.exists(csv_path): 
        raise FileNotFoundError(f"File not found: {csv_path}.") 
    
    spy_df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    
    print(f"Loaded {spy_df.shape[0]} rows and {spy_df.shape[1]} columns from: {csv_path}.") 
    print("\nSummary Statistics:")
    print(spy_df.describe())

    return spy_df

# #############################################################################
# Cell 1: Interactive Distribution Explorer with Plot Type Selection
# #############################################################################


# def cell1_interactive_distribution_explorer(
#     *,
#     figsize: Optional[Tuple[float, float]] = None,
# ) -> None:
#     """
#     Create interactive widget to explore Beta and Normal distributions.
# 
#     Demonstrates:
#     - Slider widgets for continuous parameters via build_widget_control()
#     - Dropdown for selecting distribution type
#     - Real-time plot updates using observe() callbacks
#     - Multiple synchronized plots (1x3 layout)
# 
#     :param figsize: Optional figure size (width, height). Defaults to
#         plt.rcParams["figure.figsize"]
#     """


# #############################################################################
# Cell 2: Interactive Sampling Visualization
# #############################################################################


# def cell2_interactive_sample_generator(
#     *,
#     figsize: Optional[Tuple[float, float]] = None,
# ) -> None:
#     """
#     Create interactive widget to generate and visualize random samples.
# 
#     Demonstrates:
#     - Multiple linked slider widgets via build_widget_control()
#     - Logarithmic scale slider for sample count via build_log_widget_control()
#     - Histogram visualization with theoretical overlay
#     - Sample statistics display
# 
#     :param figsize: Optional figure size (width, height). Defaults to
#         plt.rcParams["figure.figsize"]
#     """
