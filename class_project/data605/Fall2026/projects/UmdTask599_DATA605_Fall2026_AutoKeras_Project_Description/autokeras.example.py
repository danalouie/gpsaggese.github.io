# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Description
#
# This notebook examines ...
#
# The name of this notebook should follow the format: `<tool>.example.ipynb` (e.g., `pycaret.example.ipynb` for a pycaret example).
#
# References:
# - Add description of what the notebook does.
# - Point to references, e.g. (tool.example.md)
# - Add citations.
# - Keep the notebook flow clear.
# - Comments should be imperative and have a period at the end.
# - Your code should be well commented.
#
# Guide: https://github.com/causify-ai/helpers/blob/master/docs/coding/all.jupyter_notebook.how_to_guide.md
#
# New link: https://github.com/causify-ai/helpers/blob/master/docs/tools/notebooks/all.jupyter_notebook.how_to_guide.md

# %% [markdown]
# # 1. Import Packages and Load Financial Data
#
# ## 1.1. Import Packages

# %%
# %load_ext autoreload
# %autoreload 2

# System libraries.
import os
import logging
import shutil

# Third-party libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Following lines ensure inline plots render properly with installed version of matplotlib
if not hasattr(matplotlib.rcParams, "_get"):
    matplotlib.rcParams._get = matplotlib.rcParams.get

# Core packages installed for project.

import yfinance as yf
import tensorflow as tf
import autokeras as ak

# Import local functions:
import autokeras_utils as utils

# Use this for most notebooks.
import helpers.hdbg as hdbg
import helpers.hnotebook as hnotebook

_LOG = logging.getLogger(__name__)

# Initialize notebook configuration and logging.
hdbg.init_logger(verbosity=logging.INFO)
hnotebook.config_notebook()

# Set plots to default style:
plt.style.use("default")

# Convert `display` into `print()`.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

print("TF: ", tf.__version__, "AK: ", ak.__version__, "YF: ", yf.__version__)

# %%
# # To install additional packages, use:
# import helpers.hmodule as hmodule
# hmodule.install_module_if_not_present(
#     ["pycaret"],
#     use_activate=True,
#     use_sudo=False,
#     venv_path="/opt/venv",
# )

# %% [markdown]
# ## 1.2. Load Financial Data

# %%
# Load SPY dataframe from CSV file using function:

utils.load_volatility_timeseries()

# %%
# Remove directory from previous run of "time_series_forecaster", including the stale failed oracle folder
shutil.rmtree("./time_series_forecaster", ignore_errors=True)

# %% [markdown]
# ## Make the notebook flow clear
#
# Each notebook needs to follow a clear and logical flow, e.g:
# - Load data
# - Compute stats
# - Clean data
# - Compute stats
# - Do analysis
# - Show results


# %%
class Template:
    """
    Brief imperative description of what the class does in one line, if needed.
    """

    def __init__(self) -> None:
        """
        Initialize the Template class.
        """
        pass

    def method1(self, arg1: int) -> None:
        """
        Brief imperative description of what the method does in one line.

        You can elaborate more in the method docstring in this section, for e.g. explaining
        the formula/algorithm. Every method/function should have a docstring, typehints and include the
        parameters and return as follows:

        :param arg1: description of arg1
        :return: description of return
        """
        # Code blocks go here.
        # Make sure to include comments to explain what the code is doing.
        # No empty lines between code blocks.
        pass


def template_function(arg1: int) -> None:
    """
    Brief imperative description of what the function does in one line.

    You can elaborate more in the function docstring in this section, for e.g. explaining
    the formula/algorithm. Every function should have a docstring, typehints and include the
    parameters and return as follows:

    :param arg1: description of arg1
    :return: description of return
    """
    # Code blocks go here.
    # Make sure to include comments to explain what the code is doing.
    # No empty lines between code blocks.
    pass


# %% [markdown]
# ## The flow should be highlighted using headings in markdown
# ```
# # Level 1
# ## Level 2
# ### Level 3
# ```

# %% [markdown]
# # References
#
# - AutoKeras version 1.1.0 Github Repository: https://github.com/keras-team/autokeras/tree/1.1.0
# - AutoKeras Issue #1554 workaround: https://github.com/keras-team/autokeras/issues/1554
# - Jin, H., Chollet, F., Song, Q., & Hu, X. (2023). AutoKeras: An AutoML Library for Deep Learning. *Journal of Machine Learning Research*, *24*(6), 1–6. https://www.jmlr.org/papers/volume24/20-1355/20-1355.pdf
# - Keras CSV Logger:
#   - https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger
#   - https://keras.io/api/callbacks/csv_logger/
# - Pandas Series clip(): https://pandas.pydata.org/docs/reference/api/pandas.Series.clip.html
# - Portfolio Optimizer. (2023, September 20). *Range-Based Volatility Estimators: Overview and Examples of Usage*. https://portfoliooptimizer.io/blog/range-based-volatility-estimators-overview-and-examples-of-usage/#garman-klass-volatility-estimator
# - TensorFlow/Keras `Model.fit` (called under the hood by `TimeseriesForecaster.fit`): https://www.geeksforgeeks.org/deep-learning/model-fit-in-tensorflow/
#
#

# %%
