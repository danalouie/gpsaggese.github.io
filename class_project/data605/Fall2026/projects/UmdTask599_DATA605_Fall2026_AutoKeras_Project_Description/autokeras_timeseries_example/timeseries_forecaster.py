# ---
# jupyter:
#   jupytext:
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

# %% [markdown] colab_type="code"
# # Example Code From Autokeras Github Repository
#
# This notebook was downloaded from the autokeras version 1.1.0 github repository, and is provided as an example of `TimeSeriesForecaster`.
#
# AutoKeras Reference websites:
# - v1.1.0 github repository: https://github.com/keras-team/autokeras/tree/1.1.0
# - Example Notebooks: https://github.com/keras-team/autokeras/tree/1.1.0/docs/ipynb

# %%
# # !pip install autokeras

# %% colab_type="code"
import pandas as pd
import tensorflow as tf

import autokeras as ak

# %% [markdown] colab_type="text"
# To make this tutorial easy to follow, we use the UCI Airquality dataset, and try to
# forecast the AH value at the different timesteps. Some basic preprocessing has also
# been performed on the dataset as it required cleanup.
#
# ## A Simple Example
# The first step is to prepare your data. Here we use the [UCI Airquality
# dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality) as an example.
#

# %% colab_type="code"
# Modified original notebook by downloading UCI ZIP file directly and placing CSV file
# in same directory as this notebook.

# Filename:
filename = "AirQualityUCI.csv"

dataset = pd.read_csv(filename, sep=";")
dataset = dataset[dataset.columns[:-2]]
dataset = dataset.dropna()
dataset = dataset.replace(",", ".", regex=True)

val_split = int(len(dataset) * 0.7)
data_train = dataset[:val_split]
validation_data = dataset[val_split:]

data_x = data_train[
    [
        "CO(GT)",
        "PT08.S1(CO)",
        "NMHC(GT)",
        "C6H6(GT)",
        "PT08.S2(NMHC)",
        "NOx(GT)",
        "PT08.S3(NOx)",
        "NO2(GT)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
    ]
].astype("float64")

data_x_val = validation_data[
    [
        "CO(GT)",
        "PT08.S1(CO)",
        "NMHC(GT)",
        "C6H6(GT)",
        "PT08.S2(NMHC)",
        "NOx(GT)",
        "PT08.S3(NOx)",
        "NO2(GT)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
    ]
].astype("float64")

# Data with train data and the unseen data from subsequent time steps.
data_x_test = dataset[
    [
        "CO(GT)",
        "PT08.S1(CO)",
        "NMHC(GT)",
        "C6H6(GT)",
        "PT08.S2(NMHC)",
        "NOx(GT)",
        "PT08.S3(NOx)",
        "NO2(GT)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
    ]
].astype("float64")

data_y = data_train["AH"].astype("float64")

data_y_val = validation_data["AH"].astype("float64")

print(data_x.shape)  # (6549, 12)
print(data_y.shape)  # (6549,)

# %% [markdown] colab_type="text"
# The second step is to run the [TimeSeriesForecaster](/time_series_forecaster).
# As a quick demo, we set epochs to 10.
# You can also leave the epochs unspecified for an adaptive number of epochs.
#

# %% colab_type="code"
predict_from = 1
predict_until = 10
lookback = 3
clf = ak.TimeseriesForecaster(
    lookback=lookback,
    predict_from=predict_from,
    predict_until=predict_until,
    max_trials=1,
    objective="val_loss",
)
# Train the TimeSeriesForecaster with train data
clf.fit(
    x=data_x,
    y=data_y,
    validation_data=(data_x_val, data_y_val),
    batch_size=32,
    epochs=10,
)
# Predict with the best model(includes original training data).
predictions = clf.predict(data_x_test)
print(predictions.shape)
# Evaluate the best model with testing data.
print(clf.evaluate(data_x_val, data_y_val))

# %%
