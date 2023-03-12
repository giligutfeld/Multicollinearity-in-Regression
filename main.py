# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import TDS.plot
import TDS.dataset
import TDS.solution

# data processing
import pandas as pd
import numpy as np
import scipy as sp

# statistics
from scipy import stats
import statsmodels.api as sm

# data visualizations
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning library
import sklearn

import warnings

warnings.filterwarnings("ignore")


# save

# https://www.kaggle.com/datasets/mirichoi0218/insurance
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?select=CAR+DETAILS+FROM+CAR+DEKHO.csv
# https://data.world/nrippner/ols-regression-challenge

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dtf = TDS.dataset.read_data("Dataset/house_prices/train.csv")
    Ycol = 'SalePrice'
    from sklearn.linear_model import LinearRegression

    Y = dtf[Ycol]
    X = dtf.drop(Ycol, axis=1)



    sk_ols_model = LinearRegression()
    Y_pred = sk_ols_model.fit(X, Y).predict(X)

    TDS.plot.corr_map(X)
    plt.show()
    print(sklearn.metrics.r2_score(Y, Y_pred))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=Y_pred, y=Y, ax=ax)
    sns.lineplot(x=Y_pred, y=Y_pred, ax=ax, color='black')
    ax.set_xlabel("SalePrice")
    plt.show()


