"""
preprocess dataset
"""
import pandas as pd


def read_data(csv_path):
    dtf = pd.read_csv(csv_path)

    # Everything here is taken from code of lesson 10
    numeric_columns = dtf.dtypes[(dtf.dtypes == "float64") | (dtf.dtypes == "int64")].index.tolist()
    very_numerical = [nc for nc in numeric_columns if dtf[nc].nunique() > 20]

    # Filling Null Values with the column's mean
    na_columns = dtf[very_numerical].isna().sum()
    na_columns = na_columns[na_columns > 0]
    for nc in na_columns.index:
        dtf[nc].fillna(dtf[nc].mean(), inplace=True)

    return dtf[very_numerical]
