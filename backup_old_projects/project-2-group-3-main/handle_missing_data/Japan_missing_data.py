from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error



def Japan_regression(df,output_col, drop_col):
    # Mean imputations
    cols_to_impute = ["exchange_rate_USD_JY", "cpi_j", "interest_r_j", "inflation_j", "gdp_pc_j"]

    imputed_data = df.copy()
    imputer = SimpleImputer(strategy='mean')
    imputed_data[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])



    # -------------Preparation for Regression-----------------
    if drop_col is not None:
        df = imputed_data.drop(columns = drop_col) # need to drop this column because it contains lot of Nan values and we can compute them using regression later
    else:
        df = imputed_data

    # preparing data for train and test set, such that all the rows which do not contain NaN in our output column are considered for training and rest are considered for test_set
    mask_null = df[output_col].isnull()

    train_data = df[~mask_null]  # Rows without null values
    test_data = df[mask_null]  # Rows with null values

    # training set
    train_dates = pd.DataFrame(train_data.pop('date')) # saving date to merge the train set back later
    X_train = train_data.drop(columns = output_col)
    y_train = train_data[[output_col]]

    # test set
    test_dates = pd.DataFrame(test_data.pop('date')) # saving date to merge the test_set back later
    X_test = test_data.drop(columns = output_col)
    y_test = test_data[[output_col]]


    # ------------ Regression model ------------------
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    y_pred = regression_model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=[output_col])

    # print(y_pred.isnull().sum())

    # Resetting the index values of X_test and test_dates
    X_test.reset_index(drop=True, inplace=True)
    test_dates.reset_index(drop=True, inplace=True)

    # merging the test set and adding dates back to it
    merged_test = pd.concat([test_dates,X_test,y_pred],axis = 1)

    # merging train set and adding dates back to it
    merged_train = pd.concat([train_dates,X_train,y_train],axis = 1)

    # merging train and test set - which will give us a data set without missing values
    merged_df = pd.concat([merged_train,merged_test], axis = 0)

    return merged_df
