import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

import statsmodels.api as sm
from statsmodels.tsa.api import Holt

import warnings
warnings.filterwarnings("ignore")

import prepare

######## time series #######
def split_data(df):
    '''splits into train, validate, test'''
    train_size = int(len(df) * .5)
    validate_size = int(len(df) * .3)
    test_size = int(len(df) - train_size - validate_size)
    validate_end_index = train_size + validate_size

    # split into train, validation, test
    train = df[: train_size]
    validate = df[train_size : validate_end_index]
    test = df[validate_end_index : ]
    # print the shape of each df
    train.shape, validate.shape, test.shape
    return train, validate, test

def sanity_check_split(df1, train, validate, test):
    '''checks train, validate, test splits'''
    # Does the length of each df equate to the length of the original df?
    print('df lengths add to total:', len(train) + len(validate) + len(test) == len(df1))
    # Does the first row of original df equate to the first row of train?
    print('1st row of full df == 1st row train:', df1.head(1) == train.head(1))
    # Is the last row of train the day before the first row of validate? And the same for validate to test?
    print('\n Is the last row of train the day before the first row of validate? And the same for validate to test?')
    print(pd.concat([train.tail(1), validate.head(1)]))
    print(pd.concat([validate.tail(1), test.head(1)]))
    # Is the last row of test the same as the last row of our original dataframe?
    print('\n Is the last row of test the same as the last row of our original dataframe?')
    print(pd.concat([test.tail(1), df1.tail(1)]))

def chart_splits(train, validate, test):
    for col in train.columns:
        plt.plot(train[col])
        plt.plot(validate[col])
        plt.plot(test[col])
        plt.ylabel(col)
        plt.title(col)
        plt.show()

def evaluate(target_var, validate, predictions):
    '''evaluate() will compute the Mean Squared Error and the Rood Mean Squared Error to evaluate'''
    rmse = round(sqrt(mean_squared_error(validate[target_var], predictions[target_var])), 0)
    return rmse

def plot_and_eval(target_var, train, validate, predictions):
    '''
    plot_and_eval() will use the evaluate function and also plot train and test values with the predicted
     values in order to compare performance.
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label='Train', linewidth=1)
    plt.plot(validate[target_var], label='Validate', linewidth=1)
    plt.plot(predictions[target_var])
    plt.title(target_var)
    rmse = evaluate(target_var, validate, predictions)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()

def create_eval_df():
    # Create empty dataframe to store model results for comparison
    eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])
    return eval_df

# function to store the rmse so that we can compare, note: need to run create_eval_df before this function
def append_eval_df(eval_df, validate, predictions, model_type, target_var):
    rmse = evaluate(target_var, validate, predictions)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)


def last_observed_predictions(train, validate):
    '''get last observed value and set that as prediction for all in validate'''
    consumption = train['Consumption'][-1:][0]
    wind = train['Wind'][-1:][0]
    solar = train['Solar'][-1:][0]
    calc_windsolar = train['calc_windsolar'][-1:][0]
    # get preditions
    yhat_df = pd.DataFrame({'Consumption': [consumption], 'Wind': [wind], 'Solar': [solar], 
                            'calc_windsolar': [calc_windsolar]}, index = validate.index)
    return yhat_df

def plot_pred_values(train, validate, predictions):
    # plot predicted values
    for col in train.columns:
        plot_and_eval(col, train, validate, predictions)


# create function for later repitition
def make_predictions(consumption, wind, solar, calc_windsolar, validate):
    yhat_df = pd.DataFrame({'Consumption': [consumption], 'Wind': [wind], 'Solar': [solar], 
                            'calc_windsolar': [calc_windsolar]}, index = validate.index)
    return yhat_df

def simple_average_predictions(train, validate):
    # get average and use that to make predictions for all in validate
    consumption = train['Consumption'].mean()
    wind = train['Wind'].mean()
    solar = train['Solar'].mean()
    calc_windsolar = train['calc_windsolar'].mean()
    yhat_df = make_predictions(consumption, wind, solar, calc_windsolar, validate)
    return yhat_df

def rolling_avg_pred(train, validate, period):
    # compute rolling average, 
    period = period
    consumption = round(train['Consumption'].rolling(period).mean().iloc[-1], 1)
    wind = round(train['Wind'].rolling(period).mean().iloc[-1], 1)
    solar = round(train['Solar'].rolling(period).mean().iloc[-1], 1)
    calc_windsolar = round(train['calc_windsolar'].rolling(period).mean().iloc[-1], 1)

    yhat_df = make_predictions(consumption, wind, solar, calc_windsolar, validate)
    return yhat_df

def multiple_periods(list_periods, train, eval_df, validate):
    periods = list_periods

    for p in periods:
        ROLL_pred = rolling_avg_pred(train, validate, period=p)
        model_type = str(p) + 'd moving average'
        for col in train.columns:
            eval_df = append_eval_df(eval_df, validate, ROLL_pred, model_type = model_type, target_var = col)

    return eval_df

def Holts_plot(train):
    for col in train.columns:
        print(col,'\n')
        _ = sm.tsa.seasonal_decompose(train[col].resample('D').mean()).plot()
        plt.show()
