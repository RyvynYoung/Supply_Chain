import pandas as pd
import numpy as np
import scipy as sp 
import os
# from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

#######  Pick Data Prep #######
def operator_outliers(df):
    '''
    remove outliers in operator, create variable to hold list of removed values
    '''
    op_list = df.operator.value_counts()
    operdf = pd.DataFrame(op_list)
    operdf.reset_index()
    operdf = operdf.rename(columns={'index': 'name', 'operator': 'occur_times'})
    one_off = operdf[operdf.occur_times < 10].index
    keep_op_list = operdf[operdf.occur_times > 9].index
    # keep only operators with more than 9 occurances 
    df = df[df.operator.isin(keep_op_list)]
    return df

def prep_pick_data(df):
    '''
    Takes the acquired pick data, does data prep, and returns
    train, test, and validate data splits.
    '''
    # drop 28 observations with nulls
    df = df.dropna()

    # rename columns to more useful headings
    df = df.rename(columns={'PH_PICKEDB': 'operator', 'PH_PICKSTA': 'start_time', 'PH_PICKEND': 'end_time', 
                            'PH_TOTALLI': 'total_lines', 'PH_TOTALBO': 'total_boxes'})
    
    # convert start_time and end_time to date time
    df['start']= pd.to_datetime(df['start_time'])
    df['end']= pd.to_datetime(df['end_time'])   

    # # use the pd.get_dummies
    # df_dummies = pd.get_dummies(df[['gender']], drop_first=True)
    # df = pd.concat([df, df_dummies], axis=1)
    
    # Drop the redundant columns
    df = df.drop(columns=['start_time', 'end_time'])

    # add time calculation columns
    df['pick_time'] = df.end - df.start
    df['pick_seconds'] = df.pick_time.dt.total_seconds()
    df['int_day'] = df.start.dt.dayofweek
    df['day_name'] = df.start.dt.day_name()
    df['start_year'] = pd.DatetimeIndex(df['start']).year
    df['start_month'] = pd.DatetimeIndex(df['start']).month
    df['start_Y_M'] = pd.to_datetime(df['start']).dt.to_period('M')
    df['end_year'] = pd.DatetimeIndex(df['start']).year
    df['end_month'] = pd.DatetimeIndex(df['start']).month
    df['end_Y_M'] = pd.to_datetime(df['start']).dt.to_period('M')

    # run operator outlier removal before split because removal is based on domain knowledge
    df = operator_outliers(df)
    
    # run data outlier removal before split based on domain knowledge (want only 3 years with consistent range of volume)
    df = df[(~(df['start'] < '2016-01-01')) & (~(df['start'] > '2019-12-31'))]
    # split data
    train_and_validate, test = train_test_split(df, test_size=.15, random_state=123)
    train, validate = train_test_split(train_and_validate, test_size=.15, random_state=123)
    
    return train, test, validate  

#### NOTE: call the above with: train, test, validate = prep_pick_data(df)

def run(df):
    print("Prepare: Cleaning acquired data...")
    train, test, validate = prep_pick_data(df)
    print("Prepare: Completed!")
    return train, test, validate
