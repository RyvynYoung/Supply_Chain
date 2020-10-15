import pandas as pd
import numpy as np
import scipy as sp 
import os
import sklearn.preprocessing
# from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
import acquire
import prepare

def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    """This function scales the Telco2yr data"""
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(train[columns_to_scale])

    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    return train, validate, test

def scale_mall(train, validate, test):
    """This function provides the inputs and runs the add_scaled_columns function"""
    train, validate, test = add_scaled_columns(
    train,
    validate,
    test,
    scaler=sklearn.preprocessing.MinMaxScaler(),
    columns_to_scale=['spending_score', 'annual_income', 'age'],
    )
    return train, validate, test


def wrangle_pick_data():
    """
    This function takes acquired mall data, completes the prep
    and splits the data into train, validate, and test datasets
    """
    df = acquire.run()
    train, test, validate = prepare.run(df)
    #train_and_validate, test = train_test_split(df, test_size=.15, random_state=123)
    #train, validate = train_test_split(train_and_validate, test_size=.15, random_state=123)
    return train, test, validate
    #return scale_mall(train, validate, test)