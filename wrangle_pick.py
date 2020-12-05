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

######### scale data functions to be adjusted for pick data ######
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

def scale(train, validate, test):
    """This function provides the inputs and runs the add_scaled_columns function"""
    train, validate, test = add_scaled_columns(
    train,
    validate,
    test,
    scaler=sklearn.preprocessing.MinMaxScaler(),
    columns_to_scale=['total_lines'],
    )
    return train, validate, test


###### wrangle pick data ######
def wrangle_pick_data():
    """
    This function takes acquired pick data, completes the prep
    and splits the data into train, validate, and test datasets
    """
    # get data
    df = acquire.run()
    # prep and split data
    train, validate, test = prepare.run(df)
    # scale and define target
    # not completed yet
    return train, validate, test
    #return scale_mall(train, validate, test)

def createXy(X_train, X_validate, X_test):
    '''
    This function splits the train, validate, and test sets for modeling and
    does the necessary preprocessing needed
    '''
    # need to drop observations with more than 1 box to reduce noise
    train_index = X_train[X_train.total_boxes > 1].index
    X_train.drop(train_index, inplace=True)
    val_index = X_validate[X_validate.total_boxes > 1].index
    X_validate.drop(val_index, inplace=True)
    test_index = X_test[X_test.total_boxes > 1].index
    X_test.drop(test_index, inplace=True)
    # create X_train_explore that still has target for analysis
    X_train_exp = X_train.copy()
    # split for modeling create X and y
    y_train = X_train[['pick_seconds']]
    y_validate = X_validate[['pick_seconds']]
    y_test = X_test[['pick_seconds']]    
    X_train = X_train.drop(columns=['pick_seconds'])
    X_validate = X_validate.drop(columns=['pick_seconds'])
    X_test = X_test.drop(columns=['pick_seconds'])
    return X_train_exp, X_train, X_validate, X_test, y_train, y_validate, y_test

def model_preprocess1(X_train, X_validate, X_test):
    '''
    This function drops columns not used as features and scales total lines for modeling
    and converts hour to boolean column is_hr_18.
    '''
    # get only the feature columns
    X_train_scaled = X_train[['total_lines', 'hour']]
    X_validate_scaled = X_validate[['total_lines', 'hour']]
    X_test_scaled = X_test[['total_lines', 'hour']]
    # scale total_lines
    X_train_scaled, X_validate_scaled, X_test_scaled = scale(X_train_scaled, X_validate_scaled, X_test_scaled)
    # create boolean for is hour 18
    X_train_scaled['is_hr_18'] = np.where(X_train_scaled.hour == 18, 1, 0)
    X_validate_scaled['is_hr_18'] = np.where(X_validate_scaled.hour == 18, 1, 0)
    X_test_scaled['is_hr_18'] = np.where(X_test_scaled.hour == 18, 1, 0)
    # drop non-scaled and original hour columns
    X_train_scaled = X_train_scaled.drop(columns=['total_lines', 'hour'])
    X_validate_scaled = X_validate_scaled.drop(columns=['total_lines', 'hour'])
    X_test_scaled = X_test_scaled.drop(columns=['total_lines', 'hour'])
    return X_train_scaled, X_validate_scaled, X_test_scaled

def model_preprocess2(X_train, X_validate, X_test):
    '''
    This function drops columns not used as features and scales total lines for modeling.
    Does not scale tenure bin because values are ordinal, or is part time because values are boolean.
    '''
    # get only the feature columns
    X_train_scaled = X_train[['total_lines', 'tenure_bin', 'is_part_time']]
    X_validate_scaled = X_validate[['total_lines', 'tenure_bin', 'is_part_time']]
    X_test_scaled = X_test[['total_lines', 'tenure_bin', 'is_part_time']]
    # scale total_lines
    X_train_scaled, X_validate_scaled, X_test_scaled = scale(X_train_scaled, X_validate_scaled, X_test_scaled)
    # drop non-scaled column
    X_train_scaled = X_train_scaled.drop(columns=['total_lines'])
    X_validate_scaled = X_validate_scaled.drop(columns=['total_lines'])
    X_test_scaled = X_test_scaled.drop(columns=['total_lines'])
    return X_train_scaled, X_validate_scaled, X_test_scaled