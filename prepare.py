import pandas as pd
import numpy as np
import scipy as sp 
import os
# from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

######## #Clustering Exercises functions ##########
##### Zillow Clustering ########
def remove_columns(df, cols_to_remove):  
    '''
    Remove columns passed to function
    '''
    df = df.drop(columns=cols_to_remove)
    return df

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .5):
    '''
    Drops rows or columns based on the percent of values that are missing, prop_required_column = a number between 0 and 1
    that represents the proportion, for each column, of rows with non-missing values required to keep the column. 
    i.e. if prop_required_column = .6, then you are requiring a column to have at least 60% of values not-NA (no more than 40% missing).
    prop_required_row = a number between 0 and 1 that represents the proportion, for each row, of columns/variables with non-missing
     values required to keep the row. For example, if prop_required_row = .75, then you are requiring a row to have at least 75% of
    variables with a non-missing value (no more that 25% missing)
    '''
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

# note: Anthony has a different approach to handle missing values


def data_prep(df, cols_to_remove=[], prop_required_column=.5, prop_required_row=.5):
    '''
    Prep data by removing specificed columns as well as columns as rows with designated proportion of missing values.
    Remember: if prop_required_row = .75, then you are requiring a row to have at least 75% of
    variables with a non-missing value (no more that 25% missing)
    '''
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df


#######  Pick Data Prep #######
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
    
    train_and_validate, test = train_test_split(df, test_size=.15, random_state=123)
    train, validate = train_test_split(train_and_validate, test_size=.15, random_state=123)
    return train, test, validate  

#### NOTE: call the above with: train, test, validate = prep_pick_data(df)

def run(df):
    print("Prepare: Cleaning acquired data...")
    train, test, validate = prep_pick_data(df)
    print("Prepare: Completed!")
    return train, test, validate

def get_upper_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

def add_upper_outlier_columns(df, k):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    outlier_cols = {col + '_up_outliers': get_upper_outliers(df[col], k) for col in df.select_dtypes('number')}
    return df.assign(**outlier_cols)

    for col in df.select_dtypes('number'):
        df[col + '_up_outliers'] = get_upper_outliers(df[col], k)

    return df

def get_lower_outliers(s, k):
    '''
    Given a series and a cutoff value, k, returns the lower outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the lower bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    return s.apply(lambda x: max([x - lower_bound, 0]))

def add_lower_outlier_columns(df, k):
    '''
    Add a column with the suffix _outliers for all the numeric columns
    in the given dataframe.
    '''
    outlier_cols = {col + '_low_outliers': get_lower_outliers(df[col], k) for col in df.select_dtypes('number')}
    return df.assign(**outlier_cols)

    for col in df.select_dtypes('number'):
        df[col + '_low_outliers'] = get_lower_outliers(df[col], k)
    
    return df

####### example to call outliers functions ####
# malldf = prepare.add_upper_outlier_columns(df, k=1.5)  

# outlier_cols = [col for col in malldf if col.endswith('_outliers')]
# for col in outlier_cols:
#     print('~~~\n' + col)
#     data = malldf[col][malldf[col] > 0]
#     print(data.describe())





#################### Prepare Zillow Regression Data ##################

def wrangle_zillow(path):
    '''This function makes all necessary changes to the dataframe for exploration and modeling'''
    df = pd.read_csv(path)
    # Rename columns for clarity
    df.rename(columns={"hashottuborspa":"hottub_spa","fireplacecnt":"fireplace","garagecarcnt":"garage"}, inplace = True)
    df.rename(columns = {'Unnamed: 0':'delete', 'id.1':'delete1'}, inplace = True)

    # Replaces NaN values with 0
    df['garage'] = df['garage'].replace(np.nan, 0)
    df['hottub_spa'] = df['hottub_spa'].replace(np.nan, 0)
    df['lotsizesquarefeet'] = df['lotsizesquarefeet'].replace(np.nan, 0)
    df['poolcnt'] = df['poolcnt'].replace(np.nan, 0)
    df['fireplace'] = df['fireplace'].replace(np.nan, 0)
        
    ## Convert to Category
    df["zip"] = df["regionidzip"].astype('category')
    df["useid"]= df["propertylandusetypeid"].astype('category')
    df["year"]= df["yearbuilt"].astype('category')

    # Add Category Codes
    df["zip_cc"] = df["zip"].cat.codes
    df["useid_cc"] = df["useid"].cat.codes
    df["year_cc"] = df["year"].cat.codes

    # Columns to drop
    df.drop(columns= ['parcelid','id','airconditioningtypeid','architecturalstyletypeid','basementsqft','buildingclasstypeid','buildingqualitytypeid'], inplace = True)
    df.drop(columns= ['calculatedbathnbr','decktypeid','finishedfloor1squarefeet','finishedsquarefeet12','finishedsquarefeet13','finishedsquarefeet15'], inplace = True)
    df.drop(columns= ['finishedsquarefeet50','finishedsquarefeet6', 'fullbathcnt','heatingorsystemtypeid','poolsizesum','pooltypeid10','pooltypeid2'], inplace = True)
    df.drop(columns= ['pooltypeid7','propertycountylandusecode','propertyzoningdesc','rawcensustractandblock','regionidcity','regionidcounty','regionidneighborhood'], inplace = True)
    df.drop(columns= ['storytypeid','threequarterbathnbr','typeconstructiontypeid','unitcnt','yardbuildingsqft17','yardbuildingsqft26', 'numberofstories'], inplace = True)
    df.drop(columns= ['fireplaceflag','structuretaxvaluedollarcnt','assessmentyear','landtaxvaluedollarcnt', 'taxdelinquencyflag','taxdelinquencyyear'], inplace = True)
    df.drop(columns= ['censustractandblock','logerror','transactiondate','garagetotalsqft', "yearbuilt", "regionidzip", "propertylandusetypeid",'delete','delete1'], inplace = True)

    # Rows to drop
    rows_to_remove = [1600, 1628, 5099, 5969, 8109, 8407, 8521, 8849, 11562, 12430, 14313, 20313, 21502]
    df = df[~df.index.isin(rows_to_remove)]
    
    # Problem 'bedbathratio' - New Feature (Ratio of bedroomcnt and bathroomcnt)
    #df['bedbathratio'] = df.bedroomcnt.div(df.bathroomcnt, axis=0)

    # drop any nulls
    df = df.dropna()

    # split dataset
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123)
    train.shape, validate.shape, test.shape

    # Assign variables
    # x df's are all numeric cols 
    X_train = train.drop(columns=['taxvaluedollarcnt','zip','useid',"year", 'taxamount', 'fips', 'latitude', 'longitude'])
    X_validate = validate.drop(columns=['taxvaluedollarcnt','zip','useid',"year", 'taxamount', 'fips', 'latitude', 'longitude'])
    X_test = test.drop(columns=['taxvaluedollarcnt','zip','useid',"year", 'taxamount', 'fips', 'latitude', 'longitude'])
    X_train_explore = train

    # I need X_train_explore set to train so I have access to the target variable.

    # y df's are just fertility
    y_train = train[['taxvaluedollarcnt']]
    y_validate = validate[['taxvaluedollarcnt']]
    y_test = test[['taxvaluedollarcnt']]

    scaler = MinMaxScaler(copy=True).fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, 
                                columns=X_train.columns.values).\
                                set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled, 
                                    columns=X_validate.columns.values).\
                                set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled, 
                                    columns=X_test.columns.values).\
                                set_index([X_test.index.values])

    return df, X_train_explore, X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test



