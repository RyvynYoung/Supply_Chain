# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Wrangling
import pandas as pd
import numpy as np

# Exploring
import scipy.stats as stats

# Visualizing
import matplotlib.pyplot as plt
import seaborn as sns

def create_hist(germany):
    # create histograms
    for col in germany.columns:
        germany[col].hist()
        plt.title(col)
        plt.show()

def get_missing_rows(df):
    '''
    Write a function that takes in a dataframe of observations and attributes and returns a dataframe where each row is 
    an atttribute name, the first column is the number of rows with missing values for that attribute, and the second column 
    is percent of total rows that have missing values for that attribute.
    '''
    # find the number of rows in each column that are missing values
    num_rows_missing = df.isna().sum()
    # create new df with just that column
    dfrows = pd.DataFrame(num_rows_missing, columns=['num_rows_missing'])
    # add a calculation of % missing to the new df
    dfrows['pct_rows_missing'] = dfrows.num_rows_missing/df.shape[0]
    # return the new df
    return dfrows

def get_missing_cols(df):
    '''
    Write a function that takes in a dataframe and returns a dataframe with 3 columns: the number of columns missing, 
    percent of columns missing, and number of rows with n columns missing
    '''
    # add calculation columns to original df
    df['null_count'] = df.isna().sum(axis=1)
    df['pct_null'] = df.null_count/df.shape[1]
    
    # create a dataframe with just the 2 new columns
    dfcol = pd.DataFrame(df.null_count, columns=['null_count'])
    dfcol['pct_null'] = df.pct_null
    
    # create a series that has the number of rows in each group
    num_rows_ingroup = dfcol.null_count.value_counts()
    
    # create a dataframe with the count of null_count and pct_null
    groups = dfcol.groupby(['null_count', 'pct_null']).count()
    
    # create a df from the num_rows_ingroup, rename the columns, sort, and reset the index 
    dfnum_rows = pd.DataFrame(num_rows_ingroup)
    dfnum_rows = dfnum_rows.reset_index()
    dfnum_rows = dfnum_rows.rename(columns={'index': 'num_null_col', 'null_count': 'num_rows_with_count'})
    dfnum_rows = dfnum_rows.sort_values('num_null_col')
    dfnum_rows = dfnum_rows.reset_index()
    
    # reset the index on the groups df so that we can add the num_rows_with_count
    groups = groups.reset_index()
    
    # combine num_rows_with_count from dfnum_rows with groups
    groups['rows_with_count'] = dfnum_rows.num_rows_with_count
    return groups


def df_summary(df):
    '''
    This prints summary info for the dataframe. Useful for handling nulls
    '''
    print(df.shape)
    print(df.info())
    print(df.describe())
    nulls_by_row = get_missing_rows(df)
    nulls_by_col = get_missing_cols(df)
    print(nulls_by_row)
    print(nulls_by_col)
    graphs = df.hist(figsize=(24, 10), bins=20)
    print(plt.tight_layout(), graphs, plt.show())
    return df

