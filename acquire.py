import pandas as pd
import numpy as np
import os


#################### Acquire Pick Data ##################

def get_pick_data():
    '''
    This function reads in pickdf.csv and returns df
    '''
    df = pd.read_csv('pickdf.csv', index_col=0)
    return df



def run():
    print("Acquire: downloading raw data files...")
    df = get_pick_data()
    print("Acquire: Completed!")
    return df
