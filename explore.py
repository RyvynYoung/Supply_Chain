import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
from sklearn.cluster import KMeans
import statsmodels.api as sm
import summarize
import prepare


##### time series #####
def set_plotting_defaults():
    # plotting defaults
    plt.rc('figure', figsize=(13, 7))
    plt.style.use('seaborn-whitegrid')
    plt.rc('font', size=16)

def total_sales_df(df, y, period):
    ts_df = df[[y]]
    ts_df = ts_df.resample(period).sum()
    return ts_df

def average_df(df, y, period):
    ts_df = df[[y]]
    ts_df = ts_df.resample(period).mean()
    return ts_df

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

def train_exp(train):
    '''create copy of train and add category columns for exploration'''
    train_exp = train.copy()
    train_exp['month'] = train_exp.index.month
    train_exp['weekday'] = train_exp.index.day_name()
    return train_exp

def bar_plots(train):
    '''create bar plots for exploration'''
    # only create plots for non-category columns
    for col in train.columns:
        if col not in ['month', 'year', 'weekday']:
            train.groupby('month')[col].mean().plot.bar()
            plt.title(col)
            plt.show()

def sns_boxplot(train):
    '''create box plots for exploration'''
    # only create plots for non-category columns
    for col in train.columns:
        if col not in ['month', 'year', 'weekday']:
            order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sns.boxplot(data=train, y=col, x='weekday', order=order)
            plt.title(col)
            plt.show()

# note: need to define y = target before running any of the below
def resample_plot(y):
    y.resample('D').mean().plot(alpha=.5, label='Daily')
    y.resample('W').mean().plot(alpha=.8, label='Weekly')
    y.resample('M').mean().plot(label='Montly')
    plt.legend()

def lag_plots(y):
    shift_list = [-1, -7, -30, -90, -180, -365]
    for i in shift_list:
        plt.scatter(y, y.shift(i))
        plt.xlabel('$y$')
        plt.ylabel(f'$y shift{i}$')
        plt.title(f'Lag plot with lag={i}')
        plt.show()

def seasonal_plot(y):
    # .unstack turns an index level into columns
    y.groupby([y.index.year, y.index.month]).mean().unstack(0).plot(title='Seasonal Plot')

def season_subseries_plot(y):
    table = y.groupby([y.index.year, y.index.month]).mean().unstack()

    fig, axs = plt.subplots(1, 12, sharey=True, sharex=True)
    for ax, (month, subset) in zip(axs, table.iteritems()):
        subset.plot(ax=ax, title=month)
        ax.hlines(subset.mean(), *ax.get_xlim(), ls='--')
        ax.set(xlabel='')

    axs[0].set(ylabel='Average monthly temp (F)')
    fig.suptitle('Seasonal Subseries Plot') # super-title for the overall figure
    fig.subplots_adjust(wspace=.05)

def autocorrelation_plot(y):
    plot = pd.plotting.autocorrelation_plot(y.resample('W').mean())
    return plot

# next function redefines y
def seasonal_decomposition(train):
    y = train.Consumption.resample('W').mean()
    result = sm.tsa.seasonal_decompose(y)
    decomposition = pd.DataFrame({
        'y': result.observed,
        'trend': result.trend,
        'seasonal': result.seasonal,
        'resid': result.resid,
    })
    return decomposition, result

def decomposition_plot(decomposition):
    decomposition.iloc[:, 1:].plot()


##### train, test only #####
def split_data_percent(df):
    ''' splits into train and test only'''
    train_size = .70
    n = df.shape[0]
    test_start_index = round(train_size * n)

    train = df[:test_start_index] # everything up (not including) to the test_start_index
    test = df[test_start_index:] # everything from the test_start_index to the end
    # changed plotting here to pandas instead of matplotlib
    ax = train.plot()
    test.plot(ax=ax)
    return train, test

def split_human(df, yearcut1, year2):
    ''' train and test only'''
    # alternative Human based split method, use for germany data
    train = df[:yearcut1] # inclusive
    test = df[year2]

    ax = train.plot()
    test.plot(ax=ax)
    return train, test




def summary(df):
    '''
    print summary info then remove generated columns
    '''
    df = summarize.df_summary(df)
    cols_to_remove3 = ['null_count', 'pct_null', ]
    df = prepare.remove_columns(df, cols_to_remove3)
    return df

def plot_variable_pairs(df):
    '''
    visualizes pairs of variables
    '''
    g = sns.PairGrid(df) 
    g.map_diag(sns.distplot)
    g.map_offdiag(sns.regplot)


def plot_categorical_and_continuous_vars(categorical_var, continuous_var, df):
    '''
    visualize categorical and continuous variables
    '''
    plt.rc('font', size=13)
    plt.rc('figure', figsize=(13, 7))
    sns.boxplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()   
    sns.violinplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()
    sns.swarmplot(data=df, y=continuous_var, x=categorical_var)
    plt.show()


def ttest_1samp(sample, overall_mean):
    '''
    runs one sample t-test
    '''
    # set alpha value
    alpha = .05
    t, p = stats.ttest_1samp(sample, overall_mean)
    print('t=', t)
    print('p=', p)
    print("p is less than alpha=", p < alpha)
    return t, p


def pearson(continuous_var1, continuous_var2):
    '''
    runs pearson r test on 2 continuous variables
    '''
    alpha = .05
    r, p = stats.pearsonr(continuous_var1, continuous_var2)
    # print('r=', r)
    # print('p=', p)
    # if p < alpha:
    #     print("We reject the null hypothesis")
    # else:
    #     print("We fail to reject the null hypothesis")
    return r, p

def chi2test(categorical_var1, categorical_var2):
    '''
    runs chi squared test on 2 categorical variables
    '''
    alpha = 0.05
    contingency_table = pd.crosstab(categorical_var1, categorical_var2)

    chi2, p, degf, expected = stats.chi2_contingency(contingency_table)

    if p < alpha:
        print("We reject the null hypothesis")
        print(f'p     = {p:.4f}')
    else:
        print("We fail to reject the null hypothesis")
    return p

def elbow_plot(X_train_scaled, cluster_vars):
    '''
    elbow method to identify good k for us, originally used range (2,20), changed for presentation
    '''
    ks = range(2,16)
    
    # empty list to hold inertia (sum of squares)
    sse = []

    # loop through each k, fit kmeans, get inertia
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_train_scaled[cluster_vars])
        # inertia
        sse.append(kmeans.inertia_)
    # print out was used for determining cutoff, commented out for presentation
    # print(pd.DataFrame(dict(k=ks, sse=sse)))

    # plot k with inertia
    plt.plot(ks, sse, 'bx-')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('Elbow method to find optimal k')
    plt.show()

####### elbow_plot(X_train_scaled, cluster_vars = area_vars)


def run_kmeans(X_train_scaled, X_train, cluster_vars, k, cluster_col_name):
    '''
    create kmeans object
    '''
    kmeans = KMeans(n_clusters = k, random_state = 13)
    kmeans.fit(X_train_scaled[cluster_vars])
    # predict and create a dataframe with cluster per observation
    train_clusters = \
        pd.DataFrame(kmeans.predict(X_train_scaled[cluster_vars]),
                              columns=[cluster_col_name],
                              index=X_train.index)
    
    return train_clusters, kmeans

####### train_clusters, kmeans = run_kmeans(X_train_scaled, X_train, k, cluster_vars, cluster_col_name)

def kmeans_transform(X_scaled, kmeans, cluster_vars, cluster_col_name):
    '''
    creates clusters to add to validate and test tests
    '''
    kmeans.transform(X_scaled[cluster_vars])
    trans_clusters = \
        pd.DataFrame(kmeans.predict(X_scaled[cluster_vars]),
                              columns=[cluster_col_name],
                              index=X_scaled.index)
    
    return trans_clusters

####### trans_clusters = kmeans_transform(X_scaled, kmeans, cluster_vars, cluster_col_name)


def get_centroids(kmeans, cluster_vars, cluster_col_name):
    '''
    get centroids to add to X dataframes
    '''
    centroid_col_names = ['centroid_' + i for i in cluster_vars]

    centroids = pd.DataFrame(kmeans.cluster_centers_, 
             columns=centroid_col_names).reset_index().rename(columns={'index': cluster_col_name})
    
    return centroids

######### centroids = get_centroids(kmeans, cluster_vars, cluster_col_name)


def add_to_train(X_train, train_clusters, X_train_scaled, centroids, cluster_col_name):
    '''
    concatenate cluster id with dataframes
    '''
    X_train = pd.concat([X_train, train_clusters], axis=1)

    # join on clusterid to get centroids
    X_train = X_train.merge(centroids, how='left', 
                            on=cluster_col_name).\
                        set_index(X_train.index)
    
    # concatenate cluster id
    X_train_scaled = pd.concat([X_train_scaled, train_clusters], 
                               axis=1)

    # join on clusterid to get centroids
    X_train_scaled = X_train_scaled.merge(centroids, how='left', 
                                          on=cluster_col_name).\
                            set_index(X_train.index)
    
    return X_train, X_train_scaled

####### X_train, X_train_scaled = add_to_train(X_train, train_clusters, X_train_scaled, centroids, cluster_col_name)

def get_cluster_dummies(df_scaled, cluster_name):
    '''
    get dummies for the cluster name column and add those to the dataframe
    would like this to rename columns added as well, not able to make that work yet
    '''
    # create dummy vars of cluster name column
    cluster_df = pd.get_dummies(df_scaled[cluster_name])
    # concatenate the dataframe with the cluster columns to the original dataframe
    df_dummies = pd.concat([df_scaled, cluster_df], axis = 1)
    return df_dummies

def indep_target_vis(df_exp, columns_list):
    '''create loop to make plots of features with target'''
    for col in columns_list:
        plt.figure(figsize=(13, 7))
        sns.scatterplot(data=df_exp, x=col, y='logerror', hue='county')
        plt.title(f'Visualization of {col} with logerror by county')
    # # change x axis label to more descriptive name
    # # unable to get this working
    # xname = ['bedroom count', 'calculated finished sqft', 'full bath count', 'lot size sqft', 'room count', 
    #          'unit count', 'structure tax value dollars', 'tax value dollars', 'tax amount', 'LA county', 
    #          'Orange county', 'Ventura county', 'age of property', 'tax rate percentage', 
    #          'structure value dollars per sqft', 'land value dollars per sqft', 'bedroom/bathroom ratio']

    # # error I'm getting = "ValueError: too many values to unpack (expected 2)"

    # for col, n in columns, xname:
    #     plt.figure(figsize=(13, 7))
    #     sns.scatterplot(data=X_train_exp, x=col, y='logerror', hue='county')
    #     plt.title(f'Visualization of {col} with logerror by county')
    #     plt.xlabel(n)



    