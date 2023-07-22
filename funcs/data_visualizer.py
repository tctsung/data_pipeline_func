import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
import missingno as mn       # missing data visualization
from warnings import warn    
from matplotlib.dates import date2num, num2date  # transform dtype for plotting datetime
from pandas.api.types import is_float_dtype, is_categorical_dtype, is_integer_dtype
import data_cleaner as dc
class Visualizer:
    """
    TODO: Data visualization for cleaned data
    note: col dtypes should be integer, float, or datatime
    """
    def __init__(self, df, key_feature=None, corr=True, univariate=True, bivariate=True, na_pattern=True):
        assert isinstance(df, pd.DataFrame), "Input data must be a pandas DataFrame."
        self.df = df
        self.key_feature = key_feature
        self.corr = corr
        self.univariate = univariate
        self.bivariate = bivariate
        self.na_pattern = na_pattern
        print(self.__repr__())
    def __repr__(self):
        options = [f"1. Univariate distributions using `data_visualizer.univariate_dashboard(df)`: {self.univariate}", f"2. Bivariate distributions using `data_visualizer.bivariate_dashboard(df, key_feature)`: {self.bivariate}", 
        f"3. Correlation plots for continuous variables using `data_visualizer.correlation(df.corr())`: {self.corr}", f"4. Missing data pattern plots using `data_visualizer.na_plots(df)`: {self.na_pattern}"]
        options_str = "\n".join(options)
        return f"Visualizer class with the following plotting options:\n{options_str}"
    def auto_plots(self):
        if self.univariate:
            print('---------------------------')
            print('Univariate distributions: ')
            univariate_dashboard(self.df)
        if self.bivariate:
            print('---------------------------')
            print(f'Bivariate distributions using key_feature {self.key_feature}')
            assert self.key_feature in self.df.columns, f'Must defince key_feature for bivariate plots. feature {self.key_feature} is not in column names'
            bivariate_dashboard(self.df, self.key_feature)
        if self.corr:
            print('---------------------------')
            print('Correlation plots using all cols with dtype=\'float\'')
            corr = self.df.select_dtypes('float').corr()
            correlation(corr)
        if self.na_pattern:
            print('---------------------------')
            print('heatmap, barplot, correlation plots of missing data pattern')
            na_plots(self.df)
def univariate_dashboard(df, max_unique=10, fontsize=None, rotation=-30):
    """
    TODO: draw univariate plots for all features; bar plots for categorical, hist+kde for continuous
    param df: input polars dataframe
    param max_unique: max unique groups to be considered as categorical; pass to dc.auto_dtype()
    param fontsize: font size in x,y-axis; if None default font size will be printed out
    param rotation: the degree to rotate x-axis labeling
    """
    dtypes = dc.auto_dtype(df, max_unique)
    num_plots = df.shape[1]
    num_cols = min(int(num_plots ** 0.5),4)    # make row no. close to col no.
    num_rows = (num_plots - 1) // num_cols + 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*4))
    if fontsize is None:
        fontsize = num_cols*5
        print(f'Auto font size: {fontsize}')
    axes = axes.flatten()
    df = df.to_pandas()
    for (col, ax) in zip(df.columns, axes):        # zip will loop until 1 out of idx
        data = df[col].dropna()
        dtype = dtypes[col]                        # categorical/continuous/temporal
        if dtype=='continuous':                    # histogram + density for continuous
            sns.histplot(x=data, kde=True, ax=ax)
        elif dtype=='categorical':                 # bar plot for categorical data
            sns.countplot(x=data, ax=ax)           # avoid long x-axis label
        else:                                      # dtype=='temporal'
            sns.histplot(x=data, kde=True, ax=ax)  # may change in the future
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation) # rotate x-axis
        ax.set_xlabel(col, fontsize=fontsize)
        ax.set_ylabel("", fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize-2)
        ax.tick_params(axis='y', labelsize=fontsize-2)
    plt.tight_layout()                             # Adjust spacing between subplots
    plt.show()
    return fig
def bivariate_dashboard(df, key_feature, max_unique=10, stacked=False, fontsize=None):
    """
    TODO: 
    param df: input pandas dataframe
    param key_feature: the feature to include in every bivariate plot (usually the outcome variable); must be int, float, or categorical
    param stacked: if True, show the proportion instead of count in categorical vs. categorical barplot
    param fontsize: font size in x,y-axis; if None default font size will be printed out
    """
    key_dtype = df[key_feature].dtype
    assert ((key_dtype==pl.Categorical) or (key_dtype in pl.NUMERIC_DTYPES)), f"dtype of key_feature must be float, int, or categorical"
    dtypes = dc.auto_dtype(df, max_unique)       # to decide type of biplot
    # check key_feature dtype:
    key_dtype = dtypes[key_feature]
    # plot setup
    num_plots = len(df.columns)-1                # -1 because key_feature
    num_cols = min(int(num_plots ** 0.5), 4)     # make row no. close to col no.
    num_rows = (num_plots - 1) // num_cols + 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*4))
    if fontsize is None:
        fontsize = num_cols*5
        print(f'Auto font size: {fontsize}')
    axes = axes.flatten()
    df = df.to_pandas()
    features = df.columns.to_list()
    features.remove(key_feature)                 # skip when col==key_feature
    # Visualization:
    for (col, ax) in zip(features, axes):  
        data=df[[key_feature,col]].dropna()      # rm missing data
        dtype = dtypes[col]                      # check 2nd variable is categorical/continuous/temporal
        if key_dtype=='continuous':          
            if dtype=='continuous':              # scatter plot for continuous X & Y
                # sns.scatterplot(x=col, y=key_feature, data=data, ax=ax)  # failed
                # sns.regplot(x=col, y=key_feature, data=data, ax=ax)      # failed
                data.plot.scatter(key_feature, col, ax=ax)
            elif dtype=='categorical':           # violin plot for categorical X vs. continuous Y
                sns.violinplot(x=col, y=key_feature, data=data, ax=ax)
            else:                                # temporal x & 
                pass
            ax.set_xlabel(col, fontsize=fontsize)
            ax.set_ylabel(key_feature, fontsize=fontsize)
        elif key_dtype=='categorical':
            if dtype=='continuous':              # violin plot for continuous X vs. categorical Y
                sns.violinplot(x=key_feature, y=col, data=data, ax=ax)
            elif dtype=='categorical':           # count plot for Y in each categorical X
                pd.crosstab(df[key_feature], df[col], normalize='index').plot(kind='bar', stacked=stacked, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30) 
                if len(df[col].unique())>10:  # rm legend if too many gps
                    ax.legend().remove()
            else:    # temporal x
                pass
            ax.set_xlabel(key_feature, fontsize=fontsize)
            ax.set_ylabel(col, fontsize=fontsize)
        ax.set_xticks(ax.get_xticks())
        ax.tick_params(axis='x', labelsize=fontsize-2)
        ax.tick_params(axis='y', labelsize=fontsize-2)
    plt.tight_layout()  # Adjusts the spacing between subplots
    plt.show()
    return fig
def na_plots(df):
    """
    TODO: visualize the missingness in the data
    param df: input pandas dataframe
    """
    df_na = df.loc[:,df.isna().mean()!=0]  # don't plot the cols with no NaN
    mn.matrix(df_na, sort='ascending')     # sparkline represent # of missingness in each row
    plt.show()
    mn.bar(df_na, sort='ascending')        # no. of data in each feature
    plt.show()
    mn.heatmap(df_na)                      # nullity correlation
    plt.show()

def correlation(corr, hc_order=True, save_path=None, **kwargs):
    """
    TODO: draw correlation plot order by hierarchical clustering 
    param corr: the correlation matrix obtained from df.corr()
    param hc_order: if True, do the clustering and reorder corr matrix
    param save_path: str, the path & file name to save the plot
    return correlatin plot obtained from sns.heatmap()
    """ 
    check_corr(corr)   # check whether there are values with corr=1 or -1
    if hc_order:   # Calculate hierarchical clustering
        dendro_row = leaves_list(linkage(corr.values, method='ward'))
        dendro_col = leaves_list(linkage(corr.values.T, method='ward'))
        corr = corr.iloc[dendro_row, dendro_col]  # Reorder based on cluster order
    mask = np.triu(np.ones_like(corr, dtype=bool))  # mask half of the area
    corr_plt = sns.heatmap(corr, mask=mask, cmap='Blues', **kwargs)   # visualization
    if save_path:
    	corr_plt.figure.savefig(save_path,bbox_inches="tight") # bbox_inches: avoid cutoff
    plt.show()
    return corr_plt.figure
def check_corr(df):
    corr = df.select_dtypes('number').corr().abs()
    np.fill_diagonal(corr.values, np.nan)
    row_idx, col_idx = np.where(np.isclose(corr.values, 1))
    lens = len(row_idx)//2
    if lens>0:
        warn('There are features with absolute correlation==1:')
        for x,y in zip(row_idx[:lens], col_idx[:lens]):
            warn(f'{corr.index[x]} & {corr.index[y]}')
