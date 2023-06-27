import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
import missingno as mn       # missing data visualization
from warnings import warn    
from matplotlib.dates import date2num, num2date  # transform dtype for plotting datetime
from pandas.api.types import is_float_dtype, is_categorical_dtype, is_integer_dtype
# Visualizer: for exploratory analysis
class Visualizer:
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
        options = [f"1. Univariate distributions using `vis.univariate_dashboard(df)`: {self.univariate}", f"2. Bivariate distributions using `vis.bivariate_dashboard(df, key_feature)`: {self.bivariate}", 
        f"3. Correlation plots for continuous variables using `vis.correlation(df.corr())`: {self.corr}", f"4. Missing data pattern plots using `vis.na_plots(df)`: {self.na_pattern}"]
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
def univariate_dashboard(df, fontsize=None, rotation=-30):
    """
    TODO: draw univariate plots for all features; bar plots for categorical, hist+kde for continuous
    param df: input pandas dataframe
    param fontsize: font size in x,y-axis; if None default font size will be printed out
    param rotation: the degree to rotate x-axis labeling
    """
    num_plots = len(df.columns)-1
    num_cols = min(int(num_plots ** 0.5),4)    # make row no. close to col no.
    num_rows = (num_plots - 1) // num_cols + 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*4))
    if fontsize is None:
        fontsize = num_cols*5
        print(f'Auto font size: {fontsize}')
    int_cols = df.select_dtypes(include='integer').columns
    date_cols = df.select_dtypes(include='datetime64').columns
    axes = axes.flatten()
    for i, (column, ax) in enumerate(zip(df.columns, axes)):
        data = df[column].dropna()
        if data.dtype=='float':
            sns.histplot(x=data, kde=True, ax=ax)
        elif column in int_cols:   # bar plot for categorical data
            sns.countplot(x=data, ax=ax) # avoid long x-axis label
        elif column in date_cols:
            sns.histplot(x=data, kde=True, ax=ax)  # may change
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation) # rotate x-axis
        ax.set_xlabel(column, fontsize=fontsize)
        ax.set_ylabel("", fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize-2)
        ax.tick_params(axis='y', labelsize=fontsize-2)
    plt.tight_layout()  # Adjusts the spacing between subplots
    plt.show()
    return fig
def bivariate_dashboard(df, key_feature, key_dtype=None, stacked=False, fontsize=None):
    """
    TODO: 
    param df: input pandas dataframe
    param key_feature: the feature to include in every bivariate plot (usually the outcome variable); must be int, float, or categorical
    param key_dtype: dtype of key_feature, should be ('continuous', 'categorical'); if None, auto detection will be operate
    param stacked: if True, stack the categorical vs. categorical barplot
    param fontsize: font size in x,y-axis; if None default font size will be printed out
    """
    key_dtypes = (None, 'continuous', 'categorical')
    assert key_dtype in key_dtypes, f"Invalid value for 'key_dtype' parameter. Expected one of: {key_dtypes}"
    ori_dtype = df[key_feature].dtype
    assert is_categorical_dtype(ori_dtype)+is_float_dtype(ori_dtype)+is_integer_dtype(ori_dtype)==1, f"dtype of key_feature must be float, int, or categorical"
    categories = len(df[key_feature].unique())
    # check key_feature dtype:
    if key_dtype is None:
        if is_float_dtype(ori_dtype):  # treat float as continuous Y
            key_dtype='continuous'
            print(f'key_feature {key_feature} is treated as continuous variable due to dtype=\'float\'')
            if categories < 10:
                warn(f'There are only {categories} unique values in key_feature, change arg key_dtype=\'categorical\' if it\'s not a continuous variable')
        elif (categories > 30) and is_integer_dtype(ori_dtype):
            key_dtype='continuous'
            print(f'key_feature {key_feature} is treated as continuous variable since unique categories > 30\nChange arg key_dtype=\'categorical\' if it\'s not a continuous variable')
        else:
            key_dtype='categorical'
            print(f'key_feature {key_feature} is treated as categorical variable due to dtype={ori_dtype}')
            df[key_feature] = df[key_feature].astype('category') # temp transfer for plotting
            if categories > 10: 
                warn(f'There are {categories} unique values in key_feature, change arg key_dtype=\'continuous\' if it\'s not a categorical variable')
            elif categories > 30:
                warn(f'Stop because there are >30 categories in key_feature with dtype={ori_dtype}, change key_dtype to \'categorical\' if you want to force it to make biplots with this amount of categories ')
                return
    # setup
    num_plots = len(df.columns)-1       # -1 because key_feature
    num_cols = min(int(num_plots ** 0.5), 4)    # make row no. close to col no.
    num_rows = (num_plots - 1) // num_cols + 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*4))
    # visualization:
    if fontsize is None:
        fontsize = num_cols*5
        print(f'Auto font size: {fontsize}')
    # group variables to continuous & categorical:
    categorical_cols = df.select_dtypes(include=['integer', 'category', 'object']).columns
    continuous_cols = df.select_dtypes(include=['float', 'datetime64']).columns
    date_cols = df.select_dtypes(include=['datetime64']).columns
    ori_date_cols = df[date_cols].copy()
    df[date_cols] = df[date_cols].apply(date2num)
    axes = axes.flatten()
    if key_dtype=='continuous':
        for i, (column, ax) in enumerate(zip(set(df.columns) - set(key_feature), axes)):
            data=df[[key_feature,column]].dropna()
            if column==key_feature:          # histogram for key_feature
                sns.histplot(x=df[key_feature].dropna(), kde=True, ax=ax)
            elif column in continuous_cols:  # scatter plot for continuous X & Y
                sns.regplot(x=column, y=key_feature, scatter=True,line_kws={"color": "orange"}, data=data, ax=ax)
            elif column in categorical_cols: # violin plot for categorical X
                sns.violinplot(x=column, y=key_feature, data=data, ax=ax)
            ax.set_xticks(ax.get_xticks())
            ax.set_xlabel(column, fontsize=fontsize)
            ax.set_ylabel(key_feature, fontsize=fontsize)
            ax.tick_params(axis='x', labelsize=fontsize-2)
            ax.tick_params(axis='y', labelsize=fontsize-2)
    elif key_dtype=='categorical':
        for i, (column, ax) in enumerate(zip(df.columns, axes)):
            data=df[[key_feature,column]].dropna()
            if column==key_feature:
                sns.countplot(x=df[key_feature].dropna(), ax=ax)
            elif column in continuous_cols:  # violin plot for continuous X & categorical Y
                sns.violinplot(x=key_feature, y=column, data=data, ax=ax)
            elif column in categorical_cols: # count plot for Y in each categorical X
                pd.crosstab(df[key_feature], df[column]).plot(kind='bar', stacked=stacked, ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30) 
                if len(df[column].unique())>10:  # rm legend if too many gps
                    ax.legend().remove()
            ax.set_xticks(ax.get_xticks())
            ax.set_xlabel(key_feature, fontsize=fontsize)
            ax.set_ylabel(column, fontsize=fontsize)
            ax.tick_params(axis='x', labelsize=fontsize-2)
            ax.tick_params(axis='y', labelsize=fontsize-2)
    df[key_feature] = df[key_feature].astype(ori_dtype) # transform back to original dtype
    df[date_cols] = ori_date_cols  # transform back to date format
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

