import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
# Visualizer: for exploratory analysis
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
    return corr_plt.figure
def univariate_dashboard(df, fontsize=None, rotation=0):
    num_plots = len(df.columns)
    num_cols = int(num_plots ** 0.5)    # make row no. close to col no.
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


