#Docstrings are designed to integrate with Sphinx package

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes



def graph_cluster_wc(df:pd.DataFrame, tokenized_col:str,
                     cluster_col:str, clusterId:int, head:int=5,
                    ax:matplotlib.axes=None):
    """graphs the word count of each unique word in a cluster of documents
    each document must be tokenized.
    
    Keyword arguments:
    df -- a dataframe with a column contianing tokenized documents, and 
        a column of ints containing the cluster the document belongs to
    tokenized_col -- name of column containing tokenized document
    cluster_col -- name of column contianing cluster flags
    clusterId -- integer indicating which cluster you want to graph
    head -- only display top (head) words with highest word count
    ax -- matplotlib axes to graph to if needed
    """
    cluster = df[df[cluster_col]==clusterId]
    tokens_exploded = (
        cluster[tokenized_col]
        .explode()
        #turn to dataframe for dropping duplicates
        .to_frame().reset_index() 
        #drop duplicates to ensure column descriptions 
        #with repeating words don't mess with calculation
        #of percentage of descriptoins the word appears in
        .drop_duplicates()[tokenized_col]
        .value_counts()
        .head(head)
    )
    barAx = sns.barplot(
        x=tokens_exploded/cluster.shape[0],
        y=tokens_exploded.index,
        ax=ax,
        orient='h'
    )
    xTitle = "percentage of cluster with word occurence"
    if ax is not None:
        ax.set_xlabel(xTitle)
    else:
        barAx.set_xlabel(xTitle)

def graph_all_cluster_wc(col_desc:pd.DataFrame,
                         cluster_col:str,figsize=(15,8)):
    """graphs top 5 most common words in each cluster
    ensures that subplot arrangement are as square as possible
    instead of one long list
    """
    classes = np.sort(col_desc[cluster_col].unique())
    r = int(np.ceil(np.sqrt(classes.shape[0])))
    c = int(np.ceil(classes.shape[0]/r))
    fig, ax = plt.subplots(r,c,figsize=figsize)
    for i in range(r):
        for j in range(c):
            if i*c+j<classes.shape[0]:
                cluster = classes[i*c+j]
                graph_cluster_wc(
                    col_desc, 'tokenized',
                    cluster_col, cluster,
                    ax=ax[i][j]
                )
                ax[i][j].title.set_text(f'cluster {cluster}')
    plt.tight_layout()
    
def jeff_histplot(data:pd.Series,**kwargs):
    """the old sns.distplot is deprecated,
    but its replacement sns.histplot looks really ugly. 
    This simply calls the new histplot with prettier parameters
    that I like so I don't have to type it out everytime
    """
    return sns.histplot(data,alpha=0.2,edgecolor='grey',**kwargs)

def double_hist(data1:pd.Series, data2:pd.Series, bins, **kwargs):
    ax = sns.histplot(
        data1,
        stat='density',
        alpha=0.5,
        bins=bins,
        edgecolor=None,
        color='cyan'
    )
    sns.histplot(
        data2,
        stat='density',
        ax=ax,
        alpha=0.5,
        bins=bins,
        edgecolor=None,
        color='orange'
    )
    return ax
    
    