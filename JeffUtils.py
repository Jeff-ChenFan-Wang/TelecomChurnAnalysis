import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes

def get_second_deriv(series:pd.Series)->pd.Series:
    """calculates the second derivative 
    at every point in the series
    """
    df = series.rename('x').to_frame()
    df['next'] = df['x'].shift()
    df['prev'] = df['x'].shift(-1)
    return df.apply(
        lambda row:
        row['next']+row['prev'] - 2*row['x'],
        axis=1
    )

def graph_elbow(series:pd.Series)->int:
    """graphs the elbow of a pd.Series of Kmeans inertias
    to show optimum number of clusters,
    then returns the optimum number of clusters
    """
    ax = sns.lineplot(x=series.index,y=series)
    elbow = get_second_deriv(series).argmax()
    ax.axvline(elbow,0,ax.get_ylim()[1],color='red',**{'alpha':0.5})
    ax.set_xlabel('Parameter')
    return elbow

def graph_cluster_wc(df: pd.DataFrame, tokenized_col: str,
                     cluster_col: str, clusterId: int, head:int=5,
                    ax:matplotlib.axes=None):
    """graphs the word count of each unique word in a cluster of documents
    each document must be tokenized
    
    Keyword arguments:
    df -- a dataframe with a column contianing tokenized documents, 
        and a column of ints containing the cluster the document belongs to
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