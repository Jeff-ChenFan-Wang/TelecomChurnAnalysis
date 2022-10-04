import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

def get_subplot_dim(num:int)->Tuple[int,int]:
    """returns row and column dimensions closest to a square

    Args:
        num (int): total number of graphs needed

    Returns:
        Tuple[int,int]: (rows,columns) to make a group of subplots square
    """
    r = int(np.floor(np.sqrt(num)))
    c = int(np.ceil(num/r))
    return r,c

def get_second_deriv(continuous_data:pd.Series)->pd.Series:
    """Calculates the second derivative at every point in the series

    Args:
        continuous_data (pd.Series): Sorted pd.Series of numerical data

    Returns:
        pd.Series: A series of calculated second
            derivatives between each point of the input
        
    """
    df = continuous_data.rename('x').to_frame()
    df['next'] = df['x'].shift()
    df['prev'] = df['x'].shift(-1)
    return df.apply(
        lambda row:
        row['next']+row['prev'] - 2*row['x'],
        axis=1
    )

def graph_elbow(loss:pd.Series)->int:
    """graphs the elbow of a pd.Series of Kmeans inertias
    to show optimum number of clusters,
    then returns the optimum number of clusters

    Args:
        series (pd.Series): A pd.Series of Kmeans inertia at every cluster no.

    Returns:
        int: optimal number of clusters
    """
    ax = sns.lineplot(x=loss.index,y=loss)
    elbow = get_second_deriv(loss).argmax()+loss.index.min()
    ax.axvline(elbow,0,ax.get_ylim()[1],color='red',**{'alpha':0.5})
    ax.set_xlabel('Parameter')
    ax.set_title(f'Optimal # of Clusters: {elbow}')
    return elbow