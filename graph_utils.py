from lib2to3.pytree import Base
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import pickle
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    RocCurveDisplay, roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import GridSearchCV

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

def graph_cv_results(grid_cv:GridSearchCV,x:str,hue:str)->None:
    """graphs the cross validation results

    Args:
        grid_cv (GridSearchCV): trained sklearn gridsearch object
        x (str): hyperparameter name you want for x axis
        hue (str): another hyperparameter name you want to compare 
    """
    cvDat = pd.DataFrame(grid_cv.cv_results_)
    cv_tst_score_cols = cvDat.columns[
        cvDat.columns.str.contains('split[0-9]_test_score',regex=True)
    ]
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax = sns.lineplot(
        data=cvDat[[x,hue]].join(
            cvDat.apply(
                lambda row: row[cv_tst_score_cols].to_numpy(),
                axis=1
            ).rename('test_score')
        ).explode('test_score'),
        x=x,y='test_score',hue=hue,ax=ax
    )
    ax.set_title(
        grid_cv.best_params_
    )

    return

def graph_estimator_auc(
        estimators:List[BaseEstimator], data_ls:List[np.ndarray],
        y_test:np.ndarray[int],figsize:Tuple[int,int]=(12,8))->None:
    """Graphs the ROC curve of multiple estimators at once in a single plot
    and shows the AUC value

    Args:
        estimators (List[BaseEstimator]): list of trained sklearn estimators
        data_ls (List[np.ndarray]): list of test data for each estimator
        y_test (np.ndarray[int]): true labels for test set
        figsize (Tuple[int,int], optional): Figure size. Defaults to (12,8).
    """
    r,c = get_subplot_dim(len(estimators))
    fig, ax = plt.subplots(r,c,figsize=figsize)
    for estim, dat, subplot in zip(estimators,data_ls,ax.flatten()):
        RocCurveDisplay.from_estimator(
            estim,dat,y_test,ax=subplot)
        score = str(roc_auc_score(y_test,estim.predict_proba(dat)[:,1]))[:7]
        subplot.set_title(
            f'{estim.__class__.__name__} AUC: {score}'
        )
    plt.tight_layout()
    
def graph_estimator_cmat(
        estimators:List[BaseEstimator], data_ls:List[np.ndarray],
        y_test:np.ndarray[int],figsize:Tuple[int,int]=(12,8))->None:
    """Graphs the confusion matrix of multiple estimators at once in a 
    single plot

    Args:
        estimators (List[BaseEstimator]): list of trained sklearn estimators 
        data_ls (List[np.ndarray]): list of test data for each estimator
        y_test (np.ndarray[int]): true labels for test set
        figsize (Tuple[int,int], optional): Figure size. Defaults to (12,8).
    """
    r,c = get_subplot_dim(len(estimators))
    fig, ax = plt.subplots(r,c,figsize=figsize)
    for estim, dat, subplot in zip(estimators,data_ls,ax.flatten()):
        ConfusionMatrixDisplay.from_estimator(
            estim,dat,y_test,ax=subplot,cmap='bone')
        subplot.set_title(
            f'{estim.__class__.__name__}'
        )
    plt.tight_layout()
    
def graph_feat_importance(
        feat_imps:List[np.ndarray],feat_names:List[str])->None:
    """graphs the top 10 most important features for each estimator

    Args:
        feat_imps (List[np.ndarray]): feature importances of estimator
        feat_names (List[str]): feature names
    """
    plt.figure(figsize=(7,6))
    named_imps = pd.Series(
        feat_imps,index=feat_names
    ).sort_values(ascending=False).head(10)
    ax = sns.barplot(x=named_imps,y=named_imps.index)
    ax.set_title('Top 10 Most Important Features')
    
def graph_lift(
        estimator:BaseEstimator, y_test:np.ndarray[int],
        x_test:np.ndarray[np.number],figsize:Tuple[int,int]=(7,6))->None:
    """Graphs the lift of the estimator

    Args:
        estimator (BaseEstimator): trained sklearn estimator
        y_test (np.ndarray[int]): true labels for test set
        x_test (np.ndarray[np.number]): test set features
        figsize (Tuple[int,int], optional): Figure size. Defaults to (12,8).
    """
    ranked_probs = pd.DataFrame(
        {'label':y_test,'prob':estimator.predict_proba(x_test)[:,1]}
    ).sort_values('prob',ascending=False)
    ranked_probs['decile'] = np.digitize(
        ranked_probs['prob'],
        np.percentile(ranked_probs['prob'],range(100,-1,-10)),
        right=False
    )
    ranked_probs['decile'] = ranked_probs['decile'].replace(0,1)
    gain = ranked_probs.groupby('decile')['label'].sum().sort_index().cumsum()
    lift = gain/(gain.index*10)
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=lift.index,y=lift)
    ax.set_title(f'Lift Chart for {estimator.__class__.__name__}')
    ax.set_ylabel('Lift')
    for i in ax.containers:
        ax.bar_label(i,)

def pickle_model(model:object,file_path:str)->None:
    with open(file_path, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_pickle_model(file_path:str)->object:
    with open(file_path, 'rb') as handle:
        model = pickle.load(handle)
        return model