from ctypes import ArgumentError
import pandas as pd
import numpy as np
from sklearn.pipeline import NotFittedError, Pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder,
    FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from typing import List    
from sklearn.base import BaseEstimator

class CustomFeatureUnion(FeatureUnion):
    """Customized FeatureUnion class that allows access to transformers 
    via name like sklearn Pipelines and a working get_feature_names_out()
    """
    def get_transformer(self,transformer_name:str)->BaseEstimator:
        """access transformer in feature union by name

        Args:
            transformer_name (str): transformer name, accepted ones are
                'num', 'bin', or 'cat' for numerical data pipeline, binary
                data pipeline, and categorical data pipeline respectively

        Returns:
            BaseEstimator: Pipeline or transformer
        """
        return dict(self.transformer_list)[transformer_name]
    
    def _set_feature_names(
            self,num_cols:np.ndarray[str],
            bin_cols:np.ndarray[str],
            cat_cols:np.ndarray[str],
            is_pca:bool=False):
        """Retains feature names so that feature_names_out() can
        function

        Args:
            num_cols (np.ndarray[str]): numerical column names
            bin_cols (np.ndarray[str]): binary column names
            cat_cols (np.ndarray[str]): categorical column names
            is_pca (bool, optional): Whether the pipeline uses PCA. 
                Defaults to False.
        """
        self.num_cols = num_cols
        self.bin_cols = bin_cols
        self.cat_cols = cat_cols
        self.is_pca = is_pca
    
    def get_feature_names_out(self)->np.ndarray:
        """Returns feature names of each column in order

        Returns:
            np.ndarray: feature names in array like form
        """
        if self.__sklearn_is_fitted__():
            if self.is_pca:
                return np.hstack([
                    (
                        dict(self.transformer_list)['num']
                        .named_steps['select_pca_comps'].get_feature_names_out()
                    ),
                    self.bin_cols,
                    (
                        dict(self.transformer_list)['cat']
                        .named_steps['ohe'].get_feature_names_out(self.cat_cols)
                    )
                ])
            else:
                return np.hstack([
                    self.num_cols,self.bin_cols,
                    (
                        dict(self.transformer_list)['cat']
                        .named_steps['ohe'].get_feature_names_out(self.cat_cols)
                    )
                ])

class PipelineFactory:
    """A factory class for producing different sklearn pipelines. 
    As of version 1.1.2 for sci-kit learn, pipelines still cannot retain 
    feature names after transformation, hence some of the customized pipelines 
    produced by this class add a function transformer to retain feature names.
    """
    
    def __init__(self, raw_data:pd.DataFrame, radom_seed=1337):
        """initilizes the factory by determining which columns are categorical
        and which are numerical so output pipeline can make appropriate 
        transformations

        Args:
            raw_data (pd.DataFrame): raw data for pipelines to be fitted against
        """
        #predetermined seed so that results are reproducible
        self.random_seed=radom_seed 
        
        self.original_cols = raw_data.columns
        self.cat_cols = raw_data.select_dtypes(include='object').columns
        self.num_cols = raw_data.select_dtypes(include=np.number).columns
        self.bin_cols = raw_data.select_dtypes(include=bool).columns
    
    def col_names_to_idx(self,columns:List[str])->List[int]:
        """turns array of column names into array of column indices
        so that column transformer knows which columns to transform if
        feature names are lost inside the pipeline

        Args:
            columns (List[str]): list like of column names 

        Returns:
            List[int]: list like of indices of column names as they appeared
                in the original column order. Names that don't exist in 
                original columns are ignored. 
        """
        return np.in1d(self.original_cols,columns).nonzero()[0]
    
    def create_pipe(
                self, pca = False, *, 
                impute:bool, normalize:bool, 
                pca_comps:int=0)->Pipeline:
        """Factory method to create pipelines. Define whether the pipeline
        needs to impute missing data, and whether numerical columns should 
        be normalized (in the sense that it should be standardized). 
        
        The method will automatically determine which columns are categorical 
        and which are numerical based on the columns datatype for appropriate 
        transformations. Binary columns will not be normalized nor one-hot 
        encoded in any of the pipelines. 
        
        All pipelines created will one-hot encode categorical columns, with 
        unknown values encountered in the future mapped to an infrequent column

        Args:
            impute (bool): whether to use mode impution on missing data
            normalize (bool): whether to standardize data
            pca_comps (int): number of components to retain in PCA

        Raises:
            Exception: impute or normalize isn't defined
            NotImplementedError: combination of impute or normalize hasn't been
                implemented.

        Returns:
            Pipeline: sklearn pipeline with necessary transformers according to 
                input parameters, and also a functioning feature names method.
        """
        if pca:
            if pca_comps ==0:
                raise ArgumentError('Please set number of PCA comps to keep')
            else:
                return self.make_pca_pipe(pca_comps)
        else:
            if (impute is None) or (normalize is None):
                raise Exception('please instruct whether to impute or scale')
            if impute & normalize:
                return self.make_impute_ohe_scale_pipe()
            elif impute & ~normalize:
                return self.make_impute_ohe_pipe()
            elif ~impute & ~normalize:
                return self.make_ohe_pipe()
            else:
                raise NotImplementedError('Cannot determine pipeline type')
            
    def make_pca_pipe(self, n_comps:int)->CustomFeatureUnion:
        """creates a feature union of pipelines that imputes data using mode 
        impuation, normalizes and PCA's numerical columns, and one-hot encodes 
        categorical columns. 
        
        Args:
            n_comps (int): number of components to retain in PCA

        Returns:
            Pipeline: sklearn preprocessing pipeline 
        """
        num_pipe = Pipeline([
            ('select_num', 
                ColumnTransformer([
                    ('select_num','passthrough', 
                        self.col_names_to_idx(self.num_cols)),
                ])
            ),
            ('impute',SimpleImputer(strategy='most_frequent')),
            ('scale',StandardScaler()),
            ('pca', PCA(random_state=self.random_seed)),
            ('select_pca_comps', 
                ColumnTransformer([
                    ('select_pca_comps','passthrough', slice(n_comps)),
                ])
            ),
        ])
        
        bin_pipe = Pipeline([
            ('select_bin', 
                ColumnTransformer([
                    ('select_bin','passthrough', 
                        self.col_names_to_idx(self.bin_cols)),
                ])
            ),
            ('convert',FunctionTransformer(to_binary)),
            ('impute',SimpleImputer(strategy='most_frequent')),
        ])
        
        cat_pipe = Pipeline([
            ('select_cat', 
                ColumnTransformer([
                    ('select_cat','passthrough', 
                        self.col_names_to_idx(self.cat_cols)),
                ])
            ),
            ('impute',SimpleImputer(strategy='most_frequent')),
            ('ohe', 
                OneHotEncoder(handle_unknown='infrequent_if_exist')
            ),
        ])
        
        combine = CustomFeatureUnion([
            ('num',num_pipe),
            ('bin',bin_pipe),
            ('cat',cat_pipe)
        ])
        combine._set_feature_names(
            self.num_cols,
            self.bin_cols,
            self.cat_cols,
            is_pca = True
        )
        
        return combine

def to_binary(data:np.ndarray)->np.ndarray:
    """Utility function to convert booleans to integers since certain
    sklearn transformers don't play well with booleans for some reason

    Args:
        data (np.ndarray): input data with all values as booleans

    Returns:
        np.ndarray: data transformed with all values to 1 for true 
        and 0 for false
    """
    return data.astype(int)