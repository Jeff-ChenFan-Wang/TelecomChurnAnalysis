from ctypes import ArgumentError
import pandas as pd
import numpy as np
from sklearn.pipeline import NotFittedError, Pipeline, FeatureUnion, TransformerMixin
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
from sklearn.cluster import MiniBatchKMeans

class CustomFeatureUnion(FeatureUnion):
    """Customized FeatureUnion class that allows access to transformers 
    via name like sklearn Pipelines and a working get_feature_names_out() 
    since as of v1.1 pipelines feature names is still broken and featureUnion
    just didn't implement it at all
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
            self,num_cols:np.ndarray[str], bin_cols:np.ndarray[str],
            cat_cols:np.ndarray[str], engin_cols:np.ndarray[str]):
        """Retains feature names so that feature_names_out() can
        function

        Args:
            num_cols (np.ndarray[str]): numerical column names
            bin_cols (np.ndarray[str]): binary column names
            cat_cols (np.ndarray[str]): categorical column names
            engin_cols (np.ndarray[str]): column names of engineered feats
        """
        self.num_cols = num_cols
        self.bin_cols = bin_cols
        self.cat_cols = cat_cols
        self.engin_cols = engin_cols
    
    def get_feature_names_out(self)->np.ndarray:
        """Returns feature names of each column in order

        Returns:
            np.ndarray: feature names in array like form
        """
        if self.__sklearn_is_fitted__():
            col_list = np.hstack([
                self.num_cols,
                self.bin_cols,
                (
                    dict(self.transformer_list)['cat']
                    .named_steps['ohe'].get_feature_names_out(self.cat_cols)
                )
            ])
            
            if self.engin_cols is not None:
                col_list = np.hstack([col_list,self.engin_cols])
            
            return col_list

class PipelineFactory:
    """Factory class for creating sklearn preprocessing pipelines with a 
    working get_feature_names_out() method since as of v1.1.2 it is still
    broken
    """
    #list all engineered feature names here
    ENGINEERED_FEAT_NAMES = ['internetServicesSubbed','KmeansLabel']
    
    def __init__(self, raw_data:pd.DataFrame):
        """initilizes the factory by determining which columns are categorical
        and which are numerical so output pipeline can make appropriate 
        transformations

        Args:
            raw_data (pd.DataFrame): raw data for pipelines to be fitted against
                data frame mus have correct dtypes for the class to recognize
                which are numerical, categorical, and binary columns
        """     
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
            
    def create_pipe(self, intrnt_sub_list:List[str]=None, *, 
                    engineer:bool, random_seed:int, normalize:bool
            )->CustomFeatureUnion:
        """creates a feature union of pipelines 
                
        Args:
            engineer (bool): whether to feature engineer additional columns
            random_seed (int): integer for seeding KMeans 
            normalize (bool): Whether to normalize numerical features
            intrnt_sub_list (List[str]): list of columns to engineer the 
                "number of internet services subbed" column
                
        Returns:
            Pipeline: sklearn preprocessing pipeline 
        """
        num_pipe = self._make_numerical_pipe(normalize)
        bin_pipe = self._make_bin_pipe()
        cat_pipe = self._make_cat_pipe()
        num_intrnt_serv_pipe = self._make_bool_count_pipe(intrnt_sub_list)
        
        pipe_list = [
            ('num',num_pipe),
            ('bin',bin_pipe),
            ('cat',cat_pipe),
        ]
        
        if engineer:
            kmeans_pipe = self._make_kmeans_pipe(
                original_transformers = FeatureUnion(pipe_list),
                random_seed=random_seed
            )
            
            pipe_list.append(('num_intrnt',num_intrnt_serv_pipe))
            pipe_list.append(('kmeans',kmeans_pipe))
            engin_cols = self.ENGINEERED_FEAT_NAMES
        else:
            engin_cols = None

        combined = CustomFeatureUnion(pipe_list)
        combined._set_feature_names(
            self.num_cols,
            self.bin_cols,
            self.cat_cols,
            engin_cols
        )
        return combined
    
    def _make_numerical_pipe(self, normalize:bool)->Pipeline:
        """make a sklearn pipeline for numerical features

        Args:
            normalize (bool): whether to normalize features

        Returns:
            Pipeline: pipeline to be used in feature union
        """
        if normalize:
            num_pipe = Pipeline([
                ('select_num', 
                    ColumnTransformer([
                        ('select_num','passthrough', 
                            self.col_names_to_idx(self.num_cols)),
                    ])
                ),
                ('scale',StandardScaler()),
            ])
        else:
            num_pipe = Pipeline([
                ('select_num', 
                    ColumnTransformer([
                        ('select_num','passthrough', 
                            self.col_names_to_idx(self.num_cols)),
                    ])
                )
            ])
        return num_pipe
    
    def _make_bin_pipe(self)->Pipeline:
        """make a sklearn pipeline for binary features

        Returns:
            Pipeline: pipeline to be used in feature union
        """
        bin_pipe = Pipeline([
            ('select_bin', 
                ColumnTransformer([
                    ('select_bin','passthrough', 
                        self.col_names_to_idx(self.bin_cols)),
                ])
            ),
            ('convert',FunctionTransformer(to_binary)),
        ])
        return bin_pipe
    
    def _make_cat_pipe(self)->Pipeline:
        """make a sklearn pipeline for categorical features

        Returns:
            Pipeline: pipeline to be used in feature union
        """
        cat_pipe = Pipeline([
            ('select_cat', 
                ColumnTransformer([
                    ('select_cat','passthrough', 
                        self.col_names_to_idx(self.cat_cols)),
                ])
            ),
            ('ohe', 
                OneHotEncoder(handle_unknown='infrequent_if_exist')
            ),
        ])
        return cat_pipe
    
    def _make_bool_count_pipe(self, feat_names:List[str])->Pipeline:
        """pipeline that counts number of true values in each row

        Args:
            feat_names (List[str]): list of boolean columns to be counted

        Returns:
            Pipeline: pipeline to be used in feature union
        """
        count_pipe = Pipeline([
            ('select_list', 
                ColumnTransformer([
                    ('select_list','passthrough', 
                        self.col_names_to_idx(feat_names )),
                ])
            ),
            ('row_sum', 
                FunctionTransformer(sum_binaries)
            ),
        ])
        return count_pipe
    
    def _make_kmeans_pipe(self, original_transformers:FeatureUnion,
                          random_seed:int)->Pipeline:
        kmeans_pipe = Pipeline(
            original_transformers,
            ('kmeans',MiniBatchKMeans(n_clusters=8,random_state=random_seed))
        )
        return kmeans_pipe

def to_binary(data:np.ndarray[bool])->np.ndarray[int]:
    """Utility function to convert booleans to integers since certain
    sklearn transformers don't play well with booleans for some reason

    Args:
        data (np.ndarray): input data with all values as booleans

    Returns:
        np.ndarray: data transformed with all values to 1 for true 
        and 0 for false
    """
    return data.astype(int)

def sum_binaries(bool_arr:np.ndarray[bool])->np.ndarray[int]:
    """row-wise summation of all boolean columns and returns result
    as a single column to see how many true values in each row

    Args:
        bool_arr (np.ndarray[bool]): array of booleans 

    Returns:
        np.ndarray[int]: horizontal sum of each row 
    """
    return bool_arr.sum(axis=1).reshape([-1,1])