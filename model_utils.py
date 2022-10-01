import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
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
    """TODO
    """
    def get_transformer(self,transformer_name:str)->BaseEstimator:
        return dict(self.transformer_list)[transformer_name]

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
    
    def create_pipe(self, pca=False, *, impute:bool, normalize:bool)->Pipeline:
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

        Raises:
            Exception: impute or normalize isn't defined
            NotImplementedError: combination of impute or normalize hasn't been
                implemented.

        Returns:
            Pipeline: sklearn pipeline with necessary transformers according to 
                input parameters, and also a functioning feature names method.
        """
        if pca:
            return self.make_pca_impute_scale_pipe()
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
            
    def make_pca_impute_scale_pipe(self)->Pipeline:
        """creates a pipeline that imputes data using mode impuation, then
        normalizes and one-hot encodes appropriate columns, and finally
        PCA's the result. Since PCA does not retain feature names, we give 
        up on retaining feature names completely to make a more efficient
        pipeline instead. 

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
            ('pca', PCA(random_state=self.random_seed))
        ])
        
        bin_pipe = Pipeline([
            ('select_bin', 
                ColumnTransformer([
                    ('select_num','passthrough', 
                        self.col_names_to_idx(self.bin_cols)),
                ])
            ),
            ('convert',FunctionTransformer(to_binary)),
            ('impute',SimpleImputer(strategy='most_frequent')),
        ])
        
        cat_pipe = Pipeline([
            ('select_cat', 
                ColumnTransformer([
                    ('select_num','passthrough', 
                        self.col_names_to_idx(self.cat_cols)),
                ])
            ),
            ('impute',SimpleImputer(strategy='most_frequent')),
            ('ohe', 
                OneHotEncoder(handle_unknown='infrequent_if_exist')
            ),
        ])
        
        union = FeatureUnion([
            ('num',num_pipe),
            ('bin',bin_pipe),
            ('cat',cat_pipe)
        ])
        
        return union
    
    def make_impute_ohe_scale_pipe(self)->Pipeline:
        """creates a pipeline that imputes data using mode impuation, then
        normalizes and one-hot encodes appropriate columns

        Returns:
            customPipe: basically a sklearn pipeline but with a functioning
                get_feature_names_out() method.
        """
        
        pipe = Pipeline([
            ('impute',SimpleImputer(strategy='most_frequent')),
            (
                'columnTransform',
                ColumnTransformer([
                    (
                        'categorical_vars',
                        OneHotEncoder(handle_unknown='infrequent_if_exist'),
                        self.col_names_to_idx(self.cat_cols)
                    ),
                    (
                        'numeric_vars',StandardScaler(),
                        self.col_names_to_idx(self.num_cols)
                    ),
                    (
                        'binary_vars','passthrough',
                        self.col_names_to_idx(self.bin_cols)
                    )
                ])
            )
        ])

        self.original_cols = np.hstack([
            self.cat_cols,self.num_cols,self.bin_cols
        ])
        
        return pipe
    
    def make_impute_ohe_pipe(self)->Pipeline:
        """creates a pipeline that imputes data using mode impuation, then 
        one-hot encodes appropriate columns but does not normalize numerical
        columns

        Returns:
            customPipe: basically a sklearn pipeline but with a functioning
                get_feature_names_out() method.
        """
        pipe = Pipeline([
            ('impute',SimpleImputer(strategy='most_frequent')),
            (
                'retainFeatureName',
                FunctionTransformer(
                    lambda x: pd.DataFrame(x, columns=self.original_cols)
                )
            ),
            (
                'columnTransform',
                ColumnTransformer([
                    (
                        'categorical_vars',
                        OneHotEncoder(handle_unknown='infrequent_if_exist'),
                        self.col_names_to_idx(self.cat_cols)
                    ),
                    (
                        'numeric_vars','passthrough',
                        self.col_names_to_idx(self.num_cols)
                    ),
                    (
                        'binary_vars','passthrough',
                        self.col_names_to_idx(self.bin_cols)
                    )
                ])
            )
        ])

        self.original_cols = np.hstack([
            self.cat_cols,self.num_cols,self.bin_cols
        ])

        return pipe
    
    def make_ohe_pipe(self)->Pipeline:
        """creates a pipeline that one-hot encodes appropriate columns but 
        does not normalize numerical columns nor impute any missing data.

        Returns:
            customPipe: basically a sklearn pipeline but with a functioning
                get_feature_names_out() method.
        """
        pipe = Pipeline([
             (
                'columnTransform',
                ColumnTransformer([
                    (
                        'categorical_vars',
                        OneHotEncoder(handle_unknown='infrequent_if_exist'),
                        self.col_names_to_idx(self.cat_cols)
                    ),
                    (
                        'numeric_vars','passthrough',
                        self.col_names_to_idx(self.num_cols)
                    ),
                    (
                        'binary_vars','passthrough',
                        self.col_names_to_idx(self.bin_cols)
                    )
                ])
            )
        ])

        self.original_cols = np.hstack([
            self.cat_cols,self.num_cols,self.bin_cols
        ])

        return pipe

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