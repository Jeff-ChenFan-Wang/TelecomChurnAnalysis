import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, 
    OneHotEncoder, 
    FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from typing import List

class customPipe(Pipeline):
    """custom subclass of the sklearn Pipeline that has a working
    get_feature_names_out() method since the official one is still broken as of 
    version 1.1.2
    """
    def get_feature_names_out(self, input_features=None):
        return self.steps[-1][1].get_feature_names_out()
    
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
        self.radom_seed=radom_seed 
        
        self.full_col_order = raw_data.columns
        self.cat_cols = raw_data.select_dtypes(include='object').columns
        self.num_cols = raw_data.select_dtypes(include=np.number).columns
        self.bin_cols = raw_data.select_dtypes(include=bool).columns

    def get_col_indices(self, col_names:np.ndarray[str])->np.ndarray[int]:
        return np.in1d(self.full_col_order, col_names).nonzero()[0]
    
    def create_pipe(self, pca=False, *, impute:bool, normalize:bool)->customPipe:
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
            return 
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
        PCA's the result. Since 

        Returns:
            Pipeline: sklearn preprocessing pipeline 
        """
        pipe = Pipeline([
            ('impute',SimpleImputer(strategy='most_frequent')),
            (
                'columnTransform',
                ColumnTransformer([
                    (
                        'categorical_vars',
                        OneHotEncoder(handle_unknown='infrequent_if_exist'),
                        self.cat_cols
                    ),
                    ('numeric_vars',StandardScaler(),self.num_cols),
                    ('binary_vars','passthrough',self.bin_cols)
                ])
            ),
            ('PCA',PCA(random_state=self.radom_seed))
            
        ])
    
    def make_impute_ohe_scale_pipe(self)->customPipe:
        """creates a pipeline that imputes data using mode impuation, then
        normalizes and one-hot encodes appropriate columns

        Returns:
            customPipe: basically a sklearn pipeline but with a functioning
                get_feature_names_out() method.
        """
        
        pipe = customPipe([
            ('impute',SimpleImputer(strategy='most_frequent')),
            (
                'columnTransform',
                ColumnTransformer([
                    (
                        'categorical_vars',
                        OneHotEncoder(handle_unknown='infrequent_if_exist'),
                        self.get_col_indices(self.cat_cols)
                    ),
                    (
                        'numeric_vars',StandardScaler(),
                        self.get_col_indices(self.num_cols)
                    ),
                    (
                        'binary_vars','passthrough',
                        self.get_col_indices(self.bin_cols)
                    )
                ])
            )
        ])

        self.full_col_order = np.hstack([self.cat_cols,self.num_cols,self.bin_cols])
        
        return pipe
    
    def make_impute_ohe_pipe(self)->customPipe:
        """creates a pipeline that imputes data using mode impuation, then 
        one-hot encodes appropriate columns but does not normalize numerical
        columns

        Returns:
            customPipe: basically a sklearn pipeline but with a functioning
                get_feature_names_out() method.
        """
        pipe = customPipe([
            ('impute',SimpleImputer(strategy='most_frequent')),
            (
                'retainFeatureName',
                FunctionTransformer(
                    lambda x: pd.DataFrame(x, columns=self.full_col_order)
                )
            ),
            (
                'columnTransform',
                ColumnTransformer([
                    (
                        'categorical_vars',
                        OneHotEncoder(handle_unknown='infrequent_if_exist'),
                        self.cat_cols
                    ),
                    ('numeric_vars','passthrough',self.num_cols),
                    ('binary_vars','passthrough',self.bin_cols)
                ])
            )
        ])
        return pipe
    
    def make_ohe_pipe(self)->customPipe:
        """creates a pipeline that one-hot encodes appropriate columns but 
        does not normalize numerical columns nor impute any missing data.

        Returns:
            customPipe: basically a sklearn pipeline but with a functioning
                get_feature_names_out() method.
        """
        pipe = customPipe([
             (
                'columnTransform',
                ColumnTransformer([
                    (
                        'categorical_vars',
                        OneHotEncoder(handle_unknown='infrequent_if_exist'),
                        self.cat_cols
                    ),
                    ('numeric_vars','passthrough',self.num_cols),
                    ('binary_vars','passthrough',self.bin_cols)
                ])
            )
        ])
        return pipe