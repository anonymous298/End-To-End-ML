import numpy as np    
import pandas as pd    
import sys
import os  
import pickle

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object, make_column

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = get_logger('Data_Preprocessing')

@dataclass
class DataPreprocessingConfig:
    train_path: str
    test_path: str
    preprocessor_path: str = os.path.join('model', 'preprocessor.pkl')

class DataPreprocessing:
    def __init__(self, train_path, test_path):
        logger.info('Initialzing train and test path to data preprocessing')
        self.data_preprocessing_config = DataPreprocessingConfig(train_path, test_path)

    
    def load_data(self):
        logger.info('fetching training and testing data')
        try: 
            train_data = pd.read_csv(self.data_preprocessing_config.train_path)
            test_data = pd.read_csv(self.data_preprocessing_config.test_path)

            logger.info('Splitting data into dependent and Independent')
            
            train_data, test_data = make_column(train_data, test_data)

            DEPENDENT_FEATURE = 'total_score'

            X_train = train_data.drop(DEPENDENT_FEATURE, axis=1)
            y_train = train_data[DEPENDENT_FEATURE]
            X_test = test_data.drop(DEPENDENT_FEATURE, axis=1)
            y_test = test_data[DEPENDENT_FEATURE]

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)


        return (
            X_train,
            X_test, 
            y_train,
            y_test
        )
    
    def apply_preprocessing(self, X_train, X_test, y_train, y_test):
        logger.info('Applying preprocessing to data')

        try:
            num_features = X_train.select_dtypes('number').columns
            cat_features = X_train.select_dtypes('object').columns

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

        logger.info('Initializing Numerical Features Transformation Pipeline')
        num_preprocessor = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]
        )

        logger.info('Initializing Categorical Features Transformation Pipeline')
        cat_preprocessor = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder()),
                ('scaler', StandardScaler())
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ('num_preprocessor', num_preprocessor, num_features),
                ('cat_preprocessor', cat_preprocessor, cat_features)
            ]
        )

        logger.info('transforming data through preprocessing pipeline')
        try:
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        
        save_object(
            file_path=self.data_preprocessing_config.preprocessor_path,
            model=preprocessor
        )

        logger.info('Data Preprocessing Completed')

        return X_train, X_test, y_train, y_test

