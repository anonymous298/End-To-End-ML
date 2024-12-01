import numpy as np           
import pandas as pd    
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

logger = get_logger('Model_trainer')

@dataclass
class ModelConfig:
    linear_regression: LinearRegression = LinearRegression()
    support_vector_regressor: SVR = SVR()
    k_nearest_neighbors: KNeighborsRegressor = KNeighborsRegressor()
    decision_tree: DecisionTreeRegressor = DecisionTreeRegressor()
    random_forest_regressor: RandomForestRegressor = RandomForestRegressor()

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        logger.info('Initialzing Models')
        self.model_config = ModelConfig()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def start_training(self):
        logger.info('Starting Model Training')

        #! Linear Regression
        logger.info('Training Linear Regression Model')
        try:
            lr_model = self.model_config.linear_regression
            lr_model.fit(self.X_train, self.y_train)

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        
        #! Support Vector Regressor
        logger.info('Training Support Vector Regressor')
        try:
            svr_model = self.model_config.support_vector_regressor
            svr_model.fit(self.X_train, self.y_train)

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        
        #! K-Nearest Neighbor
        logger.info('Training K-Nearest Neighbor')
        try:
            knn_model = self.model_config.k_nearest_neighbors
            knn_model.fit(self.X_train, self.y_train)

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        
        #! Decision Tree
        logger.info('Training Decision Tree')
        try:
            dtree_model = self.model_config.decision_tree
            dtree_model.fit(self.X_train, self.y_train)

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        
        #! Random Forest
        logger.info('Training Random Forest')
        try:
            rf_model = self.model_config.random_forest_regressor
            rf_model.fit(self.X_train, self.y_train)

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
        
        logger.info('Model Training Completed')

        return {
            'linear regression' : lr_model,
            'support vector regressor' : svr_model,
            'k-nearest neigbor' : knn_model,
            'decision tree' : dtree_model,
            'random forest' : rf_model
        }

    
