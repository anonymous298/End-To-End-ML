import numpy as np          
import pandas as pd   
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = get_logger('Model Evaluation')

class ModelEvaluation:
    def __init__(self, models):
        self.models = models

    def train_evaluation(self, X_train, y_train):
        logger.info('Training Evaluation Started')

        model_train_evaluation = {}

        for model_name, model in self.models.items():
            logger.info(f'Started Evaluation For {model_name}')
            model = model
            
            try:
                y_pred_train = model.predict(X_train)

            except Exception as e:
                logger.error(e)
                raise CustomException(e, sys)

            try:   
                mse = mean_squared_error(y_train, y_pred_train)
                mae = mean_absolute_error(y_train, y_pred_train)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_train, y_pred_train)

            except Exception as e:
                logger.error(e)
                raise CustomException(e, sys)

            model_train_evaluation[model_name] = {
                'mse' : mse,
                'mae' : mae,
                'rmse' : rmse,
                'r2' : r2
            }
        
        logger.info('Training Evaluation Completed')

        return model_train_evaluation

    def test_evaluation(self, X_test, y_test):
        logger.info('Testing Evaluation Started')

        model_test_evaluation = {}

        for model_name, model in self.models.items():
            logger.info(f'Started Evaluation For {model_name}')
            model = model
            
            try:
                y_pred_test = model.predict(X_test)

            except Exception as e:
                logger.error(e)
                raise CustomException(e, sys)

            try:   
                mse = mean_squared_error(y_test, y_pred_test)
                mae = mean_absolute_error(y_test, y_pred_test)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred_test)

            except Exception as e:
                logger.error(e)
                raise CustomException(e, sys)

            model_test_evaluation[model_name] = {
                'mse' : mse,
                'mae' : mae,
                'rmse' : rmse,
                'r2' : r2
            }
        
        logger.info('Testing Evaluation Completed')

        return model_test_evaluation
