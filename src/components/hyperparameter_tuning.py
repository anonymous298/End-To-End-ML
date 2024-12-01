import os
import sys

from src.exception import CustomException
from src.logger import get_logger

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

logger = get_logger('hyperparameter')

class HyperParameterTuning:
    def __init__(self):
        pass
    
    def initiate_tuning(self, X_train, X_test, y_train, y_test):
        logger.info('Hyperparameter started')
        params = {
    'LinearRegression': {
        'fit_intercept': [True, False],
        'positive': [True, False]
    },
    'SVR': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'epsilon': [0.1, 0.2, 0.5]
    },
    'KNeighborsRegressor': {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1 for Manhattan, 2 for Euclidean distance
    },
    'DecisionTreeRegressor': {
        'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    },
    'RandomForestRegressor': {
        'n_estimators': [50, 100, 200],
        'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }
}


        try:
            tuned_model_scores = {}

            models = {
                'LinearRegression' : LinearRegression(),
                'SVR' : SVR(),
                'KNeighborsRegressor' : KNeighborsRegressor(),
                'DecisionTreeRegressor' : DecisionTreeRegressor(),
                'RandomForestRegressor' : RandomForestRegressor()
            }

            for i in range(len(models)):

                model = list(models.values())[i]
                para = params[list(models.keys())[i]]

                grid = GridSearchCV(
                    model, 
                    para,
                    cv=3,
                    verbose=3,
                    n_jobs=-1,
                    scoring='r2'
                )

                logger.info(f'Tuning our model: {model}')
                grid.fit(X_train, y_train)

                model.set_params(**grid.best_params_)
                model.fit(X_train, y_train)

                y_pred_test = model.predict(X_test)

                r2 = r2_score(y_test, y_pred_test)

                tuned_model_scores[list(models.keys())[i]] = r2

                logger.info('Hyperparameter Completed')

            return tuned_model_scores

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)