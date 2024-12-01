import numpy as np          
import pandas as pd   
import sys
import os

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import get_logger
from src.utils import save_object

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = get_logger('Model Evaluation')

@dataclass
class ModelEvaluationConfig:
    model_path: str = os.path.join('model', 'model.pkl')

class ModelEvaluation:
    def __init__(self):
        self.model_eval_config = ModelEvaluationConfig()

    def evaluate(self, X_train, X_test, y_train, y_test, models):
        try:
            model_score = {}
            
            logger.info('Evaluating Models Performance')
            for mod_name, mod in models.items():
                model = mod
                y_pred_test = model.predict(X_test)

                r2 = r2_score(y_test, y_pred_test)

                model_score[mod_name] = r2

            best_model_score = max(sorted(list(model_score.values())))

            best_model_name = list(model_score.keys())[
                list(model_score.values()).index(best_model_score)
            ]

            if best_model_score < 0.6:
                raise CustomException('Best Model Not Found', sys)
            
            logger.info('Saving Best Model')
            save_object(
                file_path=self.model_eval_config.model_path,
                model=models[best_model_name]
            )
            
            print(f'Best Model -> {best_model_name} -> Score: {best_model_score}')
            logger.info('Model Evaluation Completed')

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)


