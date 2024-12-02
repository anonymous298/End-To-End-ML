import os
import sys
import dill

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger('utils')

def save_object(file_path, model):
    '''
    this function is meant to save our model object to specific file path
    '''
    logger.info(f'Saving Model')

    try:
        
        dirname = os.path.dirname(file_path)

        os.makedirs(dirname, exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(model, f)

    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)

def create_column(train_data, test_data):
    COL_NAME = 'total_score'

    train_data[COL_NAME] = train_data['math score'] + train_data['reading score'] + train_data['writing score']
    test_data[COL_NAME] = test_data['math score'] + test_data['reading score'] + test_data['writing score']

    return (
        train_data,
        test_data
    )

def load_model(file_path):
    try:
        logger.info('Opening Model')

        with open(file_path, 'rb') as f:
            model = dill.load(f)

        logger.info('Model Loaded')

        return model
    
    except Exception as e:
        logger.error(e)
        raise CustomException(e, sys)