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
    logger.info(f'Saving {model} to {file_path}')

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

    train_data[COL_NAME] = train_data['math score'] + train_data['reading_score'] + train_data['writing_score']
    test_data[COL_NAME] = test_data['math score'] + test_data['reading_score'] + test_data['writing_score']

    return (
        train_data,
        test_data
    )