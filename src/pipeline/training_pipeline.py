import numpy as np          
import pandas as pd     
import sys

from src.exception import CustomException
from src.logger import get_logger

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

logger = get_logger('training_pipeline')

def training_pipeline():
    logger.info('Training Pipeline Initializing')

    logger.info('Data Ingestion initialized')
    #? Initializing Data Ingestion
    data_ingestion = DataIngestion('notebooks/data/StudentsPerformance.csv')
    train_path, test_path = data_ingestion.load_data()

    logger.info('Data Preprocessing Initialized')
    #? Initialzing Data Preprocessing
    data_preprocessing = DataPreprocessing(train_path, test_path)

    # Loading training and testing data
    X_train, X_test, y_train, y_test = data_preprocessing.load_data()

    # Preprocessing our data
    X_train, X_test, y_train, y_test = data_preprocessing.apply_preprocessing(X_train, X_test, y_train, y_test)

    logger.info('Model Training Initialized')
    #? Model Training Initialized
    model_training = ModelTrainer(X_train, X_test, y_train, y_test)
    
    # Gives Trained Models
    trained_models = model_training.start_training()

    logger.info('Model Evaluation Initialized')
    #? Model Evaluation Initialized

    model_evaluate = ModelEvaluation()

    #Start Evaluating and gives us best model
    model_evaluate.evaluate(X_train, X_test, y_train, y_test, trained_models)

    logger.info('Pipeline Completed')

if __name__ == '__main__':
    training_pipeline()