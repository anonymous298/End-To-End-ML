import os 
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import get_logger

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

logger = get_logger('Data_Ingestion')

@dataclass 
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self, file_path):
        self.data_ingestion_config = DataIngestionConfig()
        self.file_path = file_path

    def load_data(self):
        logger.info('Started loading data')
        try:
            df = pd.read_csv(self.file_path)
            logger.info('Data loaded successfully')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)
            logger.info('Saving raw data to artifacts/data')

            train, test = train_test_split(df, test_size=0.2, random_state=42)

            train.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            logger.info('Saving train data to artifact/train')

            test.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            logger.info('Saving test data to artifacts/test')

            logger.info('Data Ingestion Completed')

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)
