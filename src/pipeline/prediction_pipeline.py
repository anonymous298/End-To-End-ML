import os
import sys
import numpy as np          
import pandas as pd     

from src.exception import CustomException
from src.logger import get_logger
from src.utils import load_model

logger = get_logger('prediction')

class PredictionPipelineConfig:
    model_path: str = os.path.join('model','model.pkl')
    preprocessor_path: str = os.path.join('model','preprocessor.pkl')

class PredictionPipeline:
    def __init__(self):
        self.prediction_config = PredictionPipelineConfig()

    def take_prediction(self, dataframe):
        try:
            preprocessor = load_model(self.prediction_config.preprocessor_path)
            model = load_model(self.prediction_config.model_path)

            logger.info("Applying transformation and prediction")
            X = preprocessor.transform(dataframe)
            prediction = model.predict(X)

            logger.info('Prediction Complete')

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

        return prediction

class CustomData:
    def __init__(
            self,
            gender,
            race_ethnicity,
            parental_level_of_education,
            lunch,
            test_preparation_course,
            math_score,
            reading_score,
            writing_score,
    ):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.math_score = math_score
        self.reading_score = reading_score
        self.writing_score = writing_score

    def convert_data_to_dataframe(self):
        input_qurie = {
                'gender' : [self.gender],
                'race/ethnicity' : [self.race_ethnicity],
                'parental level of education' : [self.parental_level_of_education],
                'lunch' : [self.lunch],
                'test preparation course' : [self.test_preparation_course],
                'math score' : [self.math_score],
                'reading score' : [self.reading_score],
                'writing score' : [self.writing_score] 
            }

        logger.info('Converting inputs into DataFrame')
        input_df = pd.DataFrame(input_qurie)

        return input_df

def main():
    prediction_pipeline = PredictionPipeline()
    total_score = prediction_pipeline.take_prediction()

    print(f'Student Predicted Total Score is: {total_score}')

if __name__ == '__main__':
    main()    
