from flask import Flask, render_template, request
from src.utils import load_model
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData

import pandas as pd    

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form', methods=['GET','POST'])
def form():
    if request.method == 'POST':
        gender = request.form.get('gender')
        race_ethnicity = request.form.get('race/ethnicity')
        parental_level_of_education = request.form.get('parental level of education')
        lunch = request.form.get('lunch')
        test_preparation_course = request.form.get('test preparation course')
        math_score = request.form.get('math score')
        reading_score = request.form.get('reading score')
        writing_score = request.form.get('writing score')

        customdata = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                math_score=math_score,
                reading_score=reading_score,
                writing_score=writing_score
        )

        input_df = customdata.convert_data_to_dataframe()

        prediction_pipe = PredictionPipeline()
        prediction = prediction_pipe.take_prediction(input_df)

        return render_template('form.html', prediction=prediction)

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')