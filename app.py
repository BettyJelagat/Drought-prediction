from flask import Flask, render_template, request, jsonify

import pandas as pd
import numpy as np

import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# load the model
with open('model/weather_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

#load the scaler
with open('model/scaler_model.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        parameters = [
            float(request.form['temp_max']),
            float(request.form['temp_min']),
            float(request.form['precip']),
            float(request.form['Humidity']),
            float(request.form['w_speed']),
            float(request.form['s_wetness']),

        ]
        parameters_scaled = scaler.transform([parameters])

        prediction = model.predict([parameters_scaled])

        proba = model.predict_proba(parameters_scaled)

        max_proba = f'{round(np.max(proba)*100, 2)}%'

        return jsonify({
            'prediction': prediction[0],
            'probability': max_proba

        })
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__=='__main__':
    app.run(debug= True)  