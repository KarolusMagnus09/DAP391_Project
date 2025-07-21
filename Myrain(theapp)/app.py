from flask import Flask, request, render_template, jsonify
import xgboost as xgb
import pickle
import os
import numpy as np
import pandas as pd
import requests
import urllib.parse
import logging

app = Flask(__name__)

#XGBoost model
model_path = os.path.join('model', 'xgb_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')


# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract input features in the same order as training
        features = [
            data['humidi'], data['cloud'], data['pressure'], data['cloud_humid'],
            data['is_rainy_season'], data['is_dry_season'], data['max_temp'],
            data['min_temp'], data['range_temp'], data['rain_1d_ago'],
            data['rain_2d_ago'], data['rain_trend_3d'], data['rain_intensity'],
            data['wind_x'], data['wind_y'], data['Longitude'], data['Latitude']
        ]

        # Convert to NumPy array for prediction
        input_array = np.array([features], dtype=np.float32)

        # Predict using the model (avoid DMatrix)
        prediction_log = model.predict(input_array)[0]  # Get log-transformed prediction
        prediction = np.expm1(prediction_log)  # Exponentiate back to original scale

        print(f"Prediction (log): {prediction_log}, Prediction (original): {prediction}")  # Debug output

        return jsonify({'rain': float(prediction)})  # Return only the float value
    except Exception as e:
        print(f"Error: {str(e)}")  # Debug error
        return jsonify({'error': str(e)}), 400
#csv
corvn_df = pd.read_csv('corvn.csv')  

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/geocode', methods=['POST'])
def geocode():
    data = request.get_json()
    location = data.get('location')
    if not location:
        return jsonify({'error': 'Location name is required'}), 400

    try:
        response = requests.get(
            f'https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(location)}&format=json&limit=1',
            headers={'User-Agent': 'RainfallPredictionApp/1.0'}
        )
        response.raise_for_status()
        results = response.json()
        if not results:
            return jsonify({'error': 'Location not found'}), 404
        return jsonify({'lat': results[0]['lat'], 'lon': results[0]['lon']})
    except requests.RequestException as e:
        logger.error("Error in /geocode endpoint: %s", str(e))
        return jsonify({'error': f'Failed to fetch coordinates: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)