from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import os
import requests

app = Flask(__name__)

# URLs to your model and scaler files
MODEL_URL = 'https://github.com/bhushanasati25/appsapi/blob/main/risk_stratification_model.h5
SCALER_URL = 'https://github.com/bhushanasati25/appsapi/blob/main/scaler.pkl'

# Paths where the files will be saved
MODEL_PATH = 'risk_stratification_model.h5'
SCALER_PATH = 'scaler.pkl'

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        raise Exception(f'Failed to download file from {url}')

# Check if the model and scaler files exist; if not, download them
if not os.path.exists(MODEL_PATH):
    print('Downloading model...')
    download_file(MODEL_URL, MODEL_PATH)

if not os.path.exists(SCALER_PATH):
    print('Downloading scaler...')
    download_file(SCALER_URL, SCALER_PATH)

# Load the model and scaler
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Mapping of classes to risk levels
risk_mapping = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}

@app.route('/predict', methods=['POST'])
def predict():
    # Your existing predict function remains the same
    try:
        # Get input data from the request
        data = request.get_json(force=True)
        input_features = np.array(data['input'])

        # Scale the inputs
        input_features_scaled = scaler.transform(input_features)

        # Make predictions
        predictions = model.predict(input_features_scaled)

        # Get the predicted classes
        predicted_classes = np.argmax(predictions, axis=1)

        # Map the predicted classes to risk levels
        predicted_risks = [risk_mapping[cls] for cls in predicted_classes]

        # Return the predictions as JSON
        return jsonify({'predicted_risks': predicted_risks})
    except Exception as e:
        # Handle exceptions and return error message
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()
