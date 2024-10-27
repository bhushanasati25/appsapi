from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = tf.keras.models.load_model('risk_stratification_model.h5')
scaler = joblib.load('scaler.pkl')

# Mapping of classes to risk levels
risk_mapping = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}

@app.route('/predict', methods=['POST'])
def predict():
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
