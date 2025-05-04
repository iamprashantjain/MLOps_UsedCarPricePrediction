import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import sys
import mlflow
from dagshub import init


# Paths to artifacts
PREPROCESSOR_PATH = "artifacts/preprocessor/preprocessor.pkl"
FEATURE_MASK_PATH = "artifacts/preprocessor/selected_features.pkl"

# Set MLflow tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/iamprashantjain/MLOps_UsedCarPricePrediction.mlflow")

# Initialize DagsHub
init(repo_owner='iamprashantjain', repo_name='MLOps_UsedCarPricePrediction', mlflow=True)

app = Flask(__name__)

# Apply preprocessing and feature selection
def apply_preprocessing_and_selection(df):
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        feature_mask = joblib.load(FEATURE_MASK_PATH)

        X_transformed = preprocessor.transform(df)
        X_selected = X_transformed[:, feature_mask]

        return X_selected
    except Exception as e:
        print(f"Error in preprocessing and feature selection: {e}")

# Load the pre-trained model from mlflow
def load_model(model_name="used_car_price_model"):
    """
    Load the latest model from MLflow Model Registry using the model name.
    """
    try:
        # Get the latest version of the model
        client = mlflow.tracking.MlflowClient()

        # Fetch the model versions
        model_versions = client.get_latest_versions(model_name)
        
        if not model_versions:
            raise Exception(f"Model '{model_name}' does not exist in the registry.")
        
        # Get the latest version (assuming the first in the list is the latest)
        latest_model_version = model_versions[0]

        print(f"Loading model {model_name} version {latest_model_version.version} from stage {latest_model_version.current_stage}")

        # Load the latest version of the model
        model_uri = f"models:/{model_name}/{latest_model_version.version}"
        model = mlflow.sklearn.load_model(model_uri)
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = request.form.to_dict()
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # List of columns that must be numeric
        numeric_cols = ['year', 'odometer', 'fitnessAge', '360DegreeCamera', 'AlloyWheels',
                        'AppleCarplayAndroidAuto', 'Bluetooth', 'CruiseControl', 'GpsNavigation',
                        'InfotainmentSystem', 'LeatherSeats', 'ParkingAssist', 'PushButtonStart',
                        'RearAc', 'SpecialRegNo', 'Sunroof/Moonroof', 'TopModel', 'Tpms',
                        'VentilatedSeats', 'featureCount', 'avgEmi', 'ownership']

        # Convert them to numeric
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check for missing data
        if df.isnull().any().any():
            return jsonify({'error': f'Missing or invalid data in: {df.columns[df.isnull().any()].tolist()}'}), 400

        # Apply preprocessing
        X = apply_preprocessing_and_selection(df)

        # Load the model
        model = load_model(model_name="lass_rfecv")
        if model is None:
            return jsonify({'error': 'Model not found in registry'}), 500

        # Make predictions
        prediction = model.predict(X)[0]

        # Return the prediction
        return jsonify({'PredictedPrice': round(float(prediction), 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
