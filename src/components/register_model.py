import json
import mlflow
from utils import *
import os
import dagshub

# Set MLflow tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/iamprashantjain/MLOps_UsedCarPricePrediction.mlflow")
dagshub.init(repo_owner='iamprashantjain', repo_name='MLOps_UsedCarPricePrediction', mlflow=True)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info('Model info loaded from %s', file_path)
        return model_info
    except Exception as e:
        raise CustomException(e, sys)

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging"
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logging.info(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        raise CustomException(e, sys)

def main():
    try:
        model_info_path = 'artifacts/model/model_info.json'  # <-- Make sure this file is created during training
        model_info = load_model_info(model_info_path)

        model_name = "lass_rfecv"
        register_model(model_name, model_info)
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    main()