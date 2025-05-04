import pandas as pd
import numpy as np
import joblib
import sys
import mlflow
from dagshub import init
from utils import *

# Paths for preprocessing artifacts
PREPROCESSOR_PATH = "artifacts/preprocessor/preprocessor.pkl"
FEATURE_MASK_PATH = "artifacts/preprocessor/selected_features.pkl"

# Set MLflow tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/iamprashantjain/MLOps_UsedCarPricePrediction.mlflow")

# Initialize DagsHub
init(repo_owner='iamprashantjain', repo_name='MLOps_UsedCarPricePrediction', mlflow=True)

def apply_preprocessing_and_selection(df):
    """
    Apply preprocessing and feature selection to the input DataFrame.
    """
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        feature_mask = joblib.load(FEATURE_MASK_PATH)

        X_transformed = preprocessor.transform(df)
        X_selected = X_transformed[:, feature_mask]

        return X_selected
    except Exception as e:
        raise CustomException(e, sys)

def load_model(model_name="used_car_price_model"):
    """
    Load the latest model from MLflow Model Registry using the model name.
    """
    try:
        # Get the latest version of the model
        client = mlflow.tracking.MlflowClient()

        # Fetch the model versions
        model_versions = client.get_latest_versions(model_name)

        # Get the latest version (you can filter by stage if you want)
        latest_model_version = model_versions[0]  # Assuming the first in the list is the latest

        print(f"Loading model {model_name} version {latest_model_version.version} from stage {latest_model_version.current_stage}")

        # Load the latest version of the model
        model_uri = f"models:/{model_name}/{latest_model_version.version}"
        model = mlflow.sklearn.load_model(model_uri)
        
        return model
    except Exception as e:
        raise CustomException(e, sys)

def main():
    try:
        # Load your dataset
        df = pd.read_csv(
            r"D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\artifacts\transformed_data\train\train.csv"
        )

        # Preprocess and select features
        X_final = apply_preprocessing_and_selection(df)

        # Load the latest model from MLflow
        model = load_model(model_name="lass_rfecv")

        # Make predictions
        predictions = model.predict(X_final)

        # Attach predictions to the dataframe
        df["PredictedPrice"] = np.round(predictions, 0).astype(int)
        print(df["PredictedPrice"])

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()