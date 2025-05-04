import pandas as pd
import os
import joblib
import logging
import sys
import yaml
import mlflow
import json
import dagshub

from sklearn.linear_model import Lasso
from mlflow.models.signature import infer_signature
from utils import *

# Initialize MLflow with DagsHub
mlflow.set_tracking_uri("https://dagshub.com/iamprashantjain/MLOps_UsedCarPricePrediction.mlflow")
dagshub.init(repo_owner='iamprashantjain', repo_name='MLOps_UsedCarPricePrediction', mlflow=True)

# Load parameters
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)
model_alpha = params["model_training"]["model_alpha"]


def read_data(source_type, train_path):
    if source_type == 'path':
        try:
            df = pd.read_csv(train_path)
            logging.info(f"Data read successfully from {train_path}")
            return df
        except Exception as e:
            logging.error("Failed to read data", exc_info=True)
            raise CustomException(e, sys)
    else:
        logging.info("Other source type not configured yet")


def apply_preprocessor(df, preprocessor_path, selected_features_path):
    try:
        preprocessor = joblib.load(preprocessor_path)
        feature_mask = joblib.load(selected_features_path)

        X_transformed = preprocessor.transform(df)
        X_selected = X_transformed[:, feature_mask]

        return X_selected, preprocessor, feature_mask
    except Exception as e:
        raise CustomException(e, sys)


def train_lasso_model(X, y, alpha):
    try:
        model = Lasso(alpha=alpha, random_state=42)
        model.fit(X, y)
        logging.info("Lasso model training complete.")
        return model
    except Exception as e:
        raise CustomException(e, sys)


def save_model_artifacts(model, X_selected, selected_feature_names, base_path="artifacts/model"):
    try:
        os.makedirs(base_path, exist_ok=True)
        model_path = os.path.join(base_path, "model.pkl")
        joblib.dump(model, model_path)

        # Save selected feature names
        with open(os.path.join(base_path, "feature_names.json"), "w") as f:
            json.dump(selected_feature_names, f)

        # MLflow logging
        input_example = pd.DataFrame(X_selected[:5], columns=selected_feature_names)
        with mlflow.start_run() as run:
            signature = infer_signature(X_selected, model.predict(X_selected))
            mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)
            mlflow.log_param("alpha", model.alpha)
            mlflow.log_metric("num_features", X_selected.shape[1])

            model_info = {
                "run_id": run.info.run_id,
                "model_path": "model"
            }
            with open(os.path.join(base_path, "model_info.json"), "w") as f:
                json.dump(model_info, f)

        logging.info("Model and metadata logged with MLflow.")
        return model_path
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Load raw train data (with target)
        train_path = r"D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\artifacts\transformed_data\train\train.csv"
        train_df = read_data('path', train_path)

        X = train_df.drop(columns=["listingPrice"])
        y = train_df["listingPrice"]

        # Load preprocessing and feature selection artifacts
        preprocessor_path = "artifacts/preprocessor/preprocessor.pkl"
        selected_features_path = "artifacts/preprocessor/selected_features.pkl"
        X_selected, preprocessor, mask = apply_preprocessor(X, preprocessor_path, selected_features_path)

        # Get selected feature names
        all_feature_names = preprocessor.get_feature_names_out()
        selected_feature_names = list(all_feature_names[mask])
        logging.info(f"Selected Features: {selected_feature_names}")

        # Train final model
        model = train_lasso_model(X_selected, y, model_alpha)

        # Save model and log to MLflow
        save_model_artifacts(model, X_selected, selected_feature_names)

    except Exception as e:
        raise CustomException(e, sys)