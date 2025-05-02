import pandas as pd
import os
import joblib
from utils import *
from datetime import date
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
import yaml
import mlflow
import json
import dagshub
import mlflow.models
from mlflow.models.signature import infer_signature


# Initialize DagsHub MLflow URI
mlflow.set_tracking_uri("https://dagshub.com/iamprashantjain/MLOps_UsedCarPricePrediction.mlflow")
dagshub.init(repo_owner='iamprashantjain', repo_name='MLOps_UsedCarPricePrediction', mlflow=True)


with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

model_alpha = params["model_training"]["model_alpha"]

# 1. Read training data
def read_training_data(path):
    try:
        df = pd.read_csv(path)
        print(df.columns)
        logging.info(f"Training data read from {path}, shape: {df.shape}")
        return df
    except Exception as e:
        raise CustomException(e, sys)


# 2. Train model using Lasso + RFECV
def train_model(df, model_alpha):
    try:
        X = df.drop(columns=["listingPrice"])
        y = df["listingPrice"]

        lasso = Lasso(alpha=model_alpha, random_state=42)
        rfecv = RFECV(estimator=lasso, step=1, cv=2, scoring='neg_mean_squared_error', n_jobs=-1)
        rfecv.fit(X, y)

        logging.info(f"Model training complete. Optimal features: {rfecv.n_features_}")
        return rfecv
    except Exception as e:
        raise CustomException(e, sys)


# 3. Save model and log to MLflow in dvc pipeline since best model with best param is selected for dvc pipeline
def save_and_log_model(model, X, base_path="artifacts/model"):
    try:
        os.makedirs(base_path, exist_ok=True)

        model_path = os.path.join(base_path, "model.pkl")
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")

        # Start MLflow run and log model
        with mlflow.start_run() as run:
            signature = infer_signature(X, model.predict(X))
            mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X.iloc[:5])

            mlflow.log_param("alpha", model.estimator_.alpha)
            mlflow.log_metric("n_features_selected", model.n_features_)

            # Save model info for registration
            model_info = {
                "run_id": run.info.run_id,
                "model_path": "model"
            }
            with open(os.path.join(base_path, "model_info.json"), "w") as f:
                json.dump(model_info, f)
            logging.info("MLflow model info saved.")

        return model_path
    except Exception as e:
        raise CustomException(e, sys)
    

# Run training
if __name__ == "__main__":
    train_path = r"D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\artifacts\transformed_data\train\train.csv"
    df = read_training_data(train_path)
        
    model = train_model(df, model_alpha)

    # Pass X (the feature data) along with the model to save_and_log_model
    X = df.drop(columns=["listingPrice"])  # This is the feature matrix
    save_and_log_model(model, X)