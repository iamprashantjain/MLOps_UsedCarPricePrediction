import pandas as pd
import os
import joblib
from utils import *
from datetime import date
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
import yaml

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


# 3. Save model
def save_model(model, base_path="artifacts/model"):
    try:
        today = date.today().isoformat()
        output_dir = os.path.join(base_path, today)
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "lasso_model.pkl")
        joblib.dump(model, model_path)

        logging.info(f"Model saved to {model_path}")
        return model_path
    except Exception as e:
        raise CustomException(e, sys)




if __name__ == "__main__":
    train_path = r"D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\artifacts\transformed_data\train\train.csv"
    df = read_training_data(train_path)
    
    model = train_model(df, model_alpha)
    save_model(model)