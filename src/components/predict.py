import pandas as pd
import joblib
import os
import logging
import sys
import ast
import numpy as np
from utils import *


# Paths to artifacts
PREPROCESSOR_PATH = "artifacts/preprocessor/preprocessor.pkl"
FEATURE_MASK_PATH = "artifacts/preprocessor/selected_features.pkl"
MODEL_PATH = "artifacts/model/model.pkl"


def apply_preprocessing_and_selection(df):
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        feature_mask = joblib.load(FEATURE_MASK_PATH)

        X_transformed = preprocessor.transform(df)
        X_selected = X_transformed[:, feature_mask]

        return X_selected
    except Exception as e:
        raise CustomException(e, sys)

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        raise CustomException(e, sys)

df = pd.read_csv(r"D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\artifacts\transformed_data\train\train.csv")

# Apply transformation and feature selection
X_final = apply_preprocessing_and_selection(df)


model = load_model()
predictions = model.predict(X_final)


# Attach predictions to original data
df["PredictedPrice"] = predictions
df["PredictedPrice"] = df["PredictedPrice"].round(0).astype(int)


import pdb;pdb.set_trace()