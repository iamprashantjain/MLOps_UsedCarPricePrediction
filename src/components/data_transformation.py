import pandas as pd
import numpy as np
import os
import ast
import logging
import sys
from sklearn.preprocessing import MultiLabelBinarizer
from itertools import chain
from utils import CustomException

# 1. Read data
def read_data(source_type, train_path, test_path):
    if source_type == 'path':
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Data read successfully from {train_path} and {test_path}")
            return train_df, test_df
        except Exception as e:
            logging.error("Failed to read data", exc_info=True)
            raise CustomException(e, sys)
    else:
        logging.info("Other source type not configured yet")


# 2. Transformation
def perform_transformation(df):
    try:
        # Convert timestamps
        df['registrationDate'] = pd.to_datetime(df['registrationDate'], unit='ms').dt.date

        # Extract nested values
        df['odometer'] = df['odometer'].apply(lambda x: ast.literal_eval(x)['value'])
        df['emiDetails'] = df['emiDetails'].apply(ast.literal_eval)
        emi_df = pd.json_normalize(df['emiDetails'])
        df = df.drop(columns='emiDetails').join(emi_df)
        df['transmissionType'] = df['transmissionType'].apply(lambda x: ast.literal_eval(x)['value'])

        # Load and map RTO data
        rto_path = r'D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\rto_codes.csv'
        rto_df = pd.read_csv(rto_path)
        rto_df['rto_code'] = rto_df['state_name'] + rto_df['city_code'].astype(str).str.zfill(2)
        rto_to_city = dict(zip(rto_df['rto_code'], rto_df['city_name']))
        df['registeredCity'] = df['cityRto'].map(rto_to_city)

        # Feature encoding
        df['features'] = df['features'].apply(ast.literal_eval)
        mlb = MultiLabelBinarizer()
        features_encoded = pd.DataFrame(mlb.fit_transform(df['features']),
                                        columns=mlb.classes_, index=df.index)
        df = df.join(features_encoded)
        df['featureCount'] = df['features'].apply(len)

        # EMI and ROI
        df['avgEmi'] = df[['emiStartingValue', 'emiEndingValue']].mean(axis=1)

        # Drop unnecessary columns
        df.drop(columns=[
            'appointmentId','carName','modelGroup','features','displayText',
            'notAvailableText','tenure','registrationDate','registeredCity'
        ], errors='ignore', inplace=True)

        # Clean categorical and boolean features
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['ownership'] = df['ownership'].map({'Owned': 1, 'Leased': 0})

        bool_columns = [
            '360DegreeCamera', 'AlloyWheels', 'AppleCarplayAndroidAuto', 'Bluetooth',
            'CruiseControl', 'GpsNavigation', 'InfotainmentSystem', 'LeatherSeats',
            'ParkingAssist', 'PushButtonStart', 'RearAc', 'SpecialRegNo', 'Sunroof/Moonroof',
            'TopModel', 'Tpms', 'VentilatedSeats'
        ]

        df[bool_columns] = df[bool_columns].applymap(lambda x: 1 if x else 0)

        return df
    except Exception as e:
        logging.error("Error during data transformation", exc_info=True)
        raise CustomException(e, sys)


# 3. Save transformed data
def save_transformed_data(train_df, test_df, train_data_dir, test_data_dir):
    try:
        os.makedirs(train_data_dir, exist_ok=True)
        os.makedirs(test_data_dir, exist_ok=True)

        train_path = os.path.join(train_data_dir, "train.csv")
        test_path = os.path.join(test_data_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logging.info(f"Transformed data saved to {train_path} and {test_path}")
    except Exception as e:
        logging.error("Failed to save transformed data", exc_info=True)
        raise CustomException(e, sys)
    
    
    
if __name__ == "__main__":
    # Step 1: Read raw train and test data
    train_path = r"D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\artifacts\data_ingestion\train\train.csv"
    test_path = r"D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\artifacts\data_ingestion\test\test.csv"

    train_df, test_df = read_data('path', train_path, test_path)
    print("Train and test data read successfully.")

    # Step 2: Perform transformation
    train_df = perform_transformation(train_df)
    test_df = perform_transformation(test_df)
    print("Data transformation completed.")

    # Step 3: Save transformed data
    train_out_dir = r"D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\artifacts\transformed_data\train"
    test_out_dir = r"D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\artifacts\transformed_data\test"

    save_transformed_data(train_df, test_df, train_out_dir, test_out_dir)
    print("Transformed data saved successfully.")