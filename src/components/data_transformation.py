import pandas as pd
import numpy as np
import os
import ast
import logging
import sys

from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

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


# 2. Data Cleaning
def data_cleaning(df):
    try:
        df['registrationDate'] = pd.to_datetime(df['registrationDate'], unit='ms').dt.date
        df['odometer'] = df['odometer'].apply(lambda x: ast.literal_eval(x)['value'])
        df['emiDetails'] = df['emiDetails'].apply(ast.literal_eval)
        emi_df = pd.json_normalize(df['emiDetails'])
        df = df.drop(columns='emiDetails').join(emi_df)
        df['transmissionType'] = df['transmissionType'].apply(lambda x: ast.literal_eval(x)['value'])

        rto_path = r'D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\rto_codes.csv'
        rto_df = pd.read_csv(rto_path)
        rto_df['rto_code'] = rto_df['state_name'] + rto_df['city_code'].astype(str).str.zfill(2)
        rto_to_city = dict(zip(rto_df['rto_code'], rto_df['city_name']))
        df['registeredCity'] = df['cityRto'].map(rto_to_city)

        df['features'] = df['features'].apply(ast.literal_eval)
        mlb = MultiLabelBinarizer()
        features_encoded = pd.DataFrame(mlb.fit_transform(df['features']),
                                        columns=mlb.classes_, index=df.index)
        df = df.join(features_encoded)
        df['featureCount'] = df['features'].apply(len)

        df['avgEmi'] = df[['emiStartingValue', 'emiEndingValue']].mean(axis=1)

        df.drop(columns=[
            'appointmentId','carName','modelGroup','features','displayText',
            'notAvailableText','tenure','registrationDate','registeredCity'
        ], errors='ignore', inplace=True)

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


def perform_transformation(train_df, test_df):
    try:
        target_col = 'listingPrice'
        
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Get feature names safely
        output_feature_names = preprocessor.get_feature_names_out()
        final_columns = output_feature_names.tolist()

        transformed_train_df = pd.DataFrame(X_train_transformed, columns=final_columns, index=X_train.index)
        transformed_test_df = pd.DataFrame(X_test_transformed, columns=final_columns, index=X_test.index)

        # Add target back to final dataframe
        transformed_train_df[target_col] = y_train.values
        transformed_test_df[target_col] = y_test.values

        return transformed_train_df, transformed_test_df

    except Exception as e:
        raise CustomException(e, sys)



# 4. Save transformed data
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


# 5. Run pipeline
if __name__ == "__main__":

    # Step 1: Read raw train and test data
    train_path = r"D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\artifacts\data_ingestion\train\train.csv"
    test_path = r"D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\artifacts\data_ingestion\test\test.csv"

    train_df, test_df = read_data('path', train_path, test_path)
    logging.info("Train and test data read successfully.")

    # Step 2: Clean data
    train_df = data_cleaning(train_df)
    test_df = data_cleaning(test_df)
    logging.info("Data cleaning completed.")

    # Step 3: Transform data (OHE, scaling)
    train_df, test_df = perform_transformation(train_df, test_df)
    logging.info("Data transformation (encoding/scaling) completed.")

    # Step 4: Save transformed data
    train_out_dir = r"D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\artifacts\transformed_data\train"
    test_out_dir = r"D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\artifacts\transformed_data\test"

    save_transformed_data(train_df, test_df, train_out_dir, test_out_dir)
    logging.info("Transformed data saved successfully.")