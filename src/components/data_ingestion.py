import sys
import os
import pandas as pd
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from utils import *


# Read data
def read_data(source_type, path):
	if source_type == 'path':
		try:
			df = pd.read_csv(path)
			logging.info(f"Data read successfully from {path}")
			return df
		except Exception as e:
			logging.error("Failed to read data", exc_info=True)
			raise CustomException(e, sys)
	else:
		logging.info("Other source type not configured yet")


# Basic transformations
def basic_adjustments(df):
	try:
		#taking only 1000 rows for faster processing
		df = df.sample(1000)
	 
		#dropping useless columns
		df.drop(columns=['maskedRegNum', 'cityId', 'oemServiceHistoryAvailable'], inplace=True)
		logging.info("Basic adjustments done")
		return df
	
	except Exception as e:
		logging.error("Failed to perform basic adjustments", exc_info=True)
		raise CustomException(e, sys)


# Train-test split
def split_data(df, test_size=0.2, random_state=42):
	try:
		train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
		logging.info(f"Train-test split done with test size {test_size}")
		return train_df, test_df
	except Exception as e:
		logging.error("Failed during train-test split", exc_info=True)
		raise CustomException(e, sys)


# Orchestrator function
def data_ingestion_pipeline(source_type, path, output_dir="artifacts/data_ingestion"):
	try:
		# Create necessary directories
		raw_data_dir = os.path.join(output_dir, "raw")
		train_data_dir = os.path.join(output_dir, "train")
		test_data_dir = os.path.join(output_dir, "test")
		
		os.makedirs(raw_data_dir, exist_ok=True)
		os.makedirs(train_data_dir, exist_ok=True)
		os.makedirs(test_data_dir, exist_ok=True)

		# Read raw data
		df = read_data(source_type, path)
		
		# Save raw data to the raw directory
		raw_data_path = os.path.join(raw_data_dir, os.path.basename(path))
		df.to_csv(raw_data_path, index=False)
		logging.info(f"Raw data saved at {raw_data_path}")

		# Perform basic adjustments
		df = basic_adjustments(df)

		# Split data
		train_df, test_df = split_data(df)

		# Save train and test data
		train_path = os.path.join(train_data_dir, "train.csv")
		test_path = os.path.join(test_data_dir, "test.csv")
		
		train_df.to_csv(train_path, index=False)
		test_df.to_csv(test_path, index=False)

		logging.info(f"Train and test data saved at {train_data_dir} and {test_data_dir}")
		
		return raw_data_path, train_path, test_path

	except Exception as e:
		logging.error("Data ingestion pipeline failed", exc_info=True)
		raise CustomException(e, sys)


# if __name__ == "__main__":
# 	raw_path, train_path, test_path = data_ingestion_pipeline('path', r'D:\campusx_dsmp2\9. MLOps revisited\cars24_mlops_project\artifacts\scraped_data\cars24_scraped_data.csv')
# 	print(f"Data saved: {raw_path}, {train_path}, {test_path}")