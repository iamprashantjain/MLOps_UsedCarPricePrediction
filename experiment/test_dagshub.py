import dagshub
dagshub.init(repo_owner='iamprashantjain', repo_name='MLOps_UsedCarPricePrediction', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)