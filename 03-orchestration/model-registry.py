#!/usr/bin/env python
# coding: utf-8



import pickle
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import mlflow
import time
import importlib.util
import numpy as np
from sklearn.metrics import mean_squared_error

spec = importlib.util.spec_from_file_location("duration_prediction", "duration-prediction.py")
duration_prediction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(duration_prediction)


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

runs = client.search_runs(
    experiment_ids='1',
    filter_string="metrics.rmse < 6",
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=5,
    order_by=["metrics.rmse ASC"]
)

runs = sorted(runs, key=lambda r: r.data.metrics['rmse'], reverse=True)

model_name="nyc-taxi-regressor"

for run in runs:
    print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['rmse']:.4f}")
    run_id=run.info.run_id
    model_uri = f"runs:/{run_id}/models_mlflow"
    mlflow.register_model(model_uri=model_uri, name=model_name)

model_version = 4
new_stage = "Production"
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=False
)

model_version = 5
new_stage = "Staging"
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=False
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

latest_versions = client.get_latest_versions(name=model_name)

for version in latest_versions:
    print(f"version: {version.version}, stage: {version.current_stage}")

df = duration_prediction.download_data(2024,3)

def test_model(name, stage, X_test, y_test):
    model = mlflow.pyfunc.load_model(f"models:/{name}/{stage}")
    y_pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))  

for stage in ["Staging", "Production"]:
    versions = client.get_latest_versions(name=model_name, stages=[stage])
    
    if not versions:
        print(f"No model version found in stage: {stage}")
        continue
    
    for v in versions:
        print(f"\n--- Testing version {v.version} (stage: {stage}, run_id: {v.run_id}) ---")

        client.download_artifacts(run_id=v.run_id, path='preprocessor', dst_path='.')
        with open("preprocessor/preprocessor_v2.b", "rb") as f_in:
            dv = pickle.load(f_in)
        X_test, y_test, _ = duration_prediction.engineer_features(df , dv)

        start = time.time()
        result = test_model(name=model_name, stage=stage, X_test=X_test, y_test=y_test)
        elapsed = time.time() - start
        print(f"Result: {result}, elapsed: {elapsed:.2f}s")


client.transition_model_version_stage(
    name=model_name,
    version=5,
    stage="Production",
    archive_existing_versions=True
)


