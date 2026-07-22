#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow 

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-taxi-experiment")




def download_data(year, month):
    filename = f"data/green_tripdata_{year}-{month:02d}.parquet"
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)

        df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
        df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df['duration_t'] = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration_t >= 0) & (df.duration_t <= 60)]
    
    df = df[df.trip_type.notna()]
    
    df['lpep_pickup_hour']=df['lpep_pickup_datetime'].dt.hour
    
    df['lpep_pickup_dayofweek']=df['lpep_pickup_datetime'].dt.dayofweek
    
    df['lpep_pickup_weekend']=df['lpep_pickup_dayofweek'].apply(lambda x: 'weekend' if x >= 5 else 'weekday')
    
    df['lpep_pickup_weekend_hour'] = df['lpep_pickup_weekend'].astype(str) + '_' + df['lpep_pickup_hour'].astype(str)
    
    df['PU_DO_LocationID'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)

    categorical = ['lpep_pickup_hour','PU_DO_LocationID']
    df[categorical] = df[categorical].astype(str)
    
    df['congestion_surcharge'].fillna(0, inplace=True)
    df['improvement_surcharge'].fillna(0, inplace=True)
    
    return df




def engineer_features(df, dv):
    categorical = ['lpep_pickup_weekend_hour', 'trip_type', 'congestion_surcharge', 'improvement_surcharge', 'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
        with open('models/preprocessor_v2.b', 'wb') as f_out:
            pickle.dump((dv), f_out)
    else:
        X = dv.transform(dicts)
    
    target = 'duration_t'
    y = df[target].values
    
    return X, y, dv



def objective(params, train, valid, y_val):
    with mlflow.start_run():
        mlflow.set_tag("developer", "sameh")
        mlflow.log_param("train-data-path", "data/green_tripdata_2024-01.parquet")
        mlflow.log_param("val-data-path", "data/green_tripdata_2024-02.parquet")
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_params(params)
        mlflow.log_artifact(local_path='models/preprocessor_v2.b', artifact_path='preprocessor')
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=5
        )
        y_pred = booster.predict(valid)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

    return {'loss': rmse, 'status': STATUS_OK}



def hypertuning(X_train, y_train, X_val, y_val):
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 25, 1)),
        'learning_rate': hp.loguniform('learning_rate', -4, -2),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -2, 2),
        #'objective': 'reg:linear',
        'seed': 42
    }

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    best_result = fmin(
        fn=lambda params: objective(params, train, valid, y_val),
        space=search_space,
        algo=tpe.suggest,
        max_evals=5,
        trials=Trials()
    )

    best_result['max_depth'] = int(best_result['max_depth'])
    return best_result



def train_model(best_params, X_train, y_train, X_val, y_val):
    with mlflow.start_run() as run:

        mlflow.set_tag("developer", "sameh")
        mlflow.log_param("train-data-path", "data/green_tripdata_2024-01.parquet")
        mlflow.log_param("val-data-path", "data/green_tripdata_2024-02.parquet")
        mlflow.log_param("model_type", "xgboost")


        
        mlflow.xgboost.autolog(disable=True)
        mlflow.log_params(best_params)

        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=5
        )
        y_pred = booster.predict(valid)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)
        

        mlflow.log_artifact(local_path='models/preprocessor_v2.b', artifact_path='preprocessor')
        
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


def main(year, month):
    df_train = download_data(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = download_data(year=next_year, month=next_month)


    X_train, y_train, dv=engineer_features(df_train, None)
    X_val, y_val, _ =engineer_features(df_val, dv)

    best_params = hypertuning(X_train, y_train, X_val, y_val)
    print("Best hyperparameters:", best_params)
    

    run_id = train_model(best_params, X_train, y_train, X_val, y_val)
    
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = main(year=args.year, month=args.month)

    with open("run_id.txt", "w") as f:
        f.write(run_id)