#!/usr/bin/env python
# coding: utf-8

# Import libraries
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import xgboost as xgb

import datetime

import pickle

import warnings
warnings.filterwarnings("ignore")


# Load data to dataframe
datafile = 'london_merged.csv'
df = pd.read_csv(datafile)


# Convert dtypes to save memory
df['weather_code'] = df['weather_code'].astype('uint8')
df['is_holiday'] = df['is_holiday'].astype('uint8')
df['is_weekend'] = df['is_weekend'].astype('uint8')
df['season'] = df['season'].astype('uint8')
df['t1'] = df['t1'].astype('float16')
df['t2'] = df['t2'].astype('float16')
df['hum'] = df['hum'].astype('float16')
df['wind_speed'] = df['wind_speed'].astype('float16')


# Sort data according to timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by=['timestamp'],ascending=True)
df.reset_index(drop=True,inplace=True)


# Splitting data as Full train (80%), Test (20%)
df_full_train, df_test = train_test_split(df,test_size=0.2,shuffle=False,random_state=1)


# Set target and delete it from dataframe
y_full_train = df_full_train['cnt']
y_test = df_test['cnt']

del df_full_train['cnt']
del df_test['cnt']


# Function to train the model and predict on validation data

def train_predict(df_full_train,df_test,y_full_train,model):
    X_full_train = df_full_train.values
    model.fit(X_full_train, y_full_train)

    X_test = df_test.values
    y_pred = model.predict(X_test)
    
    y_train_pred = model.predict(X_full_train)
    
    return y_pred, y_train_pred, model


# Function to evaluate various metrics/scores on predictions on validation and training

def evaluate_scores(y_test_eval, y_pred_eval, y_full_train_eval, y_pred_full_train_eval):
    scores = {}
    scores['val_r2'] = r2_score(y_test_eval, y_pred_eval)
    scores['val_mse'] = mean_squared_error(y_test_eval, y_pred_eval,squared=True)
    scores['val_rmse'] = mean_squared_error(y_test_eval, y_pred_eval,squared=False)
    scores['val_mae'] = mean_absolute_error(y_test_eval, y_pred_eval)
    
    scores['train_r2'] = r2_score(y_full_train_eval, y_pred_full_train_eval)
    scores['train_mse'] = mean_squared_error(y_full_train_eval, y_pred_full_train_eval,squared=True)
    scores['train_rmse'] = mean_squared_error(y_full_train_eval, y_pred_full_train_eval,squared=False)
    scores['train_mae'] = mean_absolute_error(y_full_train_eval, y_pred_full_train_eval)

    rnd_digits = 5 #round upto how many digits
    for metric, value in scores.items():
        scores[metric] = round(scores[metric],rnd_digits)
    
    return scores


# Function to perform pre processing on data before training
# Combining all the step by step processing done above into a function
# Function to now create different features from timestamp
def pre_process_new_ft(df_to_process):

    df_to_process['year'] = df_to_process['timestamp'].dt.year
    df_to_process['month'] = df_to_process['timestamp'].dt.month
    df_to_process['day'] = df_to_process['timestamp'].dt.day
    df_to_process['hour'] = df_to_process['timestamp'].dt.hour
    df_to_process['day-of-week'] = pd.to_datetime(df_to_process['timestamp']).dt.dayofweek.values
    df_to_process['week-of-year'] = pd.to_datetime(df_to_process['timestamp']).dt.isocalendar().week.values
    df_to_process['day-of-year'] = pd.to_datetime(df_to_process['timestamp']).dt.dayofyear


    df_to_process['year'] = df_to_process['year'].astype('uint16')
    df_to_process['month'] = df_to_process['month'].astype('uint8')
    df_to_process['day'] = df_to_process['day'].astype('uint8')
    df_to_process['hour'] = df_to_process['hour'].astype('uint8')
    df_to_process['day-of-week'] = df_to_process['day-of-week'].astype('uint8')
    df_to_process['week-of-year'] = df_to_process['week-of-year'].astype('uint8')
    df_to_process['day-of-year'] = df_to_process['day-of-year'].astype('uint16')

    return df_to_process


# Function to encode the time features using cyclical encoding using sine and cosine. Drop original time features
def pre_process_cyclic_encode(df_to_process,drop_org=True):
    # Create cyclical encoded features
    cyclical_features = ['year', 'month', 'day', 'hour', 'day-of-week', 'week-of-year', 'day-of-year']
    for col in cyclical_features:
        df_to_process[f"{col}_x_norm"] = 2 * math.pi * df_to_process[col] / df_to_process[col].max()
        df_to_process[f"{col}_cos_x"] = np.cos(df_to_process[f"{col}_x_norm"])
        df_to_process[f"{col}_sin_x"] = np.sin(df_to_process[f"{col}_x_norm"])
        del df_to_process[f"{col}_x_norm"]

    if drop_org:
        for col in cyclical_features:
            del df_to_process[col]
            
    return df_to_process


# Train final model

# Preprocess by creating new time related features
df_full_train = pre_process_new_ft(df_full_train)
df_test = pre_process_new_ft(df_test)

# Drop the timestamp feature, since we added more meaningful features and experiment with timestamp had not helped
del df_full_train['timestamp']
del df_test['timestamp']

new_features = list(df_full_train.columns)

#Define the hyper-parameters for the XGB model
eta=0.1
n_estimators=1000
max_depth=4
min_child_weight=5
colsample_bytree=0.8

model = xgb.XGBRegressor(random_state=42, n_jobs=-1, objective="reg:squarederror", booster='gbtree', learning_rate=eta, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight, colsample_bytree=colsample_bytree)

# Train and get predictions
y_pred, y_full_train_pred, model = train_predict(df_full_train[new_features],df_test[new_features],y_full_train,model)

# Score
scores = evaluate_scores(y_test, y_pred, y_full_train, y_full_train_pred)

print(scores)


# Save model to file
model_output_file = f'capstone-xgb_model.bin'

with open(model_output_file,'wb') as f_out:
    pickle.dump((model),f_out)

