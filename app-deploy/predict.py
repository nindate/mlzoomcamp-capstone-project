#!/usr/bin/env python
# coding: utf-8

#Import libraries
import pandas as pd
import xgboost as xgb
import pickle
import math
import numpy as np
from flask import Flask, request, jsonify

#Parameters
model_file = 'capstone-xgb_model.bin'

# Load the model
print("Loading model from file on disk")
with open(model_file,'rb') as f_in:
    model = pickle.load(f_in)

    
# Function to now create different features from timestamp
def pre_process_new_ft(df_to_process):

    # Convert to appropriate dtypes
    df_to_process['weather_code'] = df_to_process['weather_code'].astype('uint8')
    df_to_process['is_holiday'] = df_to_process['is_holiday'].astype('uint8')
    df_to_process['is_weekend'] = df_to_process['is_weekend'].astype('uint8')
    df_to_process['season'] = df_to_process['season'].astype('uint8')
    df_to_process['t1'] = df_to_process['t1'].astype('float16')
    df_to_process['t2'] = df_to_process['t2'].astype('float16')
    df_to_process['hum'] = df_to_process['hum'].astype('float16')
    df_to_process['wind_speed'] = df_to_process['wind_speed'].astype('float16')

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


    # Create cyclical encoded features
    cyclical_features = ['year', 'month', 'day', 'hour', 'day-of-week', 'week-of-year', 'day-of-year']
    for col in cyclical_features:
        df_to_process[f"{col}_x_norm"] = 2 * math.pi * df_to_process[col] / df_to_process[col].max()
        df_to_process[f"{col}_cos_x"] = np.cos(df_to_process[f"{col}_x_norm"])
        df_to_process[f"{col}_sin_x"] = np.sin(df_to_process[f"{col}_x_norm"])
        del df_to_process[f"{col}_x_norm"]

    return df_to_process


# Predict Bikes Share count
def predict_bike_share_count(details):
    input_features = ['timestamp', 't1', 't2', 'hum', 'wind_speed', 'weather_code', 'is_holiday', 'is_weekend', 'season']
    df_input = pd.DataFrame(columns=input_features)
    df_input = df_input.append(details,ignore_index=True)
    df_input['timestamp'] = pd.to_datetime(df_input['timestamp'])
    df_input = pre_process_new_ft(df_input)
    del df_input['timestamp']
    prediction = model.predict(df_input)
    return prediction
    

app = Flask('bikeshares')

@app.route('/predict', methods=['POST'])
def predict():
    details = request.get_json()
    bike_share_count = predict_bike_share_count(details)

    #Need to convert numpy float to python float, hence use the float() as below
    result = {
            'predicted_bike_share_count': int(bike_share_count)
            }

    return jsonify(result)

if __name__ == "__main__":
   app.run(debug=True, host='0.0.0.0', port=9696)
   

