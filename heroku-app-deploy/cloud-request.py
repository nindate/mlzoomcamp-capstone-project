import pandas as pd
import requests

url = 'http://localhost:9696/predict'

sample_data_points = [
    {'timestamp': '2016-12-22 08:00:00', 't1': 5.0, 't2': 2.0, 'hum': 100.0, 'wind_speed': 13.0, 'weather_code': 4, 'is_holiday': 0, 'is_weekend': 0, 'season': 3},   # actual=2510
    {'timestamp': '2016-08-11 15:00:00', 't1': 22.5, 't2': 22.5, 'hum': 51.5, 'wind_speed': 22.0, 'weather_code': 2, 'is_holiday': 0, 'is_weekend': 0, 'season': 1},  # actual=1862
    {'timestamp': '2016-12-30 10:00:00', 't1': 4.0, 't2': 1.5, 'hum': 100.0, 'wind_speed': 10.0, 'weather_code': 4, 'is_holiday': 0, 'is_weekend': 0, 'season': 3},  # actual=601
    {'timestamp': '2016-12-07 06:00:00', 't1': 10.5, 't2': 10.0, 'hum': 94.0, 'wind_speed': 12.0, 'weather_code': 3, 'is_holiday': 0, 'is_weekend': 0, 'season': 3},  # actual=592
    {'timestamp': '2016-11-22 22:00:00', 't1': 8.5, 't2': 7.5, 'hum': 87.0, 'wind_speed': 8.0, 'weather_code': 7, 'is_holiday': 0, 'is_weekend': 0, 'season': 2},  # actual=571
    {'timestamp': '2016-12-25 23:00:00', 't1': 13.0, 't2': 13.0, 'hum': 79.5, 'wind_speed': 28.0, 'weather_code': 4, 'is_holiday': 0, 'is_weekend': 1, 'season': 3},  # actual=662
    {'timestamp': '2016-12-28 20:00:00', 't1': 3.5, 't2': 1.5, 'hum': 96.5, 'wind_speed': 7.0, 'weather_code': 1, 'is_holiday': 0, 'is_weekend': 0, 'season': 3},  # acutal=414
    {'timestamp': '2016-12-26 08:00:00', 't1': 8.0, 't2': 5.0, 'hum': 82.0, 'wind_speed': 22.0, 'weather_code': 1, 'is_holiday': 1, 'is_weekend': 0, 'season': 3},  # actual=263
]

details = sample_data_points[3]
prediction = requests.post(url,json=details).json()

print(f"predicted bike shares: {prediction}")
