from utils import convert_region, split_date, Clipping, boxplots
import numpy as np
import os
import pandas as pd
from IPython.display import display

csv_path = os.path.join("vietnam-weather-data", "weather.csv")

reworked_df = pd.read_csv(csv_path)
reworked_df = reworked_df[reworked_df["province"] != "Hanoi"]
reworked_df = reworked_df.rename(columns={'max': 'max_temp', 'min': 'min_temp', 'wind': 'wind_speed', 'wind_d': 'wind_direct'})


#Adding new clounms and split date column
df_copy = reworked_df.copy()
df_copy['range_temp'] = df_copy['max_temp'] - df_copy['min_temp']
df_copy['region'] = df_copy['province']
df_copy = convert_region(df_copy, 'region')
df_copy = split_date(df_copy, 'date')
df_copy['date'] = pd.to_datetime(df_copy[['year', 'month', 'day']])


# Sort the data so lags make sense (grouped by province, by time)
df_copy = df_copy.sort_values(['province', 'date'])

# Create lag features per province
df_copy['rain_1d_ago'] = df_copy.groupby('province')['rain'].shift(1)
df_copy['rain_3d_avg'] = df_copy.groupby('province')['rain'].shift(1).rolling(3).mean().reset_index(0, drop=True)
df_copy['rain_7d_sum'] = df_copy.groupby('province')['rain'].shift(1).rolling(7).sum().reset_index(0, drop=True)
df_copy[['rain_1d_ago', 'rain_3d_avg', 'rain_7d_sum']] = df_copy[
    ['rain_1d_ago', 'rain_3d_avg', 'rain_7d_sum']
].fillna(0)


# Rain trend: 3-day moving average (shifted to avoid leakage)
df_copy['rain_trend_3d'] = df_copy['rain'].rolling(window=3).mean().shift(1).fillna(0)

# Rain intensity: today's rain compared to past 7-day average (shifted)
df_copy['rain_intensity'] = (df_copy['rain'] / (df_copy['rain'].rolling(7).mean().shift(1) + 1e-5)).fillna(0)

# Seasonal indicators
df_copy['is_rainy_season'] = df_copy['month'].apply(lambda x: 1 if 5 <= x <= 11 else 0)
df_copy['is_dry_season'] = df_copy['month'].apply(lambda x: 1 if x in [12, 1, 2, 3, 4] else 0)

# Change Wind Direction to angle in degrees
direction_to_angle = {
    'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
    'S': 180, 'SW': 225, 'W': 270, 'NW': 315
}

# change into degree (radian)
df_copy['wind_angle'] = df_copy['wind_direct'].map(direction_to_angle).fillna(0)
theta = np.radians(df_copy['wind_angle'])
 
#  new features for wind speed in x and y components
df_copy['wind_x'] = df_copy['wind_speed'] * np.cos(theta)
df_copy['wind_y'] = df_copy['wind_speed'] * np.sin(theta)

# avg rain by province
df_copy['avg_rain_province'] = df_copy.groupby('province')['rain'].transform('mean')
df_copy['avg_rain_province'] = df_copy['avg_rain_province'].fillna(0)
# avg rain by region
df_copy['avg_rain_region'] = df_copy.groupby('region')['rain'].transform('mean')
df_copy['avg_rain_region'] = df_copy['avg_rain_region'].fillna(0)

# Rain lag features
df_copy['rain_2d_ago'] = df_copy.groupby('province')['rain'].shift(2)
df_copy['rain_2d_ago'] = df_copy['rain_2d_ago'].fillna(0)

# humidity and temperature interaction
df_copy['temp_humidity'] = df_copy['range_temp'] * df_copy['humidi']
# wind speed from x and y components
df_copy['wind_speed'] = np.sqrt(df_copy['wind_x']**2 + df_copy['wind_y']**2)

# cloud and humidity interaction
df_copy['cloud_humid'] = df_copy['cloud'] * df_copy['humidi']
#cloud and temperature interaction
df_copy['cloud_temp_range'] = df_copy['cloud'] * df_copy['range_temp']

coord_path = os.path.join("vietnam-weather-data", "Final_Province_Coordinates.csv")
coord_df = pd.read_csv(coord_path)
# combined to add coordinates to the main DataFrame
df_copy = pd.merge(df_copy, coord_df, on='province', how='left')
df_copy =df_copy.drop(columns=['region_x'])
df_copy = df_copy.rename(columns={'region_y': 'region'})


#Check for missing data
print('Missing data in  DataFrame:\n',df_copy.isnull().sum())
#check for duplicaiton
print('Duplicate in DataFrame: ',df_copy.duplicated().sum())


boxplots(df_copy.drop(columns=['region', 'day', 'month', 'year','is_rainy_season','is_dry_season','avg_rain_region','avg_rain_province','rain_1d_ago','rain_3d_avg','rain_7d_sum', 'rain_intensity','temp_humidity','cloud_humid','rain_trend_3d','wind_angle','wind_x','wind_y','cloud_temp_range','rain_2d_ago','Latitude','Longitude'], axis=1))


# Handling outlier by clipping
df_clean = df_copy.copy()
df_clip = df_clean.drop(columns=[
    'region', 'day', 'month', 'year', 'is_rainy_season', 'is_dry_season',
    'avg_rain_region', 'avg_rain_province', 'rain_1d_ago', 'rain_3d_avg', 'rain_7d_sum',
    'rain_intensity', 'temp_humidity', 'cloud_humid', 'rain_trend_3d', 'wind_angle',
    'wind_x', 'wind_y', 'cloud_temp_range', 'rain_2d_ago', 'Latitude', 'Longitude'
], axis=1)
# Chỉ xử lý clipping cho các cột số
for i in df_clip.select_dtypes(include='number').columns:
    df_clip[i] = Clipping(df_clip[i])

df_clean[df_clip.columns] = df_clip


#box plot after
boxplots(df_clean.drop(columns=['region', 'day', 'month', 'year','is_rainy_season','is_dry_season','avg_rain_region','avg_rain_province','rain_1d_ago','rain_3d_avg','rain_7d_sum', 'rain_intensity','temp_humidity','cloud_humid','rain_trend_3d','wind_angle','wind_x','wind_y','cloud_temp_range','rain_2d_ago','Latitude','Longitude'], axis=1))


#(optional) Save the final DataFrame to a CSV file
df_copy.to_csv("vietnam-weather-data/added_feature_and_cleaned_weather.csv", index=False)

