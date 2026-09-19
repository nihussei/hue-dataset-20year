#!/usr/bin/env python
# coding: utf-8

# ### AI-Based Forecasting Model Implementation

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, InputLayer, LayerNormalization
import io
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import MeanSquaredError, Huber
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os
from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from dateutil.relativedelta import relativedelta
from sklearn.metrics import mean_squared_error as mse
from dateutil.easter import easter
import pandas as pd
from pandas.tseries.offsets import Week, Day

# Path to text file
file_path = 'dataverse_files/All_Residential.txt'

# Read the file and extract the table part
with open(file_path, 'r') as file:
    lines = file.readlines()

# Find the start of the table (assuming it's the line after the header delimiter line)
start_index = None
for i, line in enumerate(lines):
    if line.startswith('House '):  # Assuming the table starts with "House " as the header
        start_index = i
        break

# Extract the table lines
table_lines = lines[start_index:start_index + 30]  # Adjust to the number of rows in your table

# Join the table lines into a single string for pandas to read
table_str = ''.join(table_lines)

# Use io.StringIO to treat the string as a file object for pandas
table_io = io.StringIO(table_str)

# Read the table into a pandas DataFrame using read_fwf for fixed-width formatted lines
df = pd.read_fwf(table_io)
metadata = df.drop([0]) 

# Reset index
metadata.reset_index(drop=True, inplace=True)

# Convert House column to int64
metadata['House'] = metadata['House'].astype(int)

# clean up metadata
metadata['HouseType_missing'] = metadata['HouseType'].isna().astype(int)
metadata['HouseType'] = metadata['HouseType'].fillna(metadata['HouseType'].mode()[0]) # fill in nans with most common house type
metadata['Facing_missing'] = metadata['Facing'].isna().astype(int)
metadata['Facing'] = metadata['Facing'].fillna(metadata['Facing'].mode()[0]) # fill in nans with most common facing direction
metadata['Region_missing'] = metadata['Region'].isna().astype(int)
metadata['Region'] = metadata['Region'].fillna(metadata['Region'].mode()[0]) # fill in nans with most common region
metadata['HVAC_missing'] = metadata['HVAC'].isna().astype(int)
metadata['HVAC'] = metadata['HVAC'].fillna(metadata['HVAC'].mode()[0]) # fill in nans with most common HVAC

# Load holiday data
holidays = pd.read_csv('dataverse_files/Holidays.csv')

# Load YVR weather data
weather_yvr = pd.read_csv('dataverse_files/Weather_YVR.csv')

# clean up YVR data
# fill in missing hours with nans
weather_yvr['datetime'] = pd.to_datetime(weather_yvr['date'])+pd.to_timedelta(weather_yvr['hour']-1, unit='h')
weather_yvr = weather_yvr.set_index('datetime')
weather_yvr = weather_yvr.reindex(pd.date_range(weather_yvr.index.min(), weather_yvr.index.max(), freq='h'))
weather_yvr['date'] = weather_yvr.index.strftime("%Y-%m-%d")
weather_yvr['hour'] = weather_yvr.index.hour+1
weather_yvr = weather_yvr.reset_index(drop=True)
# fill in the nans
weather_yvr['temp_missing'] = weather_yvr['temperature'].isna().astype(int)
weather_yvr['humid_missing'] = weather_yvr['humidity'].isna().astype(int)
weather_yvr['press_missing'] = weather_yvr['pressure'].isna().astype(int)
column_list = ['temperature','humidity','pressure']
for column in column_list:
    nan_mask = weather_yvr[column].isna()
    nan_indices = list(np.where(nan_mask==True)[0]) 
    for index in nan_indices:
        interpolated = np.nan
        try: # interpolate from the two hours on either side
            interpolated = (weather_yvr.iloc[index+1][column]+weather_yvr.iloc[index-1][column])/2
        except Exception:
            pass
        if not np.isnan(interpolated):
            weather_yvr.loc[index,column] = round(interpolated,2)
            continue
        try: # interpolate from the two days on either side
            interpolated = (weather_yvr.iloc[index+24][column] + weather_yvr.iloc[index-24][column])/2
        except Exception:
            pass
        if not np.isnan(interpolated):
            weather_yvr.loc[index,column] = round(interpolated,2)
            continue
        try: # grab from the day before
            interpolated = weather_yvr.iloc[index-24][column]
        except Exception:
            pass
        if not np.isnan(interpolated):
            weather_yvr.loc[index,column] = interpolated
            continue
        try: # grab from the day after
            interpolated = weather_yvr.iloc[index+24][column]
        except Exception:
            pass
        if not np.isnan(interpolated):
            weather_yvr.loc[index,column] = interpolated
weather_yvr['weather_missing'] = weather_yvr['weather'].isna().astype(int)
weather_yvr['weather'] = weather_yvr['weather'].ffill() # forward fill

weather_yvr['Region']='YVR'

# Load WYJ weather data
weather_wyj = pd.read_csv('dataverse_files/Weather_WYJ.csv')

# clean up WYJ data
# fill in missing hours with nans
weather_wyj['datetime'] = pd.to_datetime(weather_wyj['date'])+pd.to_timedelta(weather_wyj['hour']-1, unit='h')
weather_wyj = weather_wyj.set_index('datetime')
weather_wyj = weather_wyj.reindex(pd.date_range(weather_wyj.index.min(), weather_wyj.index.max(), freq='h'))
weather_wyj['date'] = weather_wyj.index.strftime("%Y-%m-%d")
weather_wyj['hour'] = weather_wyj.index.hour+1
weather_wyj = weather_wyj.reset_index(drop=True)
# fill in the nans
weather_wyj['temp_missing'] = weather_wyj['temperature'].isna().astype(int)
weather_wyj['humid_missing'] = weather_wyj['humidity'].isna().astype(int)
weather_wyj['press_missing'] = weather_wyj['pressure'].isna().astype(int)
column_list = ['temperature','humidity','pressure']
for column in column_list:
    nan_mask = weather_wyj[column].isna()
    nan_indices = list(np.where(nan_mask==True)[0]) 
    for index in nan_indices:
        interpolated = np.nan
        try: # interpolate from the two hours on either side
            interpolated = (weather_wyj.iloc[index+1][column]+weather_wyj.iloc[index-1][column])/2
        except Exception:
            pass
        if not np.isnan(interpolated):
            weather_wyj.loc[index,column] = round(interpolated,2)
            continue
        try: # interpolate from the two days on either side
            interpolated = (weather_wyj.iloc[index+24][column] + weather_wyj.iloc[index-24][column])/2
        except Exception:
            pass
        if not np.isnan(interpolated):
            weather_wyj.loc[index,column] = round(interpolated,2)
            continue
        try: # grab from the day before
            interpolated = weather_wyj.iloc[index-24][column]
        except Exception:
            pass
        if not np.isnan(interpolated):
            weather_wyj.loc[index,column] = interpolated
            continue
        try: # grab from the day after
            interpolated = weather_wyj.iloc[index+24][column]
        except Exception:
            pass
        if not np.isnan(interpolated):
            weather_wyj.loc[index,column] = interpolated
weather_wyj['weather_missing'] = weather_wyj['weather'].isna().astype(int)
# use weather data from yvr region
lookup = weather_yvr.set_index(['date', 'hour'])['weather']
weather_wyj['weather'] = pd.Series(list(zip(weather_wyj['date'], weather_wyj['hour']))).map(lookup)

weather_wyj['Region']='WYJ'

# Combine weather data
weather_data = pd.concat([weather_wyj, weather_yvr])

# Load energy consumption data for each house
energy_data = []
for i in range(1, 29):
    file_path = f'dataverse_files/Residential_{i}.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # change hour to be 1 indexed instead of 0 indexed
    df['hour'] += 1
    
    # fill in missing hours in the prosumer energy data
    gap_mask = df['hour'].diff() == 2
    gap_indices = list(np.where(gap_mask==True)[0])
    
    # fix daylight savings shifts
    df['dst'] = 0
    gap_mask = df['hour'].diff() == 2
    gap_mask &= df['date'].dt.month == 3
    gap_indices = list(np.where(gap_mask==True)[0])   

    repeat_mask = df['hour'].diff() == 0
    repeat_mask &= df['date'].dt.month == 11
    repeat_indices = list(np.where(repeat_mask==True)[0])
    
    if (len(repeat_indices) == 1) and (len(gap_indices) == 0):
        gap_indices.insert(0,0)
        
    if (len(gap_indices) == 1) and (len(repeat_indices) == 0):
        repeat_indices.append(len(df))
    
    if repeat_indices[0] < gap_indices[0]:
        gap_indices.insert(0,0)
        
    if repeat_indices[-1] < gap_indices[-1]:
        repeat_indices.append(len(df))

    for start, stop in zip(gap_indices, repeat_indices):
        df.loc[start:stop-1, 'hour'] -= 1
        df.loc[start:stop-1, 'dst'] = 1
        
    # fix hours that are 0
    zero_mask = df['hour'] == 0
    df.loc[zero_mask, 'hour'] = 24
    df.loc[zero_mask, 'date'] -= pd.Timedelta(days=1) # need to adjust the date for those hours
    
    # get rid of duplicate rows
    df = df.drop_duplicates()
    
    # clean up data
    df['datetime'] = pd.to_datetime(df['date'])+pd.to_timedelta(df['hour']-1, unit='h')
    df = df.set_index('datetime')
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='h'))
    df['date'] = df.index.strftime("%Y-%m-%d")
    df['hour'] = df.index.hour+1
    df = df.reset_index(drop=True)
    
    # get rid of nans
    df['kWh_missing'] = df['energy_kWh'].isna().astype(int)
    nan_mask = df['energy_kWh'].isna()
    nan_indices = list(np.where(nan_mask==True)[0]) 
    
    for index in nan_indices:
        interpolated = np.nan
        
        try: # interpolate from the two hours on either side
            interpolated = (df.iloc[index+1]['energy_kWh'] + df.iloc[index-1]['energy_kWh'])/2
        except Exception:
            pass
        if not np.isnan(interpolated):
            df.loc[index,'energy_kWh'] = round(interpolated,3)
            continue
        
        try: # interpolate from the two days on either side
            interpolated = (df.iloc[index+24]['energy_kWh'] + df.iloc[index-24]['energy_kWh'])/2
        except Exception:
            pass
        if not np.isnan(interpolated):
            df.loc[index,'energy_kWh'] = round(interpolated,3)
            continue
        
        try: # grab from the day before
            interpolated = df.iloc[index-24]['energy_kWh']
        except Exception:
            pass
        if not np.isnan(interpolated):
            df.loc[index,'energy_kWh'] = interpolated
            continue
        
        try: # grab from the day after
            interpolated = df.iloc[index+24]['energy_kWh']
        except Exception:
            pass
        if not np.isnan(interpolated):
            df.loc[index,'energy_kWh'] = interpolated
            
    df['House'] = i
    df['date'] = pd.to_datetime(df['date'])    
    
    energy_data.append(df)

energy_data = pd.concat(energy_data, ignore_index=True)

# Convert House column to int64
energy_data['House'] = energy_data['House'].astype(int)

# set a minimum baseline energy usage where there are hours with 0
energy_data.loc[energy_data['energy_kWh']==0,'energy_kWh'] = energy_data[energy_data['energy_kWh'] > 0]['energy_kWh'].min()

# read solar data
solar_data = pd.read_csv('dataverse_files/Solar.csv')

# Convert 'date' columns to datetime
weather_data['date'] = pd.to_datetime(weather_data['date'])
holidays['date'] = pd.to_datetime(holidays['date'])
metadata['FirstReading'] = pd.to_datetime(metadata['FirstReading'])
metadata['LastReading'] = pd.to_datetime(metadata['LastReading'])
solar_data["date"] = pd.to_datetime(solar_data["date"].str.replace("^000", "2000", regex=True),format="%Y-%m-%d", errors='coerce')+pd.to_timedelta(solar_data['hour'], unit='h') # need to set a year so datetime can be formatted correctly

# make a plot
figsize = (10,6)
fig, ax = plt.subplots(figsize=figsize)
# create the data needed for the bar chart
width = weather_data.groupby('Region')['date'].max()-weather_data.groupby('Region')['date'].min()
left = weather_data.groupby('Region')['date'].min()-weather_data.groupby('Region')['date'].min().min()
# create the bar chart
plt.barh(y=pd.Series(weather_data['Region'].unique()), width=width.dt.days, left=left.dt.days)
plt.title('Weather Data Ranges', fontweight='bold')
# x ticks 
days_range = int((pd.Series(weather_data.groupby('Region')['date'].max().max()-weather_data.groupby('Region')['date'].min().min()).dt.days)[0])
while days_range%365 != 0:
    days_range+=1
xticks = np.arange(0, days_range+1, 365) # tick at every year mark
xticklabels = pd.date_range(start=weather_data.groupby('Region')['date'].min().min(),
                            end=weather_data.groupby('Region')['date'].min().min()+dt.timedelta(days=days_range)).strftime("%m/%d/%y")
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels[::365])
ax.set_xlabel("Date", fontweight='bold')
ax.set_ylabel('Region', fontweight='bold')
ax.grid(True)
fig.tight_layout()

# get rid of uncommon weather labels by combining with others
weather_mapping = {
    'Snow Pellets': 'Snow Showers',
    'Ice Pellets': 'Freezing Rain',
    'Funnel Cloud': 'Rain Showers',
    'Heavy Snow': 'Snow',
    'Moderate Drizzle': 'Drizzle',
    'Heavy Rain Showers': 'Rain Showers',
    'Snow Grains': 'Freezing Rain',
    'Smoke': 'Haze',
    'Freezing Drizzle': 'Freezing Rain',
    'Moderate Rain Showers': 'Rain Showers',
    'Moderate Snow': 'Snow',
    'Heavy Rain': 'Moderate Rain',
    'Thunderstorms': 'Rain Showers',
}

# Apply to your data
weather_data['weather'] = weather_data['weather'].replace(weather_mapping)

# dataframe for forecasting weather data
df_weather_forecast = weather_data.copy()

# Remove all rows where 'Region' is equal to 'WYJ', and then get rid of the 'Region' column
df_weather_forecast = df_weather_forecast[df_weather_forecast['Region'] != 'WYJ']
df_weather_forecast = df_weather_forecast.drop(columns=['Region'])

df_weather_forecast["day_of_year"] = df_weather_forecast['date'].dt.dayofyear

# cyclical encoding
df_weather_forecast["hour_sin"] = np.sin(2*np.pi*df_weather_forecast["hour"]/24)
df_weather_forecast["hour_cos"] = np.cos(2*np.pi*df_weather_forecast["hour"]/24)
df_weather_forecast["year"] = df_weather_forecast["date"].dt.year
df_weather_forecast["days_in_year"] = df_weather_forecast["year"].apply(
    lambda y: 366 if pd.Timestamp(f"{y}-12-31").dayofyear == 366 else 365)
df_weather_forecast["doy_sin"] = np.sin(2*np.pi*df_weather_forecast["day_of_year"]/df_weather_forecast["days_in_year"]) 
df_weather_forecast["doy_cos"] = np.cos(2*np.pi*df_weather_forecast["day_of_year"]/df_weather_forecast["days_in_year"])

# get rid of unnecessary features
df_weather_forecast = df_weather_forecast.drop(columns=['hour','days_in_year','day_of_year','date'])
df_weather_forecast = df_weather_forecast.drop(list(df_weather_forecast.filter(regex='_missing')), axis=1)

# train/test split
def time_split(df, train_frac=0.75, val_frac=0.875):
    train_parts = []
    val_parts = []
    test_parts = []
    
    
    train_idx = int(len(df)*train_frac)
    val_idx = int(len(df)*val_frac)
    train_parts.append(df.iloc[:train_idx])
    val_parts.append(df.iloc[train_idx:val_idx])
    test_parts.append(df.iloc[val_idx:])
    
    train = pd.concat(train_parts).reset_index(drop=True)
    val = pd.concat(val_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    
    return train, val, test

df_weather_forecast_train, df_weather_forecast_val, df_weather_forecast_test = time_split(df_weather_forecast)

# XGB weather classification
df_weather_xgb_train = df_weather_forecast_train.copy()
df_weather_xgb_val = df_weather_forecast_val.copy()
df_weather_xgb_test = df_weather_forecast_test.copy()

# prepare for XGBoost Classification
def prepare_for_XGBoostClassifier(df_in):
    df_new_features = pd.DataFrame(index=df_in.index)
    
    # lag terms
    for lag in range(1, 25):
        df_new_features[f'temp_lag_{lag}'] = df_in['temperature'].shift(lag)
        df_new_features[f'hum_lag_{lag}'] = df_in['humidity'].shift(lag)
        df_new_features[f'pres_lag_{lag}'] = df_in['pressure'].shift(lag)
        
    df_new_features['weath_lag_1'] = df_in['weather'].shift(1)
    
    # Interaction terms
    df_new_features['temp_humidity_interaction'] = df_new_features['temp_lag_1']*df_new_features['hum_lag_1']
    df_new_features['temp_pressure_interaction'] = df_new_features['temp_lag_1']*df_new_features['pres_lag_1']
    df_new_features['humidity_pressure_interaction'] = df_new_features['hum_lag_1']*df_new_features['pres_lag_1']

    # rolling means
    df_new_features['rolling_mean_temp'] = (df_in['temperature']
                                            .shift(1).rolling(24).mean()
                                            .reset_index(level=0, drop=True))
    df_new_features['rolling_mean_hum'] = (df_in['humidity']
                                           .shift(1).rolling(24).mean()
                                           .reset_index(level=0, drop=True))
    df_new_features['rolling_mean_pres'] = (df_in['pressure']
                                            .shift(1).rolling(24).mean()
                                            .reset_index(level=0, drop=True))
    
    # combine the new features into the initial dataframe
    df_out = pd.concat([df_in, df_new_features], axis=1)
    
    # drop nans
    df_out = df_out.dropna().reset_index(drop=True)
    
    return df_out

df_weather_xgb_train = prepare_for_XGBoostClassifier(df_weather_xgb_train)
df_weather_xgb_val = prepare_for_XGBoostClassifier(df_weather_xgb_val)
df_weather_xgb_test = prepare_for_XGBoostClassifier(df_weather_xgb_test)

# create X/y train/test dataframes
y_train_weath = df_weather_xgb_train[['weather']].copy()
X_train = df_weather_xgb_train.drop(columns=['temperature','humidity','pressure','weather'])

y_val_weath = df_weather_xgb_val[['weather']].copy()
X_val = df_weather_xgb_val.drop(columns=['temperature','humidity','pressure','weather'])

y_test_weath = df_weather_xgb_test[['weather']].copy()
X_test = df_weather_xgb_test.drop(columns=['temperature','humidity','pressure','weather'])

# Convert categorical features to one-hot encoded features
encoder_weather_xgb = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)

# Learn categories from the training data only
categorical_columns = ['weath_lag_1']
encoder_weather_xgb.fit(X_train[categorical_columns])

def apply_one_hot_encoder(df_in, encoder, columns):
    
    # create the numpy array representing the one-hot-encoded columns
    array_encoded = encoder.transform(df_in[columns])

    # turn it into a dataframe
    df_encoded = pd.DataFrame(data=array_encoded, columns=encoder.get_feature_names_out(columns), index=df_in.index)
    
    # combine the new features into the initial dataframe, while dropping the categorical columns
    df_out = pd.concat([df_in.drop(columns=columns), df_encoded], axis=1)

    return df_out

X_train_classifier = apply_one_hot_encoder(X_train, encoder_weather_xgb, categorical_columns)
X_val_classifier = apply_one_hot_encoder(X_val, encoder_weather_xgb, categorical_columns)
X_test_classifier = apply_one_hot_encoder(X_test, encoder_weather_xgb, categorical_columns)

# label encode the weather y data
label_encoder = LabelEncoder()
y_train_weath_encode = label_encoder.fit_transform(y_train_weath)
y_val_weath_encode = label_encoder.transform(y_val_weath)
y_test_weath_encode = label_encoder.transform(y_test_weath)


model_weather = XGBClassifier(
    n_estimators=2000,          # many trees
    learning_rate=0.01,        # slow learning
    max_depth=6,               # not too deep
    min_child_weight=7,        # prevents overfitting
    subsample=0.8,             # row sampling
    colsample_bytree=0.8,      # feature sampling
    reg_alpha=0.1,             # L1 regularization
    reg_lambda=0.2,            # L2 regularization
    gamma=0.05,                 # split penalty
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",        # fast + scalable
    max_bin=256,
    random_state=42,
    early_stopping_rounds=50,
    # max_delta_step=1,          # class handling  
    n_jobs=-1                  # efficiency
)

model_weather.fit(X_train_classifier, y_train_weath_encode, eval_set=[(X_val_classifier, y_val_weath_encode)], verbose=True)
y_predict_weath = model_weather.predict(X_test_classifier)

# Calculate metrics
weather_accuracy = accuracy_score(y_test_weath_encode, y_predict_weath)
print(f'Weather Accuracy: {weather_accuracy:0.4f}')
print('\nClassification Report:\n', classification_report(y_test_weath_encode, y_predict_weath))
print('\nConfusion Matrix:\n', confusion_matrix(y_test_weath_encode,y_predict_weath))


#### LSTM (temperature, pressure, humidity) ####

df_weather_lstm_train = df_weather_forecast_train.copy()
df_weather_lstm_val = df_weather_forecast_val.copy()
df_weather_lstm_test = df_weather_forecast_test.copy()

# prepare df for LSTM forecast
def prepare_for_LSTM(df_in):
    
    def add_rolling_statistics(df_in, cols, window=24, shift=0):
        df = df_in.copy()
        for col in cols:
            df[f'rolling_mean_{col}'] = df[col].shift(shift).rolling(window).mean().reset_index(level=0, drop=True)
            df[f'rolling_std_{col}'] = df[col].shift(shift).rolling(window).std().reset_index(level=0, drop=True)
        return df
    
    # Rolling statistics (should be shifted by one for regression, not shifted for LSTM). Should do this after train/val/test split
    cols_rolling_stats = ['temperature','humidity','pressure']
    df_out = add_rolling_statistics(df_in, cols_rolling_stats)

    # Drop rows with NaN values created by rolling and lag features
    df_out = df_out.dropna().reset_index(drop=True)

    # Interaction terms
    df_out['temp_humidity_interaction'] = df_out['temperature'] * df_out['humidity']
    df_out['temp_pressure_interaction'] = df_out['temperature'] * df_out['pressure']
    df_out['humidity_pressure_interaction'] = df_out['humidity'] * df_out['pressure']
    
    return df_out

df_weather_lstm_train = prepare_for_LSTM(df_weather_lstm_train)
df_weather_lstm_val = prepare_for_LSTM(df_weather_lstm_val)
df_weather_lstm_test = prepare_for_LSTM(df_weather_lstm_test)

# Convert categorical features to one-hot encoded features
encoder_weather_lstm = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)

# Learn categories from the training data only
categorical_columns = ['weather']
encoder_weather_lstm.fit(df_weather_lstm_train[categorical_columns])

df_weather_lstm_train = apply_one_hot_encoder(df_weather_lstm_train, encoder_weather_lstm, categorical_columns)
df_weather_lstm_val = apply_one_hot_encoder(df_weather_lstm_val, encoder_weather_lstm, categorical_columns)
df_weather_lstm_test = apply_one_hot_encoder(df_weather_lstm_test, encoder_weather_lstm, categorical_columns)

# scale X vars
continuous_vars_X = ['year','rolling_mean_temperature','rolling_std_temperature','rolling_mean_humidity','rolling_std_humidity','rolling_mean_pressure','rolling_std_pressure','temp_humidity_interaction','temp_pressure_interaction','humidity_pressure_interaction']
scaler_X_weath = StandardScaler()
df_weather_lstm_train_scaled = df_weather_lstm_train.copy()
df_weather_lstm_val_scaled = df_weather_lstm_val.copy()
df_weather_lstm_test_scaled = df_weather_lstm_test.copy()
df_weather_lstm_train_scaled[continuous_vars_X] = scaler_X_weath.fit_transform(df_weather_lstm_train_scaled[continuous_vars_X])
df_weather_lstm_val_scaled[continuous_vars_X] = scaler_X_weath.transform(df_weather_lstm_val_scaled[continuous_vars_X])
df_weather_lstm_test_scaled[continuous_vars_X] = scaler_X_weath.transform(df_weather_lstm_test_scaled[continuous_vars_X])

continuous_vars_y = ['temperature','humidity','pressure']
scaler_y_weath = StandardScaler()
df_weather_lstm_train_scaled[continuous_vars_y] = scaler_y_weath.fit_transform(df_weather_lstm_train_scaled[continuous_vars_y])
df_weather_lstm_val_scaled[continuous_vars_y] = scaler_y_weath.transform(df_weather_lstm_val_scaled[continuous_vars_y])
df_weather_lstm_test_scaled[continuous_vars_y] = scaler_y_weath.transform(df_weather_lstm_test_scaled[continuous_vars_y])

def df_to_X_y_weath(df, window_size=24):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
        y.append([df_as_np[i+window_size][0], df_as_np[i+window_size][1], df_as_np[i+window_size][2]]) # temperature, humidity, pressure
    return np.array(X), np.array(y)

WINDOW_SIZE = 24

X_train_weath_lstm, y_train_weath_lstm = df_to_X_y_weath(df_weather_lstm_train_scaled, WINDOW_SIZE)
X_val_weath_lstm, y_val_weath_lstm = df_to_X_y_weath(df_weather_lstm_val_scaled, WINDOW_SIZE)
X_test_weath_lstm, y_test_weath_lstm = df_to_X_y_weath(df_weather_lstm_test_scaled, WINDOW_SIZE)

layerLSTM = 64
dropout = 0
layerDense = 32
learning_rate = 0.0001

model_lstm_weather = Sequential()
model_lstm_weather.add(InputLayer((WINDOW_SIZE, 31)))
model_lstm_weather.add(LSTM(layerLSTM, dropout=dropout))
model_lstm_weather.add(LayerNormalization())
model_lstm_weather.add(Dense(layerDense, 'relu'))
model_lstm_weather.add(Dense(3, 'linear'))
model_lstm_weather.summary()

# Define checkpoints and early stopping
folder_name = f"weather_Wsize{WINDOW_SIZE}_LSTM{layerLSTM}_drpout{dropout}_Dense{layerDense}_LR{learning_rate}_norm/"
cp = ModelCheckpoint(folder_name, monitor='val_loss', save_best_only=True)
es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Compile the model
model_lstm_weather.compile(loss=Huber(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

# Train the model
history_lstm_weather = model_lstm_weather.fit(X_train_weath_lstm, y_train_weath_lstm, validation_data=(X_val_weath_lstm, y_val_weath_lstm), epochs=100, callbacks=[cp,es])

# Plot training & validation loss values
plt.figure()
plt.plot(history_lstm_weather.history['loss'], label='Training Loss')
plt.plot(history_lstm_weather.history['val_loss'], label='Validation Loss')
# plt.yscale('log')
plt.title('Model Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()
plt.tight_layout()

# load the best checkpoint
model_lstm_weather = load_model(folder_name)
test_predictions = model_lstm_weather.predict(X_test_weath_lstm)
test_predictions_unscaled = scaler_y_weath.inverse_transform(test_predictions)
y_test_lstm_unscaled = scaler_y_weath.inverse_transform(y_test_weath_lstm)

# temperature
plt.figure()
plt.plot(test_predictions_unscaled[0:100,0], label='predicted')
# plt.plot(df_weather_lstm_test['humidity'][24:124].reset_index(drop=True), label='actual')
plt.plot(y_test_lstm_unscaled[0:100,0], label='actual2')
plt.legend()
plt.grid()
plt.tight_layout()

# Calculate metrics
mae = mean_absolute_error(y_test_lstm_unscaled[:,0],test_predictions_unscaled[:,0])
rmse = np.sqrt(mean_squared_error(y_test_lstm_unscaled[:,0],test_predictions_unscaled[:,0]))
r2 = r2_score(y_test_lstm_unscaled[:,0],test_predictions_unscaled[:,0])

# Print the scores
print('\nTEMPERATURE')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²: {r2:.4f}')

# humidity
plt.figure()
plt.plot(test_predictions_unscaled[0:100,1], label='predicted')
# plt.plot(y_test_lstm_unscaled[0:100,1], label='actual')
plt.plot(df_weather_lstm_test['humidity'][24:124].reset_index(drop=True), label='actual')
plt.legend()
plt.grid()
plt.tight_layout()

# Calculate metrics
mae = mean_absolute_error(y_test_lstm_unscaled[:,1],test_predictions_unscaled[:,1])
rmse = np.sqrt(mean_squared_error(y_test_lstm_unscaled[:,1],test_predictions_unscaled[:,1]))
r2 = r2_score(y_test_lstm_unscaled[:,1],test_predictions_unscaled[:,1])

# Print the scores
print('\nHUMIDITY')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²: {r2:.4f}')

# pressure
plt.figure()
plt.plot(test_predictions_unscaled[0:100,2], label='predicted')
# plt.plot(y_test_lstm_unscaled[0:100,2], label='actual')
plt.plot(df_weather_lstm_test['pressure'][24:124].reset_index(drop=True), label='actual')
plt.legend()
plt.grid()
plt.tight_layout()

# Calculate metrics
mae = mean_absolute_error(y_test_lstm_unscaled[:,2],test_predictions_unscaled[:,2])
rmse = np.sqrt(mean_squared_error(y_test_lstm_unscaled[:,2],test_predictions_unscaled[:,2]))
r2 = r2_score(y_test_lstm_unscaled[:,2],test_predictions_unscaled[:,2])

# Print the scores
print('\nPRESSURE')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²: {r2:.4f}')

# extend weather data by predicting out to 20 years
df_weather_20yr = weather_data.copy()

# get rid of '_missing' columns
df_weather_20yr = df_weather_20yr.drop(list(df_weather_20yr.filter(regex='_missing')), axis=1)

# Remove all rows where 'Region' is not equal to 'YVR', and then get rid of the 'Region' column
df_weather_20yr = df_weather_20yr[df_weather_20yr['Region'] == 'YVR']
df_weather_20yr = df_weather_20yr.drop(columns=['Region'])

# add rows out to 20 years from prosumer 28 start date
df_weather_20yr['datetime'] = pd.to_datetime(df_weather_20yr['date'])+pd.to_timedelta(df_weather_20yr['hour']-1, unit='h')
df_weather_20yr = df_weather_20yr.set_index('datetime')
df_weather_20yr = df_weather_20yr.reindex(pd.date_range(df_weather_20yr.index.min(), pd.to_datetime('2019-01-01 00:00:00')+relativedelta(years=20), freq='h'))
df_weather_20yr['date'] = pd.to_datetime(df_weather_20yr.index.strftime("%Y-%m-%d"))
df_weather_20yr['hour'] = df_weather_20yr.index.hour+1
df_weather_20yr = df_weather_20yr.reset_index(drop=True)

# cyclical encoding
df_weather_20yr["day_of_year"] = df_weather_20yr['date'].dt.dayofyear
df_weather_20yr["hour_sin"] = np.sin(2*np.pi*df_weather_20yr["hour"]/24)
df_weather_20yr["hour_cos"] = np.cos(2*np.pi*df_weather_20yr["hour"]/24)
df_weather_20yr["year"] = df_weather_20yr["date"].dt.year
df_weather_20yr["days_in_year"] = df_weather_20yr["year"].apply(
    lambda y: 366 if pd.Timestamp(f"{y}-12-31").dayofyear == 366 else 365)
df_weather_20yr["doy_sin"] = np.sin(2*np.pi*df_weather_20yr["day_of_year"]/df_weather_20yr["days_in_year"]) 
df_weather_20yr["doy_cos"] = np.cos(2*np.pi*df_weather_20yr["day_of_year"]/df_weather_20yr["days_in_year"])

# get rid of unnecessary features
df_weather_20yr = df_weather_20yr.drop(columns=['hour','days_in_year','day_of_year','date'])

def df_to_final_X(df, window_size=24):
    return df.iloc[-window_size:].to_numpy()[None, :, :]

def make_last_xgb_features(df_in,idx):
    last = {}

    for lag in range(1,25):
        last[f'temp_lag_{lag}'] = df_in['temperature'].iat[idx-lag]
        last[f'hum_lag_{lag}'] = df_in['humidity'].iat[idx-lag]
        last[f'pres_lag_{lag}'] = df_in['pressure'].iat[idx-lag]
        
    last['weath_lag_1'] = df_in['weather'].iat[idx-1]
    
    last['temp_humidity_interaction'] = last['temp_lag_1']*last['hum_lag_1']
    last['temp_pressure_interaction'] = last['temp_lag_1']*last['pres_lag_1']
    last['humidity_pressure_interaction'] = last['hum_lag_1']*last['pres_lag_1']
    
    last['rolling_mean_temp'] = df_in['temperature'].iloc[idx-24:idx].mean()
    last['rolling_mean_hum'] = df_in['humidity'].iloc[idx-24:idx].mean()
    last['rolling_mean_pres'] = df_in['pressure'].iloc[idx-24:idx].mean()

    df_out = pd.DataFrame([last])
    
    return df_out

# load the best checkpoint
model_lstm_weather = load_model(folder_name)

nanidx = df_weather_20yr.isna().any(axis=1).idxmax()
lastidx = len(df_weather_20yr)
# lastidx = nanidx+5

while nanidx < lastidx:
    
    df_weather_last24 = df_weather_20yr.iloc[nanidx-48:nanidx].copy().reset_index(drop=True)
    
    # prepare for XGBoost Classification
    df_weather_xgb = prepare_for_XGBoostClassifier(df_weather_last24)
    X_24hr = df_weather_xgb.drop(columns=['temperature','humidity','pressure','weather'])
    categorical_columns = ['weath_lag_1']
    X_classifier_encoded = apply_one_hot_encoder(X_24hr, encoder_weather_xgb, categorical_columns)

    # prepare for LSTM forecast    
    df_weather_lstm_last24 = prepare_for_LSTM(df_weather_last24)
    categorical_columns = ['weather']
    df_weather_lstm_last24_encoded = apply_one_hot_encoder(df_weather_lstm_last24, encoder_weather_lstm, categorical_columns)

    # scale X vars
    df_weather_lstm_24hr_scaled = df_weather_lstm_last24_encoded.copy()
    df_weather_lstm_24hr_scaled[continuous_vars_X] = scaler_X_weath.transform(df_weather_lstm_24hr_scaled[continuous_vars_X])
    df_weather_lstm_24hr_scaled[continuous_vars_y] = scaler_y_weath.transform(df_weather_lstm_24hr_scaled[continuous_vars_y])
    final_X = df_to_final_X(df_weather_lstm_24hr_scaled, WINDOW_SIZE)
    
    next_prediction_lstm = model_lstm_weather.predict(final_X)
    next_prediction_lstm_unscaled = scaler_y_weath.inverse_transform(next_prediction_lstm)
    df_weather_20yr.loc[nanidx, continuous_vars_y] = next_prediction_lstm_unscaled[0]
    
    next_prediction_weather = model_weather.predict(X_classifier_encoded.iloc[[-1]])
    df_weather_20yr.loc[nanidx, 'weather'] = label_encoder.inverse_transform(next_prediction_weather).item()
    
    if nanidx % 5000 == 0:
        df_weather_20yr.to_csv('weather_20yr_cyclical.csv', index=False)
    
    nanidx += 1

df_weather_20yr.to_csv('weather_20yr_full.csv', index=False)

# clean up solar data
solar_data = solar_data.set_index('date')
solar_data = solar_data.reindex(pd.date_range(solar_data.index.min(), solar_data.index.max(), freq='h'))
solar_data['hour'] = solar_data.index.hour
solar_data['date'] = solar_data.index
solar_data = solar_data.reset_index(drop=True)
# get rid of nans
nan_mask = solar_data['ac_output'].isna()
nan_indices = list(np.where(nan_mask==True)[0]) 

for index in nan_indices:
    interpolated = np.nan
    
    try: # interpolate from the two days on either side
        interpolated = (solar_data.iloc[index+24]['ac_output'] + solar_data.iloc[index-24]['ac_output'])/2
    except Exception:
        pass
    if not np.isnan(interpolated):
        solar_data.loc[index,'ac_output'] = round(interpolated,3)
        continue
    
# make some plots
figsize = (10,6)

fig, ax = plt.subplots(figsize=figsize)

# create the data needed for the bar chart
width = metadata['LastReading']-metadata['FirstReading']
left = metadata['FirstReading']-metadata['FirstReading'].min()

# create the bar chart
plt.barh(y=metadata['House'], width=width.dt.days, left=left.dt.days)
plt.title('Prosumer Data Ranges', fontweight='bold')

# x ticks 
days_range = int((pd.Series(metadata['LastReading'].max()-metadata['FirstReading'].min()).dt.days)[0])
while days_range%365 != 0:
    days_range+=1
xticks = np.arange(0, days_range+1, 365) # tick at every year mark
xticklabels = pd.date_range(start=metadata['FirstReading'].min(), end=metadata['FirstReading'].min()+dt.timedelta(days=days_range)).strftime("%m/%d/%y")
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels[::365])
ax.set_xlabel("Date", fontweight='bold')

# y ticks
yticks = np.arange(0, max(metadata['House'])+1, 5) 
ax.set_yticks(yticks)
ax.set_ylabel('Prosumer', fontweight='bold')

ax.grid(True)
fig.tight_layout()

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
labels = [a for a in range(1,29)]
data = [energy_data.loc[energy_data['House'] == label, 'energy_kWh'].dropna() for label in labels]
vert = True
figsize = (10,6)
plt.figure(figsize=figsize)
plt.boxplot(data, 
                tick_labels=labels,
                vert=vert,
                patch_artist=True,
                boxprops=dict(linewidth=2.5), 
                whiskerprops=dict(linewidth=2.5), 
                medianprops=dict(linewidth=2.5,color=default_colors[1]), 
                capprops=dict(linewidth=2.5,color=default_colors[2]), 
                flierprops=dict(markerfacecolor=default_colors[3], marker='o'))
plt.ylabel("Hourly Power Usage", fontweight='bold')
plt.xlabel("Prosumers", fontweight='bold')
plt.title("Box Plots of Hourly Power Usage", fontweight='bold')
plt.grid()
plt.tight_layout()

date_list = []
energy_list = []
house_list = []
min_list = []
med_list = []
mean_list = []
max_list = []
years_list = list(pd.Series(metadata['LastReading']-metadata['FirstReading']).dt.days/365)

for house in labels:
    dates = energy_data[energy_data['House']==house].groupby('date').filter(lambda x: len(x)>20)['date'].value_counts().sort_index(axis = 0).index
    date_list.extend(dates)
    house_list.extend([house]*len(dates))
    energy = [sum(energy_data.loc[(energy_data['House'] == house) & (energy_data['date'] == date), 'energy_kWh'].dropna()) for date in dates]
    energy_list.extend(energy)
    energy = pd.Series(energy)
    min_list.append(energy.min())
    med_list.append(energy.median())
    mean_list.append(energy.mean())
    max_list.append(energy.max())

daily_energy = pd.DataFrame({'date': date_list, 'energy_kWh': energy_list, 'House': house_list})

data = [daily_energy.loc[daily_energy['House'] == label, 'energy_kWh'].dropna() for label in labels]
vert = True
figsize = (10,6)
plt.figure(figsize=figsize)
plt.boxplot(data, 
                tick_labels=labels,
                vert=vert,
                patch_artist=True,
                boxprops=dict(linewidth=2.5), 
                whiskerprops=dict(linewidth=2.5), 
                medianprops=dict(linewidth=2.5,color=default_colors[1]), 
                capprops=dict(linewidth=2.5,color=default_colors[2]), 
                flierprops=dict(markerfacecolor=default_colors[3], marker='o'))
plt.ylabel("Daily Power Usage [kWh]", fontweight='bold')
plt.xlabel("Prosumers", fontweight='bold')
plt.title("Box Plots of Daily Power Usage", fontweight='bold')
plt.grid()
plt.tight_layout()

usage_stats = pd.DataFrame({'House': labels, 'Minimum': min_list, 'Median': med_list, 'Mean': mean_list, 'Maximum': max_list, 'Num_Years': years_list})

# get rid of skewed data
usage_stats = usage_stats[usage_stats['Num_Years'] > 0.9]

highest_3 = list(usage_stats['House'].loc[usage_stats['Median'].nlargest(3).index])

lowest_3 = list(usage_stats['House'].loc[usage_stats['Median'].nsmallest(3).index])

# calculate the absolute difference from the target value
abs_diff = (usage_stats['Median'] - np.median(usage_stats['Median'])).abs()

# Use .nsmallest(n) to find the indices of the n smallest differences
middle_4 =  list(usage_stats['House'].loc[abs_diff.nsmallest(5).index])

plt.figure(figsize=figsize)
usage_stats['Median'].plot(kind='hist', grid=True)
plt.ylabel("Frequency", fontweight='bold')
plt.xlabel("Median Daily Power Usage [kWh]", fontweight='bold')
plt.title("Histogram of Median Daily Power Usage", fontweight='bold')
plt.tight_layout()

plt.figure(figsize=figsize)
usage_stats['Mean'].plot(kind='hist', grid=True)
plt.ylabel("Frequency", fontweight='bold')
plt.xlabel("Mean Daily Power Usage [kWh]", fontweight='bold')
plt.title("Histogram of Mean Daily Power Usage", fontweight='bold')
plt.tight_layout()

# Merge energy data with metadata
merged_data = pd.merge(energy_data, metadata, on='House', how='left')

# Merge with weather data
merged_data = pd.merge(merged_data, weather_data, on=['date', 'hour', 'Region'], how='left')

# fix day NaNs
merged_data['day'] = merged_data['date'].dt.day_name()
merged_data['weekend'] = merged_data['day'].isin(['Saturday', 'Sunday']).astype(int)

def first_weekday(year, month, weekday):
    """
    weekday:
        Monday=0
        Tuesday=1
        ...
        Sunday=6
    """
    d = pd.Timestamp(year, month, 1)

    while d.weekday() != weekday:
        d += Day(1)

    return d

def nth_weekday(year, month, weekday, n):
    return first_weekday(year, month, weekday) + Week(n-1)

def last_weekday(year, month, weekday):

    d = pd.Timestamp(year, month + 1, 1) - Day(1)

    while d.weekday() != weekday:
        d -= Day(1)

    return d

def canadian_holidays(year):

    easter_sunday = pd.Timestamp(easter(year))

    dict_holidays = {
        "New Years":
            pd.Timestamp(year,1,1),
        "Good Friday":
            easter_sunday - Day(2),
        "Easter Monday":
            easter_sunday + Day(1),
        "Victoria Day":
            last_weekday(year,5,0) - Week(1),
        "Canada Day":
            pd.Timestamp(year,7,1),
        "Civic Day":
            first_weekday(year,8,0),
        "Labour Day":
            first_weekday(year,9,0),
        "Thanksgiving":
            nth_weekday(year,10,0,2),
        "Remembrance Day":
            pd.Timestamp(year,11,11),
        "Christmas Day":
            pd.Timestamp(year,12,25),
        "Boxing Day":
            pd.Timestamp(year,12,26),
        "Family Day":
            nth_weekday(year,2,0,3)
    }
    return dict_holidays

def merge_holidays(df_in):
    
    years = range(
        df_in['date'].dt.year.min(),
        df_in['date'].dt.year.max()+1
    )

    holiday_rows = []

    for year in years:
        dict_holidays = canadian_holidays(year)
        for name, date in dict_holidays.items():
            holiday_rows.append({"date": date,"holiday": name})
            
    df_holidays = pd.DataFrame(holiday_rows)

    df_out = pd.merge(df_in, df_holidays, on='date', how='left')
    df_out['holiday'] = df_out['holiday'].fillna('None')
    
    return df_out

merged_data = merge_holidays(merged_data)

def dst_start(year):
    return nth_weekday(year,3,6,2)

def dst_end(year):
    return first_weekday(year,11,6)

def add_dst(df_in):
    
    years = range(
        df_in['date'].dt.year.min(),
        df_in['date'].dt.year.max()+1
    )
    
    start_days = []
    stop_days = []
    
    for year in years:
        if year < 2027:
            start_days.append(dst_start(year))
        if year < 2026:
            stop_days.append(dst_end(year))
            
    start_indices = df_in.index[(df_in['date'].isin(start_days)) & (df_in['hour'] == 3)].tolist()
    stop_indices = df_in.index[(df_in['date'].isin(stop_days)) & (df_in['hour'] == 3)].tolist()
    
    if stop_indices[0] < start_indices[0]:
        start_indices.insert(0,0)
        
    if stop_indices[-1] < start_indices[-1]:
        stop_indices.append(len(df_in))
    
    df_out = df_in.copy()
    df_out['dst'] = 0
    
    for start, stop in zip(start_indices, stop_indices):
        df_out.loc[start:stop-1, 'dst'] = 1
        
    return df_out
    
# # Creating new features
merged_data['day_of_month'] = merged_data['date'].dt.day
merged_data['month'] = merged_data['date'].dt.month

# Drop columns with high NaN values or irreparable data
columns_to_drop = ['FirstReading', 'LastReading','SN','Cover']
merged_data = merged_data.drop(columns=columns_to_drop)

merged_data.fillna({
    'RUs': 0, 
    'EVs': 0, 
    'weekend': 0, 
    'dst': 0,
}, inplace=True)

# ### prepare data for forecast
df_forecast = merged_data.copy()

# get rid of prosumer 15 (only one from WYJ region)
df_forecast = df_forecast[df_forecast['House'] != 15].reset_index(drop=True)
# get rid of prosumer 7 since we don't know the region, housetype, facing, or HVAC
df_forecast = df_forecast[df_forecast['House'] != 7].reset_index(drop=True)

# Creating new features
df_forecast['day_of_week'] = df_forecast['date'].dt.dayofweek
df_forecast["day_of_year"] = df_forecast['date'].dt.dayofyear
df_forecast['hour_sin'] = np.sin(2*np.pi*df_forecast['hour']/24)
df_forecast['hour_cos'] = np.cos(2*np.pi*df_forecast['hour']/24)
df_forecast['dow_sin'] = np.sin(2*np.pi*df_forecast['day_of_week']/7)
df_forecast['dow_cos'] = np.cos(2*np.pi*df_forecast['day_of_week']/7)
df_forecast["year"] = df_forecast["date"].dt.year
df_forecast["days_in_year"] = df_forecast["year"].apply(
    lambda y: 366 if pd.Timestamp(f"{y}-12-31").dayofyear == 366 else 365)
df_forecast["doy_sin"] = np.sin(2*np.pi*df_forecast["day_of_year"]/df_forecast["days_in_year"]) 
df_forecast["doy_cos"] = np.cos(2*np.pi*df_forecast["day_of_year"]/df_forecast["days_in_year"])

# get rid of unnecessary features
columns_to_drop = ['Region','day','hour','EVs','day_of_month','month','day_of_week','day_of_year','year','days_in_year','weather']
df_forecast = df_forecast.drop(columns=columns_to_drop)
df_forecast = df_forecast.drop(list(df_forecast.filter(regex='_missing')), axis=1)

# LSTM
df_lstm = df_forecast.copy()

columns_to_drop_lstm = ['date']
df_lstm = df_lstm.drop(columns=columns_to_drop_lstm)

#split into train, validation, and test sets
def time_split_by_group(df, group_col, train_frac=0.8, val_frac=0.95):
    train_parts = []
    val_parts = []
    test_parts = []
    
    groups = df[group_col].unique()
    
    for group in groups:
        df_group = df.loc[df[group_col]==group]
        train_idx = int(len(df_group)*train_frac)
        val_idx = int(len(df_group)*val_frac)
        train_parts.append(df_group.iloc[:train_idx])
        val_parts.append(df_group.iloc[train_idx:val_idx])
        test_parts.append(df_group.iloc[val_idx:])
    
    train = pd.concat(train_parts).reset_index(drop=True)
    val = pd.concat(val_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    
    return train, val, test

df_lstm_train, df_lstm_val, df_lstm_test = time_split_by_group(df_lstm,'House')

def add_rolling_statistics(df_in, col, window=24, shift=0):
    df = df_in.copy()
    df[f'rolling_mean_{window}'] = df[col].shift(shift).rolling(window).mean().reset_index(level=0, drop=True)
    df[f'rolling_std_{window}'] = df[col].shift(shift).rolling(window).std().reset_index(level=0, drop=True)
    return df

# prepare df for LSTM forecast
def prepare_for_LSTM_kWh(df_in, window):
    
    # Rolling statistics (should be shifted by one for regression, not shifted for LSTM). Should do this after train/val/test split
    df_out = add_rolling_statistics(df_in, 'energy_kWh', window)

    # Drop rows with NaN values created by rolling and lag features
    df_out = df_out.dropna().reset_index(drop=True)

    # Interaction terms
    df_out['temp_humidity_interaction'] = df_out['temperature'] * df_out['humidity']
    df_out['temp_pressure_interaction'] = df_out['temperature'] * df_out['pressure']
    df_out['humidity_pressure_interaction'] = df_out['humidity'] * df_out['pressure']
    
    return df_out

WINDOW_SIZE = 24

df_lstm_train = prepare_for_LSTM_kWh(df_lstm_train, WINDOW_SIZE)
df_lstm_val = prepare_for_LSTM_kWh(df_lstm_val, WINDOW_SIZE)
df_lstm_test = prepare_for_LSTM_kWh(df_lstm_test, WINDOW_SIZE)

# Convert categorical features to one-hot encoded features
encoder_kWh_lstm = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
categorical_columns = ['House','HouseType','Facing','HVAC','holiday']

# Learn categories from the training data only
encoder_kWh_lstm.fit(df_lstm_train[categorical_columns])

# scale X vars
continuous_vars = ['RUs','temperature','humidity','pressure','rolling_mean_24','rolling_std_24',
                   'temp_humidity_interaction','temp_pressure_interaction','humidity_pressure_interaction']
scaler_X = StandardScaler()
scaler_X = scaler_X.fit(df_lstm_train[continuous_vars])
scaler_y = StandardScaler()
scaler_y = scaler_y.fit(df_lstm_train[['energy_kWh']])

def df_to_X_y(df, window_size=24):
    df_as_np = df.to_numpy(dtype=np.float32)
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
        y.append(df_as_np[i+window_size][0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def encode_scale_Xy(df_in, group_col, window_size_in, encoder_in, categorical_columns, continuous_vars, scaler_X, scaler_y):
    group_dfs = []
    group_dfs_scaled = []
    X_arrays = []
    y_arrays = []
    
    groups = df_in[group_col].unique()
    
    for group in groups:
        df_group = df_in.loc[df_in[group_col]==group]
        if len(df_group) < window_size_in:
            continue
        df_group = apply_one_hot_encoder(df_group, encoder_in, categorical_columns)
        df_group_scaled = df_group.copy()
        df_group_scaled[continuous_vars] = scaler_X.transform(df_group_scaled[continuous_vars])
        df_group_scaled['energy_kWh'] = scaler_y.transform(df_group_scaled[['energy_kWh']])
        X, y = df_to_X_y(df_group_scaled, window_size_in)
        
        group_dfs.append(df_group)
        group_dfs_scaled.append(df_group_scaled)
        X_arrays.append(X)
        y_arrays.append(y)
    
    # free up some memory
    del df_group
    del df_group_scaled    
    
    df_out = pd.concat(group_dfs).reset_index(drop=True)
    df_out_scaled = pd.concat(group_dfs_scaled).reset_index(drop=True)
    X_lstm = np.concatenate(X_arrays, axis=0, dtype=np.float32)
    y_lstm = np.concatenate(y_arrays, axis=0, dtype=np.float32)
    
    return df_out, df_out_scaled, X_lstm, y_lstm

df_lstm_train, df_lstm_train_scaled, X_train_lstm, y_train_lstm = encode_scale_Xy(df_lstm_train, 'House', WINDOW_SIZE, encoder_kWh_lstm, categorical_columns, continuous_vars, scaler_X, scaler_y)
df_lstm_val, df_lstm_val_scaled, X_val_lstm, y_val_lstm = encode_scale_Xy(df_lstm_val, 'House', WINDOW_SIZE, encoder_kWh_lstm, categorical_columns, continuous_vars, scaler_X, scaler_y)
df_lstm_test, df_lstm_test_scaled, X_test_lstm, y_test_lstm = encode_scale_Xy(df_lstm_test, 'House', WINDOW_SIZE, encoder_kWh_lstm, categorical_columns, continuous_vars, scaler_X, scaler_y)

layerLSTM1 = 64
# layerLSTM2 = 64
dropout = 0
layerDense = 32
learning_rate = 0.0001

model = Sequential()
model.add(InputLayer((WINDOW_SIZE, 87)))
model.add(LSTM(layerLSTM1, dropout=dropout, return_sequences=False))
# model.add(LSTM(layerLSTM2, dropout=dropout))
model.add(LayerNormalization())
model.add(Dense(layerDense, 'relu'))
model.add(Dense(1, 'linear'))
model.summary()

# Define checkpoints and early stopping
folder_name = f"Wsize{WINDOW_SIZE}_LSTM{layerLSTM1}_drpout{dropout}_Dense{layerDense}_LR{learning_rate}_norm_mse/"
cp = ModelCheckpoint(folder_name, monitor='val_loss', save_best_only=True)
es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Compile the model
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

# Train the model
del df_lstm_test_scaled, df_lstm_train_scaled, df_lstm_val_scaled
history_lstm = model.fit(X_train_lstm, y_train_lstm, validation_data=(X_val_lstm, y_val_lstm), epochs=100, callbacks=[cp,es])

# Plot training & validation loss values
plt.figure()
plt.plot(history_lstm.history['loss'], label='Training Loss')
plt.plot(history_lstm.history['val_loss'], label='Validation Loss')
plt.title('Model Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()

# load the best checkpoint
model = load_model(folder_name)
test_predictions = model.predict(X_test_lstm)
test_predictions = scaler_y.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
y_test_lstm_unscaled = scaler_y.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()
plt.figure()
plt.plot(y_test_lstm_unscaled[0:100], label='actual')
plt.plot(test_predictions[0:100], label='predicted')
plt.legend()
plt.grid()
plt.tight_layout()

# Calculate metrics
mae = mse(y_test_lstm_unscaled,test_predictions)
rmse = np.sqrt(mean_squared_error(y_test_lstm_unscaled, test_predictions))
r2 = r2_score(y_test_lstm_unscaled, test_predictions)

# Print the scores
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²: {r2:.4f}')

# extend prosumer data by predicting out to 20 years
# Chosen prosumers: middle:[8, 9, 10, 5] high:[28, 18, 14] low:[22, 2, 17]
prosumers = [28,18]

def extend_prosumer_20yr(df_in, prosumer, start_date):
    df_out = df_in.copy()
    df_out = df_out[df_out['House'] == prosumer]
    
    # add rows out to end date
    df_out['datetime'] = pd.to_datetime(df_out['date'])+pd.to_timedelta(df_out['hour']-1, unit='h')
    df_out = df_out.set_index('datetime')
    df_out = df_out.reindex(pd.date_range(df_out.index.min(), pd.to_datetime(start_date)+relativedelta(years=20), freq='h'))
    df_out['date'] = pd.to_datetime(df_out.index.strftime("%Y-%m-%d"))
    df_out['hour'] = df_out.index.hour+1
    df_out = df_out.reset_index(drop=True)
    
    df_out['House'] = prosumer
    
    df_out = add_dst(df_out)
    
    return df_out

df_P28_20yr = extend_prosumer_20yr(energy_data, 28, '2019-01-01 00:00:00' )
# df_P18_20yr = extend_prosumer_20yr(energy_data, 18, '2019-01-01 00:00:00' )
# df_P14_20yr = extend_prosumer_20yr(energy_data, 14, '2019-01-01 00:00:00' )
# df_P22_20yr = extend_prosumer_20yr(energy_data, 22, '2019-01-01 00:00:00' )
# df_P2_20yr = extend_prosumer_20yr(energy_data, 2, '2019-01-01 00:00:00' )
# df_P17_20yr = extend_prosumer_20yr(energy_data, 17, '2019-01-01 00:00:00' )
# df_P8_20yr = extend_prosumer_20yr(energy_data, 8, '2019-01-01 00:00:00' )
# df_P9_20yr = extend_prosumer_20yr(energy_data, 9, '2019-01-01 00:00:00' )
# df_P10_20yr = extend_prosumer_20yr(energy_data, 10, '2019-01-01 00:00:00' )
# df_P5_20yr = extend_prosumer_20yr(energy_data, 5, '2019-01-01 00:00:00' )

def prepare_extended_df(df_in, metadata, encoder_in, WINDOW_SIZE):
    
    # Merge with metadata
    df_out = pd.merge(df_in, metadata, on='House', how='left')

    # Merge with weather data
    weather_data_ext = pd.read_csv('dataverse_files/Weather_YVR_extended.csv')
    weather_data_ext['date'] = pd.to_datetime(weather_data_ext['date'])
    df_out = pd.merge(df_out, weather_data_ext, on=['date', 'hour'], how='left')

    df_out['temp_missing'] = 0
    df_out['humid_missing'] = 0
    df_out['press_missing'] = 0
    df_out['weather_missing'] = 0

    # fix day NaNs
    df_out['day'] = df_out['date'].dt.day_name()
    df_out['weekend'] = df_out['day'].isin(['Saturday', 'Sunday']).astype(int)

    # holiday data
    df_out = merge_holidays(df_out)

    # Creating new features
    df_out['day_of_month'] = df_out['date'].dt.day
    df_out['month'] = df_out['date'].dt.month

    # Drop columns with high NaN values or irreparable data
    columns_to_drop = ['FirstReading', 'LastReading','SN','Cover']
    df_out = df_out.drop(columns=columns_to_drop)

    df_out.fillna({
        'RUs': 0, 
        'EVs': 0, 
        'weekend': 0, 
        'dst': 0,
    }, inplace=True)

    # Creating new features
    df_out['day_of_week'] = df_out['date'].dt.dayofweek
    df_out['day_of_year'] = df_out['date'].dt.dayofyear
    df_out['hour_sin'] = np.sin(2*np.pi*df_out['hour']/24)
    df_out['hour_cos'] = np.cos(2*np.pi*df_out['hour']/24)
    df_out['dow_sin'] = np.sin(2*np.pi*df_out['day_of_week']/7)
    df_out['dow_cos'] = np.cos(2*np.pi*df_out['day_of_week']/7)
    df_out["year"] = df_out["date"].dt.year
    df_out["days_in_year"] = df_out["year"].apply(
        lambda y: 366 if pd.Timestamp(f"{y}-12-31").dayofyear == 366 else 365)
    df_out["doy_sin"] = np.sin(2*np.pi*df_out["day_of_year"]/df_out["days_in_year"]) 
    df_out["doy_cos"] = np.cos(2*np.pi*df_out["day_of_year"]/df_out["days_in_year"])

    # get rid of unnecessary features
    columns_to_drop = ['Region','day','hour','EVs','day_of_month','month','day_of_week','day_of_year','year','days_in_year','weather']
    df_out = df_out.drop(columns=columns_to_drop)
    df_out = df_out.drop(list(df_out.filter(regex='_missing')), axis=1)
    
    columns_to_drop_lstm = ['date']
    df_out = df_out.drop(columns=columns_to_drop_lstm)
    
    # Rolling statistics (should be shifted by one for regression, not shifted for LSTM). Should do this after train/val/test split
    df_out = add_rolling_statistics(df_out, 'energy_kWh', window=WINDOW_SIZE)

    # Interaction terms
    df_out['temp_humidity_interaction'] = df_out['temperature'] * df_out['humidity']
    df_out['temp_pressure_interaction'] = df_out['temperature'] * df_out['pressure']
    df_out['humidity_pressure_interaction'] = df_out['humidity'] * df_out['pressure']
    
    categorical_columns = ['House','HouseType','Facing','HVAC','holiday']
    df_out = apply_one_hot_encoder(df_out, encoder_in, categorical_columns)
    
    return df_out

df_P28_20yr = prepare_extended_df(df_P28_20yr, metadata, encoder_kWh_lstm, WINDOW_SIZE)

def scale_Xy(df_in, scalerX_in, scalery_in):
    # scale X and y vars
    df_out = df_in.copy()
    continuous_vars = ['RUs','temperature','humidity','pressure','rolling_mean_24','rolling_std_24',
                       'temp_humidity_interaction','temp_pressure_interaction','humidity_pressure_interaction']
    df_out[continuous_vars] = scalerX_in.transform(df_out[continuous_vars])
    df_out['energy_kWh'] = scalery_in.transform(df_out[['energy_kWh']])
    return df_out

nanidx = df_P28_20yr['energy_kWh'].isna().idxmax()
# lastidx = len(df_P28_20yr)
lastidx = nanidx+75
while nanidx < lastidx:
    df_P28_20yr_last24 = df_P28_20yr.iloc[nanidx-WINDOW_SIZE:nanidx].copy().reset_index(drop=True)
    df_P28_last24_scaled = scale_Xy(df_P28_20yr_last24, scaler_X, scaler_y)
    final_X = df_P28_last24_scaled.to_numpy()[None, :, :]
    next_prediction_lstm = model.predict(final_X)
    next_prediction_lstm_unscaled = scaler_y.inverse_transform(next_prediction_lstm)
    df_P28_20yr.loc[nanidx, 'energy_kWh'] = next_prediction_lstm_unscaled[0]
    df_P28_20yr.loc[nanidx,'rolling_mean_24'] = np.mean(df_P28_20yr.loc[nanidx-(WINDOW_SIZE-1):nanidx,'energy_kWh'])
    df_P28_20yr.loc[nanidx,'rolling_std_24'] = np.std(df_P28_20yr.loc[nanidx-(WINDOW_SIZE-1):nanidx,'energy_kWh'],ddof=1)
    
    if nanidx % 5000 == 0:
        df_P28_20yr.to_csv('df_P28_20yr_temp.csv', index=False)
    
    nanidx += 1

df_P28_20yr.to_csv('df_P28_20yr_full.csv', index=False)

plt.figure()
plt.plot(df_P28_20yr.loc[nanidx-360:nanidx-75,'energy_kWh'], label='actual')
plt.plot(df_P28_20yr.loc[nanidx-75:nanidx-1,'energy_kWh'], label='forecast')
plt.grid()
plt.legend()
plt.tight_layout()
