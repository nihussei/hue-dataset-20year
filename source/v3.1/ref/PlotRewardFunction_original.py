

import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import datetime as dt
from dateutil.easter import easter
from pandas.tseries.offsets import Week, Day
from prosumer import Prosumer
from battery import Battery
from function_definitions import calculate_gammas, calculate_mu, total_net_cost, normalize, calculate_ede, calculate_welfare, calculate_P_life
import copy
import os
from pathlib import Path

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

def clean_up_weather_data(filepath,region):
    # Load YVR weather data
    df = pd.read_csv(filepath)
    
    # fill in missing hours with nans
    df['datetime'] = pd.to_datetime(df['date'])+pd.to_timedelta(df['hour']-1, unit='h')
    df = df.set_index('datetime')
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='h'))
    df['date'] = df.index.strftime("%Y-%m-%d")
    df['hour'] = df.index.hour+1
    df = df.reset_index(drop=True)
    # fill in the nans
    df['temp_missing'] = df['temperature'].isna().astype(int)
    df['humid_missing'] = df['humidity'].isna().astype(int)
    df['press_missing'] = df['pressure'].isna().astype(int)
    column_list = ['temperature','humidity','pressure']
    for column in column_list:
        nan_mask = df[column].isna()
        nan_indices = list(np.where(nan_mask==True)[0]) 
        for index in nan_indices:
            interpolated = np.nan
            try: # interpolate from the two hours on either side
                interpolated = (df.iloc[index+1][column]+df.iloc[index-1][column])/2
            except Exception:
                pass
            if not np.isnan(interpolated):
                df.loc[index,column] = round(interpolated,2)
                continue
            try: # interpolate from the two days on either side
                interpolated = (df.iloc[index+24][column] + df.iloc[index-24][column])/2
            except Exception:
                pass
            if not np.isnan(interpolated):
                df.loc[index,column] = round(interpolated,2)
                continue
            try: # grab from the day before
                interpolated = df.iloc[index-24][column]
            except Exception:
                pass
            if not np.isnan(interpolated):
                df.loc[index,column] = interpolated
                continue
            try: # grab from the day after
                interpolated = df.iloc[index+24][column]
            except Exception:
                pass
            if not np.isnan(interpolated):
                df.loc[index,column] = interpolated
    df['weather_missing'] = df['weather'].isna().astype(int)

    df['Region']=region
    
    return df

# create YVR weather dataframe
weather_yvr = clean_up_weather_data('dataverse_files/Weather_YVR.csv','YVR')
weather_yvr['weather'] = weather_yvr['weather'].ffill() # forward fill missing weather data

# create WYJ weather dataframe
weather_wyj = clean_up_weather_data('dataverse_files/Weather_WYJ.csv','WYJ')
# use weather data from yvr region
lookup = weather_yvr.set_index(['date', 'hour'])['weather']
weather_wyj['weather'] = pd.Series(list(zip(weather_wyj['date'], weather_wyj['hour']))).map(lookup)

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
middle_4 =  list(usage_stats['House'].loc[abs_diff.nsmallest(4).index])

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

# Creating new features
merged_data['day_of_month'] = merged_data['date'].dt.day
merged_data['month'] = merged_data['date'].dt.month

# prepare solar data
solar_data['day_of_month'] = solar_data['date'].dt.day
solar_data['month'] = solar_data['date'].dt.month
solar_data['hour'] += 1 # change hour to be 1 indexed instead of 0 indexed
solar_data = solar_data.drop(columns='date')
# merged_data = pd.merge(merged_data, solar_data, on=['day_of_month','month', 'hour'], how='left')

# add high, medium, and low output scenarios to solar data
solar_data['output_high'] = 1.3*solar_data['ac_output']/28

solar_data['output_low'] = 0.7*solar_data['ac_output']/28

solar_data['output_avg'] = solar_data['ac_output']/28

# Drop columns with high NaN values or irreparable data
columns_to_drop = ['FirstReading', 'LastReading','SN','Cover']
merged_data = merged_data.drop(columns=columns_to_drop)

merged_data.fillna({
    'dc_output': 0, 
    'ac_output': 0, 
    'RUs': 0, 
    'EVs': 0, 
    'weekend': 0, 
    'dst': 0,
    'holiday': 0,
}, inplace=True)

# ### Comparison of Allocation Methods with chosen prosumers
# choose prosumers 
# highest_3 = [28, 18, 14]
# lowest_3 = [22, 2, 6]
# middle_4 = [8, 9, 10, 7]
prosumers = [28, 18, 14, 22, 2, 6, 8, 9, 10, 7]

# make dataframes for each needed time interval
df_1year = []
df_24hrs = []
df_winter = []
df_summer = []

solar_data_high = solar_data[['hour', 'day_of_month', 'month', 'output_high']].copy()
solar_data_high = solar_data_high.rename(columns={'output_high': 'ac_output'})
solar_data_low = solar_data[['hour', 'day_of_month', 'month', 'output_low']].copy()
solar_data_low = solar_data_low.rename(columns={'output_low': 'ac_output'})
solar_data_avg = solar_data[['hour', 'day_of_month', 'month', 'output_avg']].copy()
solar_data_avg = solar_data_avg.rename(columns={'output_avg': 'ac_output'})

for prosumer in prosumers:
    prosumer_data = merged_data[merged_data['House'] == prosumer].copy()
    # prosumer_data = prosumer_data.drop(columns=['date'])
    
    # drop leap day (for comparison purposes)
    rows = prosumer_data[(prosumer_data["month"] == 2) & (prosumer_data["day_of_month"] == 29)].index
    prosumer_data = prosumer_data.drop(rows)
    
    # low/high, low/low, low/med, high/high, high/low, high/med, med/high, med/low, med/med, med/med
    if prosumer in [28, 22, 8]:
        prosumer_data = pd.merge(prosumer_data, solar_data_high, on=['day_of_month','month', 'hour'], how='left')
    elif prosumer in [18, 2, 9]:
        prosumer_data = pd.merge(prosumer_data, solar_data_low, on=['day_of_month','month', 'hour'], how='left')
    else:
        prosumer_data = pd.merge(prosumer_data, solar_data_avg, on=['day_of_month','month', 'hour'], how='left')
        
    # one year
    first_index = (prosumer_data['month'] == 8).idxmax()
    df = prosumer_data.loc[first_index:first_index+365*24-1].copy() # one year
    df_1year.append(df)
    
    # 24 hours
    df = prosumer_data.loc[first_index:first_index+23].copy() # one day
    df_24hrs.append(df)
    
    # winter
    first_index = (prosumer_data['month'] == 12).idxmax()
    last_index = (prosumer_data.loc[first_index:]['month'] == 3).idxmax()
    df = prosumer_data.loc[first_index:last_index-1].copy() # 3 months
    df_winter.append(df)
    
    # summer
    first_index = ((prosumer_data['month'] == 6) & ( prosumer_data['day_of_month'] == 1)).idxmax()
    last_index = (prosumer_data.loc[first_index:]['month'] == 9).idxmax()
    df = prosumer_data.loc[first_index:last_index-1].copy() # 3 months
    df_summer.append(df)
        
df_1year = pd.concat(df_1year, ignore_index=True)
df_24hrs = pd.concat(df_24hrs, ignore_index=True)
df_winter = pd.concat(df_winter, ignore_index=True)
df_summer = pd.concat(df_summer, ignore_index=True)

df_window = df_1year.copy()

prosumer_list = []
for p in prosumers:
    prosumer = Prosumer(df_window[df_window['House'] == p])
    prosumer.priority_weight = np.random.randint(1,len(prosumers)+1)
    prosumer_list.append(prosumer)

bess_capacity = 40  # kWh
bess_soc = 0.5  # initial state of charge, fraction
    
def proportional_allocation(prosumer_list, BESS):
    
    prosumers_data = copy.deepcopy(prosumer_list)
    
    num_intervals = len(prosumers_data[0].timeData)
    
    # reset values    
    bess_soc_list = []  # Store BESS SoC over time
    for p in prosumers_data:
        p.timeData['energy_allocated'] = np.zeros(num_intervals)
        p.timeData['accepted_pv'] = np.zeros(num_intervals)
        
    bess_soc = BESS.soc_init*BESS.capacity_init
            
    # Proportional allocation algorithm (Alg. 1)
    for t in range(num_intervals):
        
        if t > 0:
            bess_capacity = BESS.timeData['capacity'][t-1]
        else:
            bess_capacity = BESS.capacity_init 
        
        # charging phase
        net_pv_power_total = sum(np.array([p.timeData.loc[t,'contribution'] for p in prosumers_data])) # (eq. 2)
        SoC_rem = bess_capacity*BESS.soc_max-bess_soc
        if net_pv_power_total <= SoC_rem:
            for p in prosumers_data:
                p.timeData.loc[t,'accepted_pv'] = p.timeData.loc[t,'contribution']
        else:
            for p in prosumers_data:
                acceptance_factor = 0 if net_pv_power_total == 0 else p.timeData.loc[t,'contribution']/net_pv_power_total # (eq. 3)
                p.timeData.loc[t,'accepted_pv'] = SoC_rem*acceptance_factor # (eq. 4)
        accepted_pv_power = np.array([p.timeData.loc[t,'accepted_pv'] for p in prosumers_data])
        bess_soc = min(bess_capacity*BESS.soc_max, bess_soc+np.sum(accepted_pv_power)) # (eq. 5)
        
        # discharging phase
        net_load_total = np.sum([p.timeData.loc[t,'net_load'] for p in prosumers_data])
        SoC_rem = bess_soc-bess_capacity*BESS.soc_min
        if SoC_rem >= net_load_total: # (eq. 7)
            for p in prosumers_data:
                p.timeData.loc[t,'energy_allocated'] = p.timeData.loc[t,'net_load'] # (eq. 8)
        else:
            for p in prosumers_data:
                allocation_factor = p.timeData.loc[t,'net_load']/net_load_total # (eq. 9)
                p.timeData.loc[t,'energy_allocated'] = SoC_rem*allocation_factor # (eq. 10)      
        energy_allocated = np.array([p.timeData.loc[t,'energy_allocated'] for p in prosumers_data])
        bess_soc = max(bess_capacity*BESS.soc_min, bess_soc-np.sum(energy_allocated)) # (eq. 11)
        
        # update battery model
        BESS.timeData['SoC'][t] = bess_soc/bess_capacity
        BESS.timeData['T_c'][t] = BESS.T_ref
        BESS.timeData['DoD'][t] = (1-bess_soc/bess_capacity)
        BESS.calculate_degradation(t)
        
    return prosumers_data, bess_soc_list
    
def priority_based_allocation(prosumer_list, BESS):
    
    prosumers_data = copy.deepcopy(prosumer_list)
    
    num_intervals = len(prosumers_data[0].timeData)
    
    # reset values    
    bess_soc_list = []  # Store BESS SoC over time
    for p in prosumers_data:
        p.timeData['energy_allocated'] = np.zeros(num_intervals)
        p.timeData['accepted_pv'] = np.zeros(num_intervals)
        
    bess_soc = BESS.soc_init*BESS.capacity_init
        
    # Priority-based allocation algorithm (Alg. 2)
    for t in range(num_intervals):
        
        if t > 0:
            bess_capacity = BESS.timeData['capacity'][t-1]
        else:
            bess_capacity = BESS.capacity_init 
            
        # charging phase
        net_pv_power_total = sum(np.array([p.timeData.loc[t,'contribution'] for p in prosumers_data])) # (eq. 2)
        SoC_rem = bess_capacity-bess_soc
        # follow baseline mechanism (eq. 14)
        if net_pv_power_total <= SoC_rem:
            for p in prosumers_data:
                p.timeData.loc[t,'accepted_pv'] = p.timeData.loc[t,'contribution']
        else:
            for p in prosumers_data:
                acceptance_factor = 0 if net_pv_power_total == 0 else p.timeData.loc[t,'contribution']/net_pv_power_total # (eq. 3)
                p.timeData.loc[t,'accepted_pv'] = SoC_rem*acceptance_factor # (eq. 4)  
        accepted_pv_power = np.array([p.timeData.loc[t,'accepted_pv'] for p in prosumers_data])
        bess_soc = min(bess_capacity*BESS.soc_max, bess_soc+np.sum(accepted_pv_power)) # (eq. 5/15)
        
        # discharging phase
        net_load_total = np.sum([p.timeData.loc[t,'net_load'] for p in prosumers_data])
        if bess_soc >= net_load_total: # (eq. 7/16)
            for p in prosumers_data:
                p.timeData.loc[t,'energy_allocated'] = p.timeData.loc[t,'net_load'] # (eq. 8/17)
        else:
            for p in prosumers_data:
                p_score = p.priority_weight*p.timeData.loc[t,'net_load'] # (eq. 12)
                priority_factor = p_score/sum(ps.priority_weight*ps.timeData.loc[t,'net_load'] for ps in prosumers_data) # (eq. 13)
                p.timeData.loc[t,'energy_allocated'] = priority_factor*bess_soc
        energy_allocated_total = np.sum(np.array([p.timeData.loc[t,'energy_allocated'] for p in prosumers_data]))
        bess_soc = max(bess_capacity*BESS.soc_min, bess_soc-energy_allocated_total) # (eq. 11)
        bess_soc_list.append(bess_soc)  # Append current SoC to the list   
        
        # update battery model
        BESS.timeData['SoC'][t] = bess_soc/bess_capacity
        BESS.timeData['T_c'][t] = BESS.T_ref
        BESS.timeData['DoD'][t] = (1-bess_soc/bess_capacity)
        BESS.calculate_degradation(t)
        
    return prosumers_data, bess_soc_list
    
def contribution_based_allocation(prosumer_list, BESS):
    
    prosumers_data = copy.deepcopy(prosumer_list)
    
    num_intervals = len(prosumers_data[0].timeData)
    
    # reset values    
    bess_soc_list = []  # Store BESS SoC over time
    for p in prosumers_data:
        p.timeData['energy_allocated'] = np.zeros(num_intervals)
        p.timeData['accepted_pv'] = np.zeros(num_intervals)
        
    bess_soc = BESS.soc_init*BESS.capacity_init
        
    # Relative contribution-based algorithm
    alpha = 0.5
    beta = 0.5
    for t in range(num_intervals):
        
        if t > 0:
            bess_capacity = BESS.timeData['capacity'][t-1]
        else:
            bess_capacity = BESS.capacity_init 
            
        # Charging phase (Alg. 3)
        net_pv_power = np.array([p.timeData.loc[t,'contribution'] for p in prosumers_data])
        SoC_rem = bess_capacity-bess_soc
        if np.sum(net_pv_power) <= SoC_rem:
            for p in prosumers_data:
                p.timeData.loc[t,'accepted_pv'] = p.timeData.loc[t,'contribution']
        else:
            for p in prosumers_data:
                acceptance_factor = p.timeData.loc[t,'contribution']/np.sum(net_pv_power) # eq. 20
                p.timeData.loc[t,'accepted_pv'] = SoC_rem * acceptance_factor # eq. 21

        # update contribution index
        for p in prosumers_data:
            p.timeData.loc[t,'contribution_index'] = p.timeData.loc[t-1,'contribution_index'] if t > 0 else 1 # initialize CI to 1 at first time step
            C_nt = p.timeData.loc[t,'accepted_pv']/p.timeData.loc[t,'load'] # eq. 22. TODO: have a way to handle when load is 0
            p.timeData.loc[t,'contribution_index'] += alpha * C_nt # eq. 23
        
        # update SoC
        accepted_pv_power = np.array([p.timeData.loc[t,'accepted_pv'] for p in prosumers_data])
        bess_soc = min(bess_capacity*BESS.soc_max, bess_soc + np.sum(accepted_pv_power)) # eq. 24

        # discharging phase    
        if bess_soc >= np.sum([p.timeData.loc[t,'net_load'] for p in prosumers_data]):
            # Alg. 4
            for p in prosumers_data:
                Lmax = max(p.timeData.loc[:t+1,'load'])
                # p.timeData.loc[t,'contribution_index'] = p.timeData.loc[t-1,'contribution_index'] if t > 0 else 1
                Pmax = p.timeData.loc[t,'contribution_index']*Lmax/beta # eq. 25
                p.timeData.loc[t,'energy_allocated'] = min(p.timeData.loc[t,'net_load'],Pmax) # eq. 26
                
                # update contribution index
                Palc = p.timeData.loc[t,'energy_allocated']
                p.timeData.loc[t,'contribution_index'] = max(p.timeData.loc[t,'contribution_index']-
                                                                 beta*Palc/Lmax,0) # eq. 27. This is dangerous if CI ever goes to 0
        else:
            # Alg. 5
            SoC_avl = bess_soc
            for p in prosumers_data:
                if SoC_avl <= np.sum([p.timeData.loc[t,'net_load'] for p in prosumers_data]):
                    if t > 0:
                        CI_sum = sum(ps.timeData.loc[t-1,'contribution_index'] for ps in prosumers_data)
                        if CI_sum == 0:
                            CIR = 1/len(prosumers_data) # everyone has equal weight, protection so the denominator doesn't go to 0
                        else:
                            CIR = p.timeData.loc[t-1,'contribution_index']/CI_sum # eq. 29
                    else:
                        CIR = 1/len(prosumers_data) # first time step, everyone has equal weight
                        
                    # p.timeData.loc[t,'energy_allocated'] = max(p.timeData.loc[t,'net_load']-SoC_avl*CIR,0) # (eq. 30a)
                    p.timeData.loc[t,'energy_allocated'] = max(SoC_avl*CIR,0) # (eq. 30a)
                else:
                    p.timeData.loc[t,'energy_allocated'] = p.timeData.loc[t,'net_load'] # (eq. 30b)

                # update SoC_avl
                SoC_avl -= p.timeData.loc[t,'energy_allocated']
                
                # update contribution index
                Lmax = max(p.timeData.loc[:t+1,'load'])
                Palc = p.timeData.loc[t,'energy_allocated']
                # p.timeData.loc[t,'contribution_index'] = p.timeData.loc[t-1,'contribution_index'] if t > 0 else 1
                p.timeData.loc[t,'contribution_index'] = max(p.timeData.loc[t,'contribution_index']
                                                             -beta*Palc/Lmax,0) # eq. 27. This is dangerous if CI ever goes to 0
                
        # update SoC for that time step
        energy_allocated = np.array([p.timeData.loc[t,'energy_allocated'] for p in prosumers_data])
        bess_soc = max(bess_capacity*BESS.soc_min, bess_soc-np.sum(energy_allocated)) # eq. 28
        bess_soc_list.append(bess_soc)  # Append current SoC to the list
        
        # update battery model
        BESS.timeData['SoC'][t] = bess_soc/bess_capacity
        BESS.timeData['T_c'][t] = BESS.T_ref
        BESS.timeData['DoD'][t] = (1-bess_soc/bess_capacity)
        BESS.calculate_degradation(t)
        
    return prosumers_data, bess_soc_list

def real_time_allocation(prosumers_data, bess_capacity, bess_soc):
    
    rates = [6.72, 16.72, 11.72]
    for p in prosumers_data:
        conditions = [
            prosumers_data[p]['hour'].isin([24,1,2,3,4,5,6,7]),
            prosumers_data[p]['hour'].isin([17,18,19,20,21]),
            prosumers_data[p]['hour'].isin([8,9,10,11,12,13,14,15,16,22,23])
        ]
        rates = [6.72, 16.72, 11.72]
        prosumers_data[p]['rate_RTA'] = np.select(conditions, rates, default=0) 
        
    num_intervals = len(prosumers_data[next(iter(prosumers_data))]['load'])
    
    # reset values    
    bess_soc_list = []  # Store BESS SoC over time
    for p in prosumers_data:
        prosumers_data[p]['energy_allocated'] = np.zeros(num_intervals)
        prosumers_data[p]['accepted_pv'] = np.zeros(num_intervals)
            
    # Real-time allocation algorithm (Alg. 6)
    for t in range(num_intervals):
        # charging phase
        lambda_t = prosumers_data[next(iter(prosumers_data))].loc[t,'rate_RTA']
        delta_t = (lambda_t-min(rates))/(max(rates)-min(rates))
        omega_t = 1-delta_t
        
        net_pv_power_total = sum(np.array([prosumers_data[p].loc[t,'contribution'] for p in prosumers_data])) # (eq. 2)
        SoC_rem = bess_capacity-bess_soc
        if net_pv_power_total <= SoC_rem:
            for p in prosumers_data:
                prosumers_data[p].loc[t,'accepted_pv'] = prosumers_data[p].loc[t,'contribution']
        else:
            for p in prosumers_data:
                acceptance_factor = 0 if net_pv_power_total == 0 else prosumers_data[p].loc[t,'contribution']/net_pv_power_total # (eq. 3)
                prosumers_data[p].loc[t,'accepted_pv'] = SoC_rem*acceptance_factor*omega_t # (eq. 33)
        accepted_pv_power = np.array([prosumers_data[p].loc[t,'accepted_pv'] for p in prosumers_data])
        bess_soc = min(bess_capacity, bess_soc + np.sum(accepted_pv_power)) # (eq. 5)
        
        # discharging phase
        net_load_total = np.sum([prosumers_data[p].loc[t,'net_load'] for p in prosumers_data])
        if bess_soc >= net_load_total: # (eq. 7)
            for p in prosumers_data:
                prosumers_data[p].loc[t,'energy_allocated'] = prosumers_data[p].loc[t,'net_load'] # (eq. 8)
        else:
            for p in prosumers_data:
                allocation_factor = prosumers_data[p].loc[t,'net_load']/net_load_total # (eq. 9)
                prosumers_data[p].loc[t,'energy_allocated'] = bess_soc*allocation_factor*delta_t # (eq. 36)      
        energy_allocated = np.array([prosumers_data[p].loc[t,'energy_allocated'] for p in prosumers_data])
        bess_soc = max(0, bess_soc - np.sum(energy_allocated)) # (eq. 11)
        bess_soc_list.append(bess_soc)  # Append current SoC to the list   
    
    for p in prosumers_data:
        prosumers_data[p]['p_bought'] = (prosumers_data[p]['net_load']-prosumers_data[p]['energy_allocated']).clip(lower=0)
        prosumers_data[p]['paid_RTA'] = prosumers_data[p]['p_bought']*prosumers_data[p]['rate_RTA']/100    
    
    return prosumers_data, bess_soc_list

BESS = Battery(bess_capacity, bess_soc, len(prosumer_list[0].timeData))
BESS.k_T = 0.07 # C**-1
BESS.k_sigma = 2.0
BESS.k_delta1 = 1.0e-7
BESS.k_delta2 = 1.0e-7
BESS.k_t = 1.14*(1.0e-9) # hour^-1
BESS.C_repl = 16000
BESS.D_EoL = 0.2
# daily operating cost
BESS.eps_b = 0.02 # CAD$/kWh/day (typical values range between $0.01–0.05/kWh/day)

BESS_proportional = copy.deepcopy(BESS)
prosumer_list_proportional, bess_soc_list = proportional_allocation(prosumer_list, BESS_proportional)

BESS_priority = copy.deepcopy(BESS)
prosumer_list_priority, bess_soc_list = priority_based_allocation(prosumer_list, BESS_priority)

BESS_contribution = copy.deepcopy(BESS)
prosumer_list_contribution, bess_soc_list = contribution_based_allocation(prosumer_list, BESS_contribution)

# prosumers_data, bess_soc_list = real_time_allocation(prosumers_data, bess_capacity, bess_soc)
# make_plots(prosumers_data, bess_soc_list, 'realTimeAlloc', save_figs=True)

def pricing_TimeOfUse(df_in):
    df = df_in.copy()
    
    conditions = [
        df['hour'].isin([24,1,2,3,4,5,6,7]),
        df['hour'].isin([17,18,19,20,21]),
        df['hour'].isin([8,9,10,11,12,13,14,15,16,22,23])
    ]
    rates = [6.72, 16.72, 11.72]
    df['rate_TOU'] = np.select(conditions, rates, default=0) 
    df['paid_TOU'] = df['p_bought']*df['rate_TOU']/100
    
    return df

def pricing_TieredRate(df_in):
    df = df_in.copy()

    rates = [11.72, 14.08]
        
    threshold = round(len(df)/24)*22.1918 # Tier 1 threshold is calculated by multiplying the number of days in a billing period by 22.1918 kWh/day;
    
    # Running cumulative energy purchased
    df['cum_kwh'] = df['p_bought'].cumsum()
    
    # Assign rate based on threshold
    df['rate_TieredRate'] = np.where(
        df['cum_kwh'] <= threshold,
        rates[0],
        rates[1]
    )
            
    df['paid_TieredRate'] = df['p_bought']*df['rate_TieredRate']/100
    
    return df

def pricing_FlatRate(df_in):    
    df = df_in.copy()

    df['rate_FlatRate'] = 12.63
    df['paid_FlatRate'] = df['p_bought']*df['rate_FlatRate']/100
    
    return df

# calculate how much power each prosumer bought
for prosumer in prosumer_list_proportional:
    prosumer.timeData['p_bought'] = (prosumer.timeData['net_load']-prosumer.timeData['energy_allocated']).clip(lower=0)
    prosumer.timeData = pricing_TimeOfUse(prosumer.timeData)
    prosumer.timeData = pricing_TieredRate(prosumer.timeData)
    prosumer.timeData = pricing_FlatRate(prosumer.timeData)
    prosumer.timeData['paid'] = prosumer.timeData['paid_TOU']

for prosumer in prosumer_list_priority:
    prosumer.timeData['p_bought'] = (prosumer.timeData['net_load']-prosumer.timeData['energy_allocated']).clip(lower=0)
    prosumer.timeData = pricing_TimeOfUse(prosumer.timeData)
    prosumer.timeData = pricing_TieredRate(prosumer.timeData)
    prosumer.timeData = pricing_FlatRate(prosumer.timeData)
    prosumer.timeData['paid'] = prosumer.timeData['paid_TOU']

for prosumer in prosumer_list_contribution:
    prosumer.timeData['p_bought'] = (prosumer.timeData['net_load']-prosumer.timeData['energy_allocated']).clip(lower=0)
    prosumer.timeData = pricing_TimeOfUse(prosumer.timeData)
    prosumer.timeData = pricing_TieredRate(prosumer.timeData)
    prosumer.timeData = pricing_FlatRate(prosumer.timeData)
    prosumer.timeData['paid'] = prosumer.timeData['paid_TOU']

# calculate PV utilization
for prosumer in prosumer_list_proportional:
    prosumer.timeData['SCR'] = (prosumer.timeData[['pv', 'load']].min(axis='columns')+prosumer.timeData['energy_allocated'])/prosumer.timeData['load']

for prosumer in prosumer_list_priority:
    prosumer.timeData['SCR'] = (prosumer.timeData[['pv', 'load']].min(axis='columns')+prosumer.timeData['energy_allocated'])/prosumer.timeData['load']

for prosumer in prosumer_list_contribution:
    prosumer.timeData['SCR'] = (prosumer.timeData[['pv', 'load']].min(axis='columns')+prosumer.timeData['energy_allocated'])/prosumer.timeData['load']
  
# calculate mu for each prosumer
calculate_mu(prosumer_list_proportional)
calculate_mu(prosumer_list_priority)
calculate_mu(prosumer_list_contribution)

# calculate daily operating cost
BESS_proportional.timeData['C_op'] = np.full(len(BESS_proportional.timeData['D']), 
                                             BESS_proportional.eps_b*BESS_proportional.capacity_init)
BESS_priority.timeData['C_op'] = np.full(len(BESS_priority.timeData['D']),
                                         BESS_priority.eps_b*BESS_priority.capacity_init)
BESS_contribution.timeData['C_op'] = np.full(len(BESS_contribution.timeData['D']), 
                                             BESS_contribution.eps_b*BESS_contribution.capacity_init)

# calculate replacement cost for each prosumer
for prosumer in prosumer_list_proportional:
    prosumer.calculate_Delta_f(BESS_proportional)
    
for prosumer in prosumer_list_priority:
    prosumer.calculate_Delta_f(BESS_priority)   

for prosumer in prosumer_list_contribution:
    prosumer.calculate_Delta_f(BESS_contribution)
    
calculate_gammas(prosumer_list_proportional)
calculate_gammas(prosumer_list_priority)
calculate_gammas(prosumer_list_contribution)

L_EoL=0.20
rho_base = .015 # Annual degradation rate under the baseline operating strategy (so calendar aging only), assumed
Y_base = L_EoL/rho_base
Y_plan = 1

rho_op = 1.0-BESS_proportional.timeData['SoH'][8759] # Annual degradation rate under our optimized operating strategy.
Y_op = L_EoL/rho_op
for prosumer in prosumer_list_proportional:
    prosumer.calculate_C_repl(BESS_proportional, Y_op, Y_base, Y_plan)

rho_op = 1.0-BESS_priority.timeData['SoH'][8759] # Annual degradation rate under our optimized operating strategy.
Y_op = L_EoL/rho_op
for prosumer in prosumer_list_priority:
    prosumer.calculate_C_repl(BESS_priority, Y_op, Y_base, Y_plan)

rho_op = 1.0-BESS_contribution.timeData['SoH'][8759] # Annual degradation rate under our optimized operating strategy.
Y_op = L_EoL/rho_op
for prosumer in prosumer_list_contribution:
    prosumer.calculate_C_repl(BESS_contribution, Y_op, Y_base, Y_plan)

# calculate total cost for each prosumer    
total_net_cost(prosumer_list_proportional,BESS_proportional)
total_net_cost(prosumer_list_priority,BESS_priority)
total_net_cost(prosumer_list_contribution,BESS_contribution)

# reliability metrics
for prosumer in prosumer_list_proportional:
    prosumer.timeData['Reliability'] = 1.0

for prosumer in prosumer_list_priority:
    prosumer.timeData['Reliability'] = 1.0

for prosumer in prosumer_list_contribution:
    prosumer.timeData['Reliability'] = 1.0
    
# normalize the metrics
normalize(prosumer_list_proportional, 'SCR')
normalize(prosumer_list_proportional, 'Reliability')
normalize(prosumer_list_proportional, 'total_net_cost', maximize=False)

normalize(prosumer_list_priority, 'SCR')
normalize(prosumer_list_priority, 'Reliability')
normalize(prosumer_list_priority, 'total_net_cost', maximize=False)

normalize(prosumer_list_contribution, 'SCR')
normalize(prosumer_list_contribution, 'Reliability')
normalize(prosumer_list_contribution, 'total_net_cost', maximize=False)

# calculate Equally-distributed-equivalent welfare scores (EDE)
df_EDE_proportional = pd.DataFrame(columns=['Reliability','PV','Cost'],
                                   index=range(len(prosumer_list_proportional[0].timeData)))
df_EDE_priority = pd.DataFrame(columns=['Reliability','PV','Cost'],
                               index=range(len(prosumer_list_priority[0].timeData)))
df_EDE_contribution = pd.DataFrame(columns=['Reliability','PV','Cost'],
                                   index=range(len(prosumer_list_contribution[0].timeData)))

save_figs = True
# define the path
figure_dir = os.getcwd()+'\\reward_figures\\'
# create it if it doesn't exist
Path(figure_dir).mkdir(parents=True, exist_ok=True)
epsilon_list = [0.5, 1.0, 2.0]
k_list = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
for epsilon in epsilon_list:
    for k in k_list:

        df_EDE_proportional['Reliability'] = calculate_ede(prosumer_list_proportional,'Reliability_norm',epsilon=epsilon)
        df_EDE_proportional['PV'] = calculate_ede(prosumer_list_proportional,'SCR_norm',epsilon=epsilon)
        df_EDE_proportional['Cost'] = calculate_ede(prosumer_list_proportional,'total_net_cost_norm',epsilon=epsilon)
        
        df_EDE_priority['Reliability'] = calculate_ede(prosumer_list_priority,'Reliability_norm',epsilon=epsilon)
        df_EDE_priority['PV'] = calculate_ede(prosumer_list_priority,'SCR_norm',epsilon=epsilon)
        df_EDE_priority['Cost'] = calculate_ede(prosumer_list_priority,'total_net_cost_norm',epsilon=epsilon)
        
        df_EDE_contribution['Reliability'] = calculate_ede(prosumer_list_contribution,'Reliability_norm',epsilon=epsilon)
        df_EDE_contribution['PV'] = calculate_ede(prosumer_list_contribution,'SCR_norm',epsilon=epsilon)
        df_EDE_contribution['Cost'] = calculate_ede(prosumer_list_contribution,'total_net_cost_norm',epsilon=epsilon)
        
        # calculate J
        weights = {"PV": 1/3,
                   "Cost": 1/3,
                   "Reliability": 1/3}
        
        J_proportional = calculate_welfare(df_EDE_proportional, weights)
        J_priority = calculate_welfare(df_EDE_priority, weights)
        J_contribution = calculate_welfare(df_EDE_contribution, weights)
        
        # calculate P_life
        P_life_proportional = calculate_P_life(prosumer_list_proportional,BESS_proportional)
        P_life_priority = calculate_P_life(prosumer_list_priority,BESS_priority)
        P_life_contribution = calculate_P_life(prosumer_list_contribution,BESS_contribution)
        
        # calculate reward
        r_proportional = J_proportional-k*P_life_proportional
        r_priority = J_priority-k*P_life_priority
        r_contribution = J_contribution-k*P_life_contribution
        
        plt.figure(figsize=(8, 4))
        plt.plot(np.cumsum(r_proportional), label="Proportional Allocation")
        plt.plot(np.cumsum(r_priority), label="Priority-Based Allocation")
        plt.plot(np.cumsum(r_contribution), label="Contribution-Based Allocation")
        plt.title(f"Cumulative Rewards (ε={epsilon}, kappa={k})")
        plt.xlabel("Hour")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid()
        if save_figs:
            plt.savefig(os.path.join(figure_dir,f"cumulativeRewards_eps{epsilon}_kap{k}.png"), dpi=300)
        
        plt.figure(figsize=(8, 4))
        plt.plot(r_proportional, label="Proportional Allocation")
        plt.plot(r_priority, label="Priority-Based Allocation")
        plt.plot(r_contribution, label="Contribution-Based Allocation")
        plt.title(f"Hourly Rewards (ε={epsilon}, kappa={k})")
        plt.xlabel("Hour")
        plt.ylabel("Reward")
        plt.xlim(0, 48) 
        plt.legend()
        plt.grid()
        if save_figs:
            plt.savefig(os.path.join(figure_dir,f"hourlyRewards_eps{epsilon}_kap{k}.png"), dpi=300)
        
