from datetime import datetime
import os
import pickle

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tabulate import tabulate
import humanize
import numpy as np
import pandas as pd

from scripts.oversample import oversample
from scripts.time_categories import day_period, season
from scripts.violations import normalize_violations


# Flags
drop_location_raw = False
perform_oversampling = True
encode_labels = True

print('\nReading CSV...')
df = pd.read_csv('../data/CT-clean.csv', header=0)

# Drop non-essential base columns
print('\nDropping non-essential columns...')
drop_cols = [
    'county_fips',
    'driver_age',
    'driver_race_raw',
    'fine_grained_location',
    'id',
    'is_arrested',
    'officer_id',
    'police_department',
    'search_type_raw',
    'search_type',
    'state',
]
if drop_location_raw:
    drop_cols.append('location_raw')
df.drop(drop_cols, axis=1, inplace=True)

# Drop empty stop_outcome and county_name rows
print('\nDropping empty stop_outcome and county_name rows...')
df.dropna(subset=['stop_outcome', 'county_name'], axis=0, inplace=True)

# Remove records with age less than 15
print('\nDropping driver_age rows under 15 y.o....')
weird_ages_rows = df[df["driver_age_raw"] < 15]['driver_age_raw']
df.drop(index=weird_ages_rows.index, inplace=True)

# Fill in empty stop_time values with median value
print('\nFilling empty stop_time values with median...')
populated = df[df.stop_time.notnull()]['stop_time'].sort_values()
median_stop_time = populated.iloc[populated.shape[0] // 2]
df['stop_time'].fillna(median_stop_time, inplace=True)

# Categorize stop_time into time-of-day(morning, afternoon, evening, small hours) and stop_date by season
print('\nCategorizing stop_time and time-of-day...')
df['day_period'] = pd.to_datetime(df['stop_time']).apply(day_period)
df['season'] = df['stop_date'].apply(season)

# Transform driver_gender to binary
print('\nTransforming driver_gender to binary is_male...')
df['is_male'] = df['driver_gender'].apply(lambda x: 1 if x == 'M' else 0)

if not drop_location_raw:
    print('\nLabelEncoding location_raw...')
    encoder = LabelEncoder()
    df['location_raw'] = encoder.fit_transform(df['location_raw'])

# Normalize violations
print('\nNormalizing violations...')
df_violations = normalize_violations(df)

# Append one-hot encoded violations
print('\nAppending one-hot encoded violations to df...')
df = pd.concat([df, df_violations], axis=1)

# Drop columns no longer needed
print('\nDropping no longer needed columns...')
drop_cols = [
    'driver_gender',
#     'county_name',
#     'location_raw',
#     'officer_id',
    'stop_date',
    'stop_time',
    'violation_raw',
    'violation',
]
df.drop(drop_cols, axis=1, inplace=True)

# Drop duplicate rows
print('\nDropping duplicate rows...')
df.drop_duplicates(inplace=True)

# Convert booleans to 0 and 1
print('\nConverting boolean columns to 0 and 1...')
df['search_conducted'] = df['search_conducted'].apply(lambda x: int(x))
df['contraband_found'] = df['contraband_found'].apply(lambda x: int(x))

# Normalize driver_age
print('\nNormalizing driver_age...')
scaler = MinMaxScaler() # default=(0, 1)
df['driver_age_raw'] = scaler.fit_transform(df['driver_age_raw'].reshape(-1, 1))

# Categorical variables to one-hot encode or label encode
cols_to_encode = [
    'county_name',
    'day_period',
    'driver_race',
    'season',
    'stop_duration',
]

if encode_labels:
    print('\nLabelEncoding appropriate columns...')
    for col in cols_to_encode:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
else:
    df = pd.get_dummies(df, columns=cols_to_encode)

if perform_oversampling:
    print('\nOversampling rows...')
    df = oversample(df)

# Write dataframe to pickle file
filename = '../data/df'
if drop_location_raw:
    filename += '-no_loc_raw'
if encode_labels:
    filename += '-labelencoded'
if perform_oversampling:
    filename += '-oversampled'
filename += '.pkl'

print('\nDumping dataframe to pickle file...')
df.to_pickle(filename)

print('\nFinished preprocessing.')
