from datetime import datetime
import importlib
import os
import pickle

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle
from tabulate import tabulate
import humanize
import numpy as np
import pandas as pd

from scripts.sampling import oversample, undersample
from scripts.time_categories import day_period, season
from scripts.violations import normalize_violations


def split_train_test(df, decimal_pct):
    lop_off_idx = round(df.shape[0] * decimal_pct)
    test = df[:lop_off_idx]
    train = df[lop_off_idx:]
    return (train, test)


def categorize_dates_times(df):
    print('Categorizing stop_time and time-of-day...')
    df['hour'] = pd.to_numeric(df['stop_time'].apply(lambda x: str(x).split(':')[0]), downcast='signed')
    df['minute'] = pd.to_numeric(df['stop_time'].apply(lambda x: str(x).split(':')[1]), downcast='signed')
    df['month'] = pd.to_numeric(df['stop_date'].apply(lambda x: str(x).split('-')[1]), downcast='signed')
    df['day'] = pd.to_numeric(df['stop_date'].apply(lambda x: int(str(x).split('-')[2])), downcast='signed')
    return df


def drop_threshold_ages(df, age_threshold):
    print('Dropping driver_age rows under {} y.o....'.format(age_threshold))
    weird_ages_rows = df[df["driver_age_raw"] < 15]['driver_age_raw']
    df.drop(index=weird_ages_rows.index, inplace=True)
    return df


def binarize(df, columns):
    print('Converting boolean columns to 0 and 1...')
    for col in columns:
        df[col] = df[col].apply(lambda x: int(x))
    return df


def normalize_driver_age(df, columns):
    print('Normalizing driver_age...')
    scaler = MinMaxScaler() # default=(0, 1)
    for col in columns:
        df[col] = scaler.fit_transform(df[col].reshape(-1, 1))
    return df


def preprocess(df, **kwargs):
    # Set flags
    drop_location_raw = kwargs['drop_location_raw']
    perform_oversampling = kwargs['perform_oversampling']
    encode_labels = kwargs['encode_labels']

    print("Flags:\n\tdrop_location_raw: {}\n\tperform_oversampling: {}\n\tencode_labels: {}".format(drop_location_raw,
        perform_oversampling, encode_labels))

    # Drop non-essential base columns
    print('Dropping non-essential columns...')
    drop_cols = [
        'county_fips',
        # 'county_name',
        'driver_age',
        # 'driver_age_raw',
        'driver_race_raw',
        'driver_race',
        'fine_grained_location',
        'id',
        # 'is_arrested',
        'officer_id',
        'police_department',
        'search_type_raw',
        'search_type',
        'state',
        'search_conducted',
        'contraband_found',
    ]
    if drop_location_raw:
        drop_cols.append('location_raw')
    df.drop(drop_cols, axis=1, inplace=True)

    # Drop empty stop_outcome and county_name rows
    print('Dropping empty stop_outcome and county_name rows...')
    # df.dropna(subset=['stop_outcome', 'county_name'], axis=0, inplace=True)
    df.dropna(subset=['stop_outcome'], axis=0, inplace=True)

    # Remove records with age less than 15
    df = drop_threshold_ages(df, 15)

    # Fill in empty stop_time values with median value
    print('Filling empty stop_time values with median...')
    populated = df[df.stop_time.notnull()]['stop_time'].sort_values()
    median_stop_time = populated.iloc[populated.shape[0] // 2]
    df['stop_time'].fillna(median_stop_time, inplace=True)

    # Fill in empty is_arrested values with False
    df['is_arrested'].fillna(False, inplace=True)

    # Categorize stop_time into time-of-day(morning, afternoon, evening, small hours) and stop_date by season
    df = categorize_dates_times(df)

    # Transform driver_gender to binary
    # print('Transforming driver_gender to binary is_male...')
    # df['is_male'] = df['driver_gender'].apply(lambda x: 1 if x == 'M' else 0)

    if not drop_location_raw:
        print('LabelEncoding location_raw...')
        encoder = LabelEncoder()
        df['location_raw'] = encoder.fit_transform(df['location_raw'].astype(str))

    # Normalize violations and append one-hot encoded violations
    print('Normalizing violations and appending one-hot encoded violations to df...')
    df_violations = normalize_violations(df)
    df = pd.concat([df, df_violations], axis=1)


    # Encode officer_id
    # df['officer_id'] = df['officer_id'].apply(lambda x: 'ofcr_{}'.format(x))

    # Drop columns no longer needed
    print('Dropping no longer needed columns...')
    drop_cols = [
        # 'driver_gender',
        'stop_date',
        'stop_time',
        'violation_raw',
        'violation',
    ]
    df.drop(drop_cols, axis=1, inplace=True)

    # Drop duplicate rows
    print('Dropping duplicate rows...')
    df.drop_duplicates(inplace=True)

    # Convert booleans to 0 and 1
    # df = binarize(df, ['search_conducted', 'contraband_found'])


    # Normalize driver_age
    df = normalize_driver_age(df, ['driver_age_raw'])

    # Categorical variables to one-hot encode or label encode
    cols_to_encode = [
        'county_name',
        # 'day_period',
        'driver_gender',
        # 'driver_race',
        # 'season',
        'stop_duration',
    ]

    # Eithen label-encode or one-hot encode categorical variables
    if encode_labels:
        print('LabelEncoding appropriate columns...')
        for col in cols_to_encode:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype('str'))
    else:
        df = pd.get_dummies(df, columns=cols_to_encode)

    # LabelEncode stop_outcome
    encoder = LabelEncoder()
    df['stop_outcome'] = encoder.fit_transform(df['stop_outcome'])

    # Split dataset into training and testing sets
    split_pct = 0.20
    (train, test) = split_train_test(df, split_pct)

    # Oversample
    if perform_oversampling:
        print('Oversampling rows...')
        train = shuffle(oversample(train), random_state=0)

    # Undersample largest outcome
    # train = undersample(train)

    # Emulate dropout layer
    # dropout_pct = .75
    # train = train[:round(train.shape[0] * dropout_pct)]

    return (train, test)


if __name__ == '__main__':
    stages = [{
        # 'drop_location_raw': True,
        # 'perform_oversampling': False,
        # 'encode_labels': False,
    # }, {
    #     'drop_location_raw': False,
    #     'perform_oversampling': False,
    #     'encode_labels': False,
    # }, {
        'drop_location_raw': False,
        'perform_oversampling': False,
        'encode_labels': True,
    # }, {
    #     'drop_location_raw': False,
    #     'perform_oversampling': True,
    #     'encode_labels': True,
    }]

    for idx, stage in enumerate(stages, 1):
        print('\n\nStage {}:\n\nReading CSV...'.format(idx))
        df_in = pd.read_csv('../data/CT-clean.csv', header=0)
        (train, test) = preprocess(df_in, **stage)

        print('[Row counts] train: {}  test: {}'.format(train.shape[0], test.shape[0]))

        print('\nPickling dataframe to file...')
        # train.to_pickle('../data/stage{}-train.pkl'.format(idx))
        # test.to_pickle('../data/stage{}-test.pkl'.format(idx))
        train.to_pickle('../data/stage6-train.pkl'.format(idx))
        test.to_pickle('../data/stage6-test.pkl'.format(idx))

    print('\nFinished preprocessing.')
