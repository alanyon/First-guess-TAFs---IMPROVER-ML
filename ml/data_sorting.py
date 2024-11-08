import numpy as np
import pandas as pd

import common.configs as co


def calc_bust(row):
    """
    For creating bust/no bust label column based on all bust labels.

    :param row: Row of dataframe
    :type row: pandas.Series

    :return: Bust/no bust string
    :rtype: str
    """
    if all(row[col] == 'no_bust' for col in co.BUST_COLS):
        return 'no_bust'
    return 'bust'


def get_ml_df(site_df, bust_labels=None):
    """
    Organises data into dataframe for use with machine learning
    algorithms.

    :param det: Deterministic BestData dataframe
    :type det: pandas.DataFrame
    :param perc: Percentiles BestData dataframe
    :type perc: pandas.DataFrame
    :param bust_labels: List of bust label dictionaries
    :type bust_labels: list

    :return: Dataframe with required data
    :rtype: pandas.DataFrame
    """
    # Rearrange site_df
    ml_df = site_df.pivot(index='time', columns='percentile')

    # Relabel columns using percentile value
    ml_df.columns = [f'{col}_{percentile}' for col, percentile in ml_df.columns]

    # Convert 'time' from index to column
    ml_df = ml_df.reset_index()

    # Add bust labels if required
    if bust_labels:
        wind_labels, dir_labels, vis_labels, cld_labels = bust_labels
        ml_df['wind_bust_label'] = ml_df['time'].map(wind_labels)
        ml_df['dir_bust_label'] = ml_df['time'].map(dir_labels)
        ml_df['vis_bust_label'] = ml_df['time'].map(vis_labels)
        ml_df['cld_bust_label'] = ml_df['time'].map(cld_labels)

    # Add columns based on date/time
    ml_df['year'] = ml_df.apply(lambda x: x['time'].year, axis=1)
    ml_df['month'] = ml_df.apply(lambda x: x['time'].month, axis=1)
    ml_df['day'] = ml_df.apply(lambda x: x['time'].day, axis=1)
    ml_df['hour'] = ml_df.apply(lambda x: x['time'].hour, axis=1)

    # Create lead time column
    t_0 = ml_df['time'].values[0]
    ml_df['lead'] = ml_df.apply(lambda x: (x['time'] - t_0).total_seconds() / 3600, axis=1)

    # Drop time column
    ml_df.drop('time', axis=1, inplace=True)

    # Make bust_labels the last columns if required
    if bust_labels:
        cols = ml_df.columns.tolist()
        for bust_label in co.BUST_COLS:
            cols.append(cols.pop(cols.index(bust_label)))
        ml_df = ml_df[cols]

    # Change NaNs to no_bust (occurs when airport opens late or similar)
    for col in ml_df.columns:
        if 'bust' in col:
            ml_df[col] = ml_df[col].fillna('no_bust')

    # Add a column to incorporate all busts if required
    if bust_labels:
        ml_df['any_bust'] = ml_df.apply(calc_bust, axis=1)

    return ml_df


