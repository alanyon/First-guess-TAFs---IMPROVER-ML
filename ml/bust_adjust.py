"""
Module to predict bust labels and use these to adjust model data before
generating TAFs.

Functions:
    adjust_site_df: Adjusts BestData values based on bust labels.
    change_model: Changes model data based on bust labels.
    dt_calc: For creating column with datetimes.
    get_labels: Converts predicted integer classes to string labels.
    update_taf: Predicts bust labels, adjusts BestData, re-writes TAFs.
    write_taf: Generates TAF, writing to text file.

Written by Andre Lanyon
"""
import numpy as np
import os
from datetime import datetime
import pandas as pd

import common.calculations as ca
import common.configs as co
import ml.data_sorting as ds
import generate.generate_taf as ge


# Import environment variables
OUTPUT_DIR = os.environ['OUTPUT_DIR']

# Turn off pandas 'chained' warning
pd.options.mode.chained_assignment = None


def adjust_vis_cld(site_df, row, param, perc):
    """
    Changes vis or cloud data based on bust labels.

    Args:
        site_df (pandas.DataFrame): Site model data
        row (pandas.Series): Row of dataframe
        param (str): Weather parameter
        perc (int): Percentile
    Returns:
        site_df (pandas.DataFrame): Adjusted site model data
    """
    # Get TAF rules for airport
    rules = site_df.attrs['rules']

    # Get bust label column name from parameter
    col = f'{param}_labels'

    # For parameter-related busts
    if row[col] != 'no_bust':

        # Get df at valid time and percentile
        perc_time_df = site_df.loc[(site_df['percentile'] == perc) 
                                    & (site_df['time'] == row['vdt'])]

        # Get old TAF category
        old_cat = perc_time_df[f'{param}_cat'].values[0]

        # If increased bust, increase to next category up
        if row[col] == f'{param}_increase':
            new_cat = float(int(old_cat + 1))

        # If decreased bust, decrease to next category down
        elif row[col] == f'{param}_decrease':
            new_cat = float(int(old_cat - 1))

        # Get new value(s) and update det_df
        if param == 'vis':
            new_vis = ca.get_vis(new_cat, rules)
            for col, val in zip(['vis', 'vis_cat'], [new_vis, new_cat]):
                site_df.loc[(site_df['percentile'] == perc) 
                            & (site_df['time'] == row['vdt']), col] = val
        elif param == 'cld':
            new_cld_3, new_cld_5 = ca.get_cld(new_cat, rules)
            for col, val in zip(['cld_3', 'cld_5', 'cld_cat'], 
                                [new_cld_3, new_cld_5, new_cat]):
                site_df.loc[(site_df['percentile'] == perc) 
                            & (site_df['time'] == row['vdt']), col] = val

    return site_df


def adjust_site_df(site_df, bust_labels, icao):
    """
    Adjusts BestData values based on bust labels.

    Args:
        site_df (pandas.DataFrame): Site model data
        bust_labels (pandas.DataFrame): Dataframe with bust labels
        icao (str): ICAO airport identifier
    Returns:
        site_df (pandas.DataFrame): Adjusted site model data
    """
    # Loop through each row of bust labels dataframe
    for _, row in bust_labels.iterrows():

        # Ignore when TAF is not predicted to go bust
        if row['all_labels'] == 'no_bust':
            continue

        # Change data for each percentile
        for perc in [30, 40, 50, 60, 70]:

            # Change vis and cloud percentile data
            site_df = adjust_vis_cld(site_df, row, 'vis', perc)
            site_df = adjust_vis_cld(site_df, row, 'cld', perc)

            # For wind-related increased busts, increase winds by 5kt -
            # if this is not enough, should be predicted in next 
            # iteration
            if row['wind_labels'] == 'wind_increase':
                site_df.loc[(site_df['percentile'] == perc) & 
                            (site_df['time'] == row['vdt']), 'wind_gust'] += 5

            # For wind-related decreased busts, only decrease means
            # (leaves possibility of TEMPO/PROB of gusts)
            if row['wind_labels'] == 'wind_decrease':

                # Get current mean wind speed
                old_mean = site_df.loc[(site_df['percentile'] == perc) & 
                                       (site_df['time'] == row['vdt']), 
                                       'wind_mean'].values[0]

                # Decrease by 5, ensuring new value positive
                new_mean = max(old_mean - 10, 0)
                site_df.loc[(site_df['percentile'] == perc) & 
                            (site_df['time'] == row['vdt']), 
                            'wind_mean'] = new_mean

    return site_df


def dt_calc(row):
    """
    For creating column with datetimes.

    Args:
        row (pandas.Series): Row of dataframe
    Returns:
        (datetime.datetime): Date and time
    """
    return datetime(row['year'], row['month'], row['day'], row['hour'])


def get_labels(X, clf_models, wx_type):
    """
    Converts predicted integer classes to string labels.

    Args:
        X (pandas.DataFrame): Input data
        clf_models (dict): Dictionary of classifier models
        wx_type (str): Weather parameter
    Returns:
        pred_labels (np.ndarray): Array of predicted labels
    """
    # If no classifier available, return all no busts
    if clf_models[wx_type] is None:
        pred_labels = np.array(['no_bust'] * len(X))
        return pred_labels

    # Predict label classes (0, 1, etc)
    y_pred = clf_models[wx_type].predict(X)

    # Convert class integers to labels using label dictionary
    label_dict = clf_models[f'{wx_type}_label_dict']
    lab_dict_inv = {val: key for key, val in label_dict.items()}
    pred_labels = np.vectorize(lab_dict_inv.get)(y_pred)

    return pred_labels


def pred_adjust(site_df, tdf, clf_models, icao):
    """
    Predict busts and adjustmodel data based on these predictions

    Args:
        site_df (pandas.DataFrame): Site model data
        tdf (pandas.DataFrame): Model data
        clf_models (dict): Classifier models
        icao (str): ICAO airport identifier
    Returns:
        site_df (pandas.DataFrame): Adjusted site model data
    """
    # Get X columns from dataframe
    X = tdf[co.PARAM_COLS]
    X = X.apply(pd.to_numeric)

    # Predict bust/no bust labels
    pred_labels = get_labels(X, clf_models, 'all')

    # Get subset of data where busts are predicted
    tdf['all_pred_labels'] = pred_labels
    just_busts = tdf[tdf['all_pred_labels'] == 'bust']
    X_busts = just_busts[co.PARAM_COLS]
    X_busts = X_busts.apply(pd.to_numeric)

    # If any busts are predicted, predict bust type labels
    if not just_busts.empty:

        # Predict bust types for each weather type
        for wx_type in ['wind', 'vis', 'cld']:

            # Predict wx type bust labels
            pred_labels = get_labels(X_busts, clf_models, wx_type)

            # Add labels to big dataframe
            lab_col = f'{wx_type}_pred_labels'
            just_busts[lab_col] = pred_labels
            tdf[lab_col] = just_busts[lab_col]
            tdf[lab_col].fillna('no_bust', inplace=True)

        # Add datetime column
        tdf['vdt'] = tdf.apply(dt_calc, axis=1)

        # For bust predictions
        bust_labels_pred = tdf[tdf.columns[-5:]]
        bust_labels_pred.columns = ['all_labels', 'wind_labels', 'vis_labels',
                                    'cld_labels', 'vdt']

        # Update data used for TAF with bust label predictions
        site_df = adjust_site_df(site_df, bust_labels_pred, icao)

    return site_df


def update_taf(tdf, site_df, clf_models):
    """
    Uses classifier models to predict bust labels, adjusts BestData
    based on these labels, then re-writes TAF.

    Args:
        tdf (pandas.DataFrame): Model data
        site_df (pandas.DataFrame): Site model data
        clf_models (dict): Classifier models
        icao (str): ICAO airport identifier
    Returns:
        None
    """ 
    # Get ICAO code       
    icao = site_data.attrs['icao']

    # Generate TAF using old data and write to text file
    old_taf = ge.taf_gen(site_df)
    old_txt_file = f'{OUTPUT_DIR}/tafs/{icao}_old.txt'
    with open(old_txt_file, 'a', encoding='utf-8') as o_file:
        o_file.write(old_taf)

    # Adjust data 5 times to allow for up to 5 TAF group adjustments
    for ind in range(5):

        # Adjust model data based on predicted bust labels
        site_df = pred_adjust(site_df, tdf, clf_models, icao)

        # Update ml dataframe
        tdf = ds.get_ml_df(site_df)

    # Generate new TAF and write to text file
    new_taf = ge.taf_gen(site_df)
    new_txt_file = f'{OUTPUT_DIR}/tafs/{icao}_new.txt'
    with open(new_txt_file, 'a', encoding='utf-8') as n_file:
        n_file.write(new_taf)