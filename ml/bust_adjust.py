"""
Module to predict bust labels and use these to adjust model data before
generating TAFs.

Functions:
    adjust_vis_cld: Changes vis or cloud data based on bust labels.
    dt_calc: For creating column with datetimes.
    get_labels: Converts predicted integer classes to string labels.
    pred_adjust: Predicts busts, adjusts model data using predictions.
    update_taf: Predicts bust labels, adjusts model data, re-writes TAF.

Written by Andre Lanyon
"""
import itertools
import os
import time
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

import common.calculations as ca
import common.configs as co
import generate.generate_taf as ge
import ml.data_sorting as ds

# Import environment variables
OUTPUT_DIR = os.environ['OUTPUT_DIR']
FAKE_DATE = os.environ['FAKE_DATE']

# Turn off pandas 'chained' warning
pd.options.mode.chained_assignment = None


def main():

    # Get icao from date icao dictionary
    icao = co.DATE_ICAOS[FAKE_DATE]

    print('icao', icao)

    # Unpickle classifier models
    with open(f'{OUTPUT_DIR}/pickles/clfs_data_{icao}', 'rb') as file_object:
        unpickler = pickle.Unpickler(file_object)
        test_data, clf_models = unpickler.load()

    print(clf_models)
    exit()

    # Use classifiers to predict bust labels and re-write TAFs
    for (tdf, site_df) in test_data:
        update_taf(tdf, site_df, clf_models, 'XGBoost')
        update_taf(tdf, site_df, clf_models, 'Random Forest')


def adjust_vis_cld(site_df, row, param):
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
    pred_col = f'{param}_pred_labels'

    # For no bust predicted, do not adjust anything
    if row[pred_col] == 'no_bust':
        return site_df

    # Get old TAF categories for each percentile
    old_cats = []
    for perc in [30, 40, 50, 60, 70]:

        # Get df at valid time and percentile
        perc_time_df = site_df.loc[(site_df['percentile'] == perc)
                                    & (site_df['time'] == row['vdt'])]

        # Get old TAF category
        old_cat = perc_time_df[f'{param}_cat'].values[0]

        # Do not adjust any data if any of the old categories are low 
        # (i.e. low cloud/fog)
        if old_cat < 3:
            return site_df

        # Append old category to list
        old_cats.append(old_cat)

    # If not already returned, loop through percentiles again and adjust
    # data based on bust label
    for perc, old_cat in zip([30, 40, 50, 60, 70], old_cats):

        # If increased bust, increase to next category up
        if row[pred_col] == f'{param}_increase':
            new_cat = float(int(old_cat + 1))

        # If decreased bust, decrease to next category down
        elif row[pred_col] == f'{param}_decrease':
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


def dt_calc(row):
    """
    For creating column with datetimes.

    Args:
        row (pandas.Series): Row of dataframe
    Returns:
        (datetime.datetime): Date and time
    """
    return datetime(row['year'], row['month'], row['day'], row['hour'])


def get_labels(X, clf_models, wx_type, c_name):
    """
    Converts predicted integer classes to string labels.

    Args:
        X (pandas.DataFrame): Input data
        clf_models (dict): Dictionary of classifier models
        wx_type (str): Weather parameter
        clf_type (str): Classifier type
    Returns:
        pred_labels (np.ndarray): Array of predicted labels
    """
    # If no classifier available, return all no busts
    if clf_models[f'{wx_type}_{c_name}'] is None:
        pred_labels = np.array(['no_bust'] * len(X))
        return pred_labels

    # Predict label classes (0, 1, etc)
    y_pred = clf_models[f'{wx_type}_{c_name}'].predict(X)

    # # If scores are low, set to no bust
    # label_dict = clf_models[f'{wx_type}_{c_name}_label_dict']
    # lab_dict_inv = {val: key for key, val in label_dict.items()}

    # # Do no use prediction if precision is less than half
    # for label, score in clf_models[f'{wx_type}_{c_name}_scores'].items():
    #     if score <= 0.5:
    #         y_pred[y_pred == label_dict[label]] = 0

    # Convert class integers to labels using label dictionary
    label_dict = clf_models[f'{wx_type}_{c_name}_label_dict']
    lab_dict_inv = {val: key for key, val in label_dict.items()}
    pred_labels = np.vectorize(lab_dict_inv.get)(y_pred)

    return pred_labels


def my_precision(estimator, X, y):
    """
    Creates custom precision score that gives a micro average but
    ignores the 'no bust' class, for use in hyperparameter optimisation.

    Args:
        estimator (sklearn classifier): Classifier
        X (pandas.DataFrame): Input data
        y (pandas.Series): Target data
    Returns:    
        precision_score (float): micro-averaged precision score
    """
    # Make predictions
    y_pred = estimator.predict(X)

    # Calculate micro-averaged precision score, ignoring the 'no bust'
    # class (0)
    return precision_score(y, y_pred, labels=[1, 2], average='micro')


def pred_adjust(site_df, tdf, clf_models, icao, c_name):
    """
    Predict busts and adjustmodel data based on these predictions

    Args:
        site_df (pandas.DataFrame): Site model data
        tdf (pandas.DataFrame): Model data
        clf_models (dict): Classifier models
        icao (str): ICAO airport identifier
        c_name (str): Classifier type
    Returns:
        site_df (pandas.DataFrame): Adjusted site model data
    """
    # Get X columns from dataframe
    X = tdf[co.PARAM_COLS]
    X = X.apply(pd.to_numeric)

    # Create bust predicts dataframe, starting with valid times
    bust_preds = pd.DataFrame({'vdt': tdf.apply(dt_calc, axis=1)})

    # Predict bust types for each weather type
    for wx_type in ['vis', 'cld']:

        # Predict wx type bust labels
        pred_labels = get_labels(X, clf_models, wx_type, c_name)

        # Add labels to bust preds dataframe
        bust_preds[f'{wx_type}_pred_labels'] = pred_labels

    # Loop through each row of bust labels dataframe
    for _, row in bust_preds.iterrows():

        # Change data for each percentile and each parameter
        for param in ['vis', 'cld']:
            site_df = adjust_vis_cld(site_df, row, param)

    return site_df


def update_taf(tdf, site_df, clf_models, clf_type):
    """
    Uses classifier models to predict bust labels, adjusts BestData
    based on these labels, then re-writes TAF.

    Args:
        tdf (pandas.DataFrame): Model data
        site_df (pandas.DataFrame): Site model data
        clf_models (dict): Classifier models
        clf_type (str): Classifier type
    Returns:
        None
    """
    # Get ICAO code
    icao = site_df.attrs['icao']

    # Simplified classifier name
    c_name = clf_type.replace(' ', '_').lower()

    # Generate TAF using old data and write to text file
    if c_name == 'xgboost':
        old_taf = ge.taf_gen(site_df)
        old_txt_file = f'{OUTPUT_DIR}/tafs/{icao}_old.txt'
        with open(old_txt_file, 'a', encoding='utf-8') as o_file:
            o_file.write(old_taf)

    # Adjust data 5 times to allow for up to 5 TAF group adjustments
    for ind in range(5):

        # Adjust model data based on predicted bust labels
        site_df = pred_adjust(site_df, tdf, clf_models, icao, c_name)

        # Update ml dataframe
        tdf = ds.get_ml_df(site_df)

    # Generate new TAF and write to text file
    new_taf = ge.taf_gen(site_df)
    new_txt_file = f'{OUTPUT_DIR}/tafs/{icao}_{c_name}_new.txt'
    with open(new_txt_file, 'a', encoding='utf-8') as n_file:
        n_file.write(new_taf)


if __name__ == "__main__":

    # Run main function and time it
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Finished in {elapsed_time:.2f} minutes")
