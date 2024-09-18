"""
Module to create classifier models used to predict when and how IMPROVER
generated TAFs will go bust.

Functions:
    main: Main function.
    get_clf_binary: Creates binary classifier.
    get_clf_label: Creates classifier for bust labels.
    get_xy: Concatenates dataframes.
    get_model: Gets optimal classifier and plots results.
    predict_btypes: Predicts bust types given busts.
    train_test: Splits data into training and testing sets.

Written by Andre Lanyon
"""
import itertools
import os
import pickle
from datetime import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

import ml.bust_adjust as ba
import common.configs as co

# Import environment variables
OUTPUT_DIR = os.environ['OUTPUT_DIR']

SCORES = {'Accuracy': accuracy_score, 'F1 score': f1_score,
          'Recall': recall_score, 'Precision': precision_score}
CLASSIFIERS = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVC': SVC(kernel="rbf", C=0.025, probability=True, random_state=42),    
    }

# Seaborn settings
sns.set_style('darkgrid')


def main():
    """
    Main function to build classifiers to predict bust labels.
    """
    # Turn off pandas 'chained' warning
    pd.options.mode.chained_assignment = None

    # To collect classifier times and scores
    m_scores = {'Classifier': [], 'Evaluation Metric': [], 'Score': []}
    m_times = {'Classifier': [], 'Time': []}

    # Create classifiers for each airport
    # for icao in co.ML_ICAOS:
    for icao in ['EGLL']:

        print(f'Training models for {icao}')

        # Directory to send plots to
        plot_dir = f'{OUTPUT_DIR}/ml_plots/{icao}'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Unpickle data
        i_pickle = f'{OUTPUT_DIR}/pickles/pickle_{icao}'
        with open(i_pickle, 'rb') as file_object:
            unpickler = pickle.Unpickler(file_object)
            taf_data = unpickler.load()

        # Split data into train/test and oversample using SMOTE
        X_trains, all_y_trains, X_tests, all_y_tests, test_datas = split_data(
            taf_data
        )

        # # Try different classifiers on any bust model
        # m_scores, m_times = try_models(X_train, y_train, X_test, y_test, 
        #                                plot_dir, lab_dict, m_scores, m_times)       

        # To collect classifier models in
        clf_models = {}

        # Create classifier for each bust type
        for bust_type in co.BUST_COLS:

            # Get models to predict bust/no bust and bust type
            clf_models = get_clf(clf_models, X_trains, all_y_trains, X_tests, 
                                 all_y_tests, plot_dir, bust_type
            )

    print('Finished')


def get_clf(clf_models, X_trains, all_y_trains, X_tests, all_y_tests, plot_dir, 
            bust_type):
    """
    Creates required binary classifier (bust/no bust).

    Args:  
        clf_models (dict): Classifier dictionary
        X_train (pandas.DataFrame): Training input data
        y_train (pandas.Series): Training target data
        X_test (pandas.DataFrame): Testing input data
        y_test (pandas.Series): Testing target data
        all_y_test (pandas.DataFrame): Testing target data including bust text
        lab_dict (dict): Label dictionary
        plot_dir (str): Directory to save plots to
    Returns:
        clf_models (dict): Classifier dictionary
        test_df (pandas.DataFrame): Testing data
    """
    # Get main X and y data
    X_train = X_trains['all']
    all_y_train = all_y_trains['all']
    X_test = X_tests['all']
    all_y_test = all_y_tests['all']

    # Create columns of class integers based on bust labels
    labels = list(pd.unique(all_y_train[bust_type]))
    if 'no_bust' in labels:
        labels.remove('no_bust')
    lab_dict = dict({'no_bust': 0},
                    **{label: ind + 1 for ind, label in enumerate(labels)})
    binary_lab_dict = {key: 0 if key == 'no_bust' else 1 for key in lab_dict}

    # Train binary (bust/no bust) model and make predictions
    y_train_bin = all_y_train[bust_type].map(binary_lab_dict)
    y_test_bin = all_y_test[bust_type].map(binary_lab_dict)
    new_bin_lab_dict = {'no_bust': 0, 'bust': 1}

    # Oversample minority class using SMOTE
    smo = SMOTE(random_state=8)
    X_train, y_train_bin = smo.fit_resample(X_train, y_train_bin)

    # Define classifier, train and make predictions
    rf_any = RandomForestClassifier()
    rf_any.fit(X_train, y_train_bin)
    y_pred_bin = rf_any.predict(X_test)
    score = precision_score(y_test_bin, y_pred_bin)

    print(f'\nPrecision score for {bust_type}: {score}')

    # Convert predictions back to strings
    lab_dict_inv = {val: key for key, val in new_bin_lab_dict.items()}
    pred_labels_bin = np.vectorize(lab_dict_inv.get)(y_pred_bin)
    
    # Plot confusion matrix
    fname = f'{plot_dir}/cm_{bust_type}_any.png'
    plot_confusion_matrix(new_bin_lab_dict, y_test_bin, y_pred_bin, fname)

    # Get subsets where busts have occurred
    X_train_busts = X_trains[bust_type]
    all_y_train_busts = all_y_trains[bust_type]
    X_test_busts = X_tests[bust_type]
    all_y_test_busts = all_y_tests[bust_type]
 
    # Create columns of class integers based on bust labels
    labels = list(pd.unique(all_y_train_busts[bust_type]))
    lab_dict_busts = {label: ind for ind, label in enumerate(labels)}

    # Train binary bust type model
    y_train_busts = all_y_train_busts[bust_type].map(lab_dict_busts)
    y_test_busts = all_y_test_busts[bust_type].map(lab_dict_busts)

    # Oversample minority class using SMOTE
    smo = SMOTE(random_state=8)
    X_train_busts, y_train_busts = smo.fit_resample(X_train_busts, 
                                                    y_train_busts)

    # Define classifier, train and make predictions
    rf_busts = RandomForestClassifier()
    rf_busts.fit(X_train_busts, y_train_busts)
    y_pred_busts = rf_busts.predict(X_test_busts)
    score = accuracy_score(y_test_busts, y_pred_busts)

    print(f'\nAccuracy score for {bust_type}: {score}')

    # Add classifier and label dictionary to classifier dictionary
    clf_models[bust_type.split('_')[0]] = {'any_clf': rf_any,
                                           'any_lab_dict': new_bin_lab_dict, 
                                           'just_busts_clf': rf_busts,
                                           'just_busts_lab_dict': lab_dict_busts}

    # Plot confusion matrix
    fname = f'{plot_dir}/cm_{bust_type}_busts.png'
    plot_confusion_matrix(lab_dict_busts, y_test_busts, y_pred_busts, fname)

    # Use both models to predict bust types

    # Subset testing data with rows where bust predicted
    test_df = pd.concat([X_test, all_y_test], axis=1)
    test_df['bust_predicts'] = pred_labels_bin
    test_df['bust_class_predicts'] = y_pred_bin
    test_bust_preds = test_df[test_df['bust_predicts'] == 'bust']
    X_test_bust_preds = test_bust_preds[co.PARAM_COLS]
    all_y_test_bust_preds = test_bust_preds[co.BUST_COLS]

    # Predict bust types
    y_pred = rf_busts.predict(X_test_bust_preds)

    # Convert predictions back to strings
    lab_dict_inv = {val: key for key, val in lab_dict_busts.items()}
    pred_labels= np.vectorize(lab_dict_inv.get)(y_pred)

    # Add new columns with label and class predictions
    X_test_bust_preds['lab_preds'] = pred_labels
    test_df['lab_preds'] = X_test_bust_preds['lab_preds']
    test_df['lab_preds'].fillna('no_busts', inplace=True)
    X_test_bust_preds['class_preds'] = y_pred
    test_df['class_preds'] = X_test_bust_preds['class_preds']
    test_df['class_preds'].fillna(0, inplace=True)

    # Get actual and predicted bust labels for all of testing data
    y_test_both = test_df[bust_type].map(lab_dict)
    y_pred_both = test_df['class_preds']

    # Plot confusion matrix
    fname = f'{plot_dir}/cm_{bust_type}_both.png'
    plot_confusion_matrix(lab_dict, y_test_both, y_pred_both, fname)

    return clf_models


def get_xy(t_data):
    """
    Concatenates and edits dataframes.

    Args:
        t_data (list): List of various data
        return_label_dict (bool): Indicator for whether to return 
                                  label dictionary. Defaults to False.
    Returns:
        X (pandas.DataFrame): X data
        all_y (pandas.DataFrame): All y data 
        lab_dict (dict): Label dictionary   
    """
    # Concatenate dataframes in list
    tdf = pd.concat([tlist[0] for tlist in t_data], ignore_index=True)

    # Also need the whole dataset for any bust models
    X = tdf[co.PARAM_COLS]
    X = X.apply(pd.to_numeric)
    all_y = tdf[co.BUST_COLS]

    return X, all_y


def get_model(X_train, y_train, X_test, y_test, clf_str, plot_dir, lab_dict):
    """
    Finds optimal hyperparameters, trains random forest classifier, 
    predicts classes, prints scores and creates plots of results.

    Args:
        X_train (pandas.DataFrame): Training input data
        y_train (pandas.DataFrame): Training target data
        X_test (pandas.DataFrame): Testing input data
        y_test (pandas.DataFrame): Testing target data
        clf_str (str): Classifier type
        plot_dir (str): Directory to save plots to
        lab_dict (dict): Label dictionary
    Returns:
        model (sklearn.ensemble.RandomForestClassifier): Trained random 
                                                         forest
        y_pred (pandas.DataFrame): Predicted classes
        pred_labels (pandas.DataFrame): Predicted labels
    """
    # Define hyperparameters to test
    params = {
        'n_estimators': [50, 100, 200, 400, 800],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'random_state': [3]
    }

    # Different variables for different classifier types
    if clf_str == 'any_bust':
        score = precision_score
        scoring='precision'
        req_scores = SCORES
    else:
        score = accuracy_score
        scoring='accuracy'
        req_scores = {'accuracy': accuracy_score}

    # First, train and test model with default hyperparameters
    default_model = RandomForestClassifier()

    # model = RFECV(default_model)
    # model.fit(X_train, y_train)
    # selected_features = np.array(co.PARAM_COLS)[model.get_support()]
    # y_pred = model.predict(X_test)
    # print(f'\nSelected features: {selected_features}')

    default_model.fit(X_train, y_train)
    y_pred_default = default_model.predict(X_test)
    default_score = score(y_test, y_pred_default)

    # # Get the number of samples in the original datasets
    # num_samples = len(X_train)

    # # Subset data if too large
    # if num_samples > 100:
        
    #     # Set random seed for reproducibility
    #     np.random.seed(42)

    #     # Calculate the number of samples for the 10% subset
    #     subset_size = int(num_samples * 0.1)

    #     # Generate random indices for the subset
    #     subset_indices = np.random.choice(num_samples, size=subset_size, 
    #                                       replace=False)

    #     # Create the random 10% subsets
    #     X_train_subset = X_train.iloc[subset_indices]
    #     y_train_subset = y_train.iloc[subset_indices]

    # # Otherwise, use full dataset
    # else:
    #     X_train_subset = X_train
    #     y_train_subset = y_train

    # # Use RandomizedSearchCV to obtain best hyperparameters
    # random_search = RandomizedSearchCV(RandomForestClassifier(), 
    #                                    param_distributions=params, n_jobs=8,
    #                                    scoring=scoring, random_state=42,
    #                                    n_iter=50)
    # random_search.fit(X_train_subset, y_train_subset)

    # # Train model using optimised hyperparameters
    # best_model = random_search.best_estimator_
    # best_model.fit(X_train, y_train)
    # y_pred_best = best_model.predict(X_test)
    # best_score = score(y_test, y_pred_best)

    # # Print scores
    # print(f'\nResults for {clf_str.replace("_", " ")} classifier\n')
    # print('Default_score', default_score)
    # print('Best_score', best_score)

    # # Check optimised parameters produce better results
    # if best_score > default_score:
    #     print(f'\nBest hypers: {best_model.get_params()}')
    #     model = best_model
    #     y_pred = y_pred_best
    # # Otherwise, use default settings
    # else:
    #     print('\nDefault hypers best')
    #     model = default_model
    #     y_pred = y_pred_default

    ###TESTING
    model = default_model
    y_pred = y_pred_default

    # Print features in order of importance if required
    imp_feats = pd.DataFrame(zip(model.feature_names_in_,
                                 model.feature_importances_),
                             columns=['Variable', 'Importance'])
    imp_feats = imp_feats.sort_values('Importance', ascending=False)
    imp_feats['Variable'] = imp_feats['Variable'].str.lower()
    imp_feats['Variable'] = imp_feats['Variable'].apply(
        lambda x: x.replace('_', ' ')
    )

    # Convert predictions back to strings
    lab_dict_inv = {val: key for key, val in lab_dict.items()}
    pred_labels = np.vectorize(lab_dict_inv.get)(y_pred)

    # Bar plot of importance features
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=imp_feats, x='Importance', y='Variable')
    ax.set_title('Feature Importance', fontsize=20)
    ax.set_xlabel('Feature importance score', fontsize=18)
    ax.set_ylabel('Features', fontsize=18)
    plt.tight_layout()
    fig.savefig(f'{plot_dir}/feat_imp_{clf_str}.png')
    plt.close()

    # Plot confusion matrix
    fname = f'{plot_dir}/cm_{clf_str}.png'
    plot_confusion_matrix(lab_dict, y_test, y_pred, fname)

    return model, y_pred, pred_labels


def pickle_unpickle(p_data, file_path):
    """
    Pickles and unpickles data.

    Args:
        p_data (list): Data to pickle/unpickle
        file_path (str): File path for pickling/unpickling
    Returns:
        p_data (list): Pickled/unpickled data
    """
    with open(file_path, 'wb') as f_object:
        pickle.dump(p_data, f_object)
    with open(file_path, 'rb') as file_object:
        unpickler = pickle.Unpickler(file_object)
        p_data = unpickler.load()

    return p_data


def plot_confusion_matrix(lab_dict, y_test, y_pred, fname):
    """
    Plots confusion matrix.

    Args:
        lab_dict (dict): Label dictionary
        y_test (pandas.Series): Testing target data
        y_pred (pandas.Series): Predicted target data
        fname (str): File path for saving plot
    Returns:
        None
    """
    # Define plot labels
    labels = list(lab_dict.keys())
    nice_labels = [co.NICE_LABELS[label] for label in labels]

    # Get all classes (sometimes not all classes are predicted)
    all_classes = sorted(set(y_test) | set(y_pred))

    # Create figure and axis
    fig, ax = plt.subplots()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=all_classes)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, ax=ax)

    # Add labels
    ax.set_xlabel('Predicted label', fontsize=18)
    ax.set_ylabel('True label', fontsize=18)
    ax.set_xticklabels(nice_labels)
    ax.set_yticklabels(nice_labels)

    # Save plot
    plt.tight_layout()
    fig.savefig(fname)
    plt.close()


def plot_model_scores(m_scores):
    """
    Plots classifier evaluation metrics.

    Args:
        m_scores (dict): Model scores
    Returns:
        None
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 6))

    # Create box plot separating by day period if necessary
    box_plot = sns.boxplot(data=m_scores, x='Evaluation Metric', y='Score', 
                           hue='Classifier', showfliers=False)

    # Remove legend title and put outside axis
    handles, labels = ax.get_legend_handles_labels()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.8),
              fontsize=15, title='Classifier', 
              title_fontproperties={'size':18, 'weight':'bold'})

    # Set font sizes on axes
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel('Score', fontsize=18, weight='bold')
    ax.set_xlabel('Evaluation Metric', fontsize=18, weight='bold')

    # Add vertical lines to separate scores
    # ax.axvline(-0.5, color='k', linestyle='-', linewidth=1, alpha=0.3)
    for x_loc in range(len(set(m_scores['Evaluation Metric']))):
        ax.axvline(x_loc + 0.5, color='white', linestyle='-', linewidth=1)

    # Save and close plot
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/ml_plots/classifier_scores.png', 
                bbox_inches = "tight")
    plt.close()


def plot_model_times(m_times):
    """
    Plots classifier processing times.

    Args:
        m_times (dict): Model times
    Returns:
        None
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create box plot separating by day period if necessary
    box_plot = sns.boxplot(data=m_times, x='Classifier', y='Time', 
                           showfliers=False)

    # Set font sizes on axes
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=18, weight='bold')
    ax.set_xlabel('Classifier', fontsize=18, weight='bold')

    # Add vertical lines to separate scores
    # ax.axvline(-0.5, color='k', linestyle='-', linewidth=1, alpha=0.3)
    for x_loc in range(len(set(m_times['Classifier']))):
        ax.axvline(x_loc + 0.5, color='white', linestyle='-', linewidth=1)

    # Save and close plot
    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/ml_plots/classifier_times.png', 
                bbox_inches = "tight")
    plt.close()


def predict_btypes(testing_data, clf_models, plot_dir):
    """
    Predicts bust type given bust (already predicted) and using
    classifier previously created, then tests on full dataset.

    Args:
        testing_data (pandas.DataFrame): Testing data
        clf_models (dict): Classifier dictionary
        plot_dir (str): Directory to save plots to
    """
    # Get just rows with busts predicted
    just_bust_preds = testing_data[testing_data['bust_predicts'] == 'bust']

    # Define X data
    X = just_bust_preds[co.PARAM_COLS]

    for bust_col in co.BUST_COLS:

       # Get bust type specific classifier
        b_type_clf = clf_models[bust_col.split('_')[0]]

        # Move on if no classifier
        if b_type_clf is None:
            print(f'No classifier for {bust_col}')
            continue

        # Get label dictionary
        lab_dict = clf_models[bust_col.split('_')[0] + '_label_dict']

        # Create columns of class integers based on bust labels and
        # label dictionary
        class_col = bust_col.replace('label', 'class')
        testing_data[class_col] = testing_data[bust_col].map(lab_dict)

        # Predict bust values
        y_pred = b_type_clf.predict(X)

        # Convert predictions back to strings
        lab_dict_inv = {val: key for key, val in lab_dict.items()}
        pred_labels= np.vectorize(lab_dict_inv.get)(y_pred)

        # Add new columns with label and class predictions
        lab_preds = '_'.join([bust_col, 'predictions'])
        just_bust_preds[lab_preds] = pred_labels
        testing_data[lab_preds] = just_bust_preds[lab_preds]
        testing_data[lab_preds].fillna('no_busts', inplace=True)
        class_preds = '_'.join([class_col, 'predictions'])
        just_bust_preds[class_preds] = y_pred
        testing_data[class_preds] = just_bust_preds[class_preds]
        testing_data[class_preds].fillna(0, inplace=True)

        # Get actual and predicted bust labels for all of testing data
        y_test = testing_data[class_col]
        y_pred = testing_data[class_preds]

        # Plot confusion matrix
        wx_type = bust_col.split('_')[0]
        plot_fname = f'{plot_dir}/cm_{wx_type}.png'
        plot_confusion_matrix(lab_dict, y_test, y_pred, plot_fname)


def split_data(icao_data):
    """
    Splits data into training and testing sets and oversamples minority 
    class using SMOTE.

    Args:
        icao_data (list): TAF data for airport
    Returns:
        X_train_res (pandas.DataFrame): Training data
        y_train_res (pandas.Series): Training labels
        X_test (pandas.DataFrame): Testing data
        y_test (pandas.Series): Testing labels
        all_y_test (pandas.DataFrame): All testing labels
        lab_dict (dict): Label dictionary
        testing_data (pandas.DataFrame): Testing data
    """

    # Split into training, validating and testing datasets
    train_data, test_data = train_test_split(icao_data, test_size=0.2)

    # Concatenate dataframes in each dataset
    X_train, all_y_train = get_xy(train_data)
    X_test, all_y_test = get_xy(test_data)

    return X_train, all_y_train, X_test, all_y_test, test_data


def try_models(X_train, y_train, X_test, y_test, plot_dir, lab_dict, m_scores,
               m_times):
    """
    Tries different classifiers on any bust model.

    Args:
        X_train (pandas.DataFrame): Training input data
        y_train (pandas.DataFrame): Training target data
        X_test (pandas.DataFrame): Testing input data
        y_test (pandas.DataFrame): Testing target data
        plot_dir (str): Directory to save plots to
        lab_dict (dict): Label dictionary
        m_scores (dict): Model scores
        m_times (dict): Model processing times
    Returns:
        m_scores (dict): Updated model scores
        m_times (dict): Updated model processing times
    """
    # Loop through all classifiers to test
    for model_name, model in CLASSIFIERS.items():

        # Fit model and predict classes, and time it
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_time = time.time()
        elapsed_time = (end_time - start_time)
        m_times['Classifier'].append(model_name)
        m_times['Time'].append(elapsed_time)
        print(f'\nElapsed time for {model_name}: {elapsed_time:.2f} seconds')

        # Get required scores
        print(f'Scores for {model_name}')
        for score_name, score_func in SCORES.items():
            m_score = score_func(y_test, y_pred)
            print(score_name, m_score)
            m_scores['Classifier'].append(model_name)
            m_scores['Evaluation Metric'].append(score_name)
            m_scores['Score'].append(m_score)

        # Plot confusion matrix
        fname = f'{plot_dir}/cm_{model_name}_any_bust.png'
        plot_confusion_matrix(lab_dict, y_test, y_pred, fname)

    return m_scores, m_times


if __name__ == "__main__":

    # Run main function and time it
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Finished in {elapsed_time:.2f} minutes")