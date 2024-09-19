"""
Module to create classifier models used to predict when and how IMPROVER
generated TAFs will go bust.

Functions:
    main: Main function.
    get_clf: Creates classifier to predict bust labels.
    get_clf_label: Creates classifier for bust labels.
    get_xy: Concatenates dataframes and separates into X/y.
    pickle_unpickle: Pickles and unpickles data.
    plot_confusion_matrix: Plots confusion matrix.
    plot_model_scores: Plots classifier evaluation metrics.
    plot_model_times: Plots classifier processing times.
    split_data: Splits data into training and testing sets.

Written by Andre Lanyon
"""
import os
import pickle
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import common.configs as co
import ml.bust_adjust as ba

# Import environment variables
OUTPUT_DIR = os.environ['OUTPUT_DIR']

# Other constants
SCORES = {'F1 score': f1_score,
          'Recall': recall_score, 'Precision': precision_score}
CLASSIFIERS = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
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
    m_scores = {
        'vis': {'Classifier': [], 'Evaluation Metric': [], 'Score': []},
        'cld': {'Classifier': [], 'Evaluation Metric': [], 'Score': []}
    }
    m_times = {'vis': {'Classifier': [], 'Time': []},
               'cld': {'Classifier': [], 'Time': []}}

    # Create classifiers for each airport
    for icao in co.ML_ICAOS:
    # for icao in ['EGAA']:

        # Unpickle data if available
        if os.path.exists(f'{OUTPUT_DIR}/pickles/clfs_data_{icao}'):
            print(f'Unpickling data for {icao}')
            with open(f'{OUTPUT_DIR}/pickles/clfs_data_{icao}',
                      'rb') as file_object:
                unpickler = pickle.Unpickler(file_object)
                test_data, clf_models, m_scores, m_times = unpickler.load()

        # Otherwise, create classifiers
        else:

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

            # Split data into train/test
            X_train, all_y_train, X_test, all_y_test, test_data = split_data(
                taf_data
            )

            # To collect classifier models in
            clf_models = {}

            # Create classifier for each bust type
            for bust_type in co.BUST_COLS:

                # Loop through all classifiers to test
                for model_name, model in CLASSIFIERS.items():

                    # Get models to predict bust/no bust and bust type
                    clf_models, m_scores, m_times = get_clf(
                        clf_models, X_train, all_y_train, X_test, all_y_test,
                        plot_dir, bust_type, model_name, model, m_scores,
                        m_times
                    )

            # Pickle/unpickle files including bust label classifier models
            bl_data = [test_data, clf_models, m_scores, m_times]
            bl_fname = f'{OUTPUT_DIR}/pickles/clfs_data_{icao}'
            test_data, clf_models, m_scores, m_times = pickle_unpickle(
                bl_data, bl_fname
            )

        # Use classifiers to predict bust labels and re-write TAFs
        for (tdf, site_df) in test_data:
            ba.update_taf(tdf, site_df, clf_models, 'xgboost')

    # Pickle/unpickle classifier model scores
    fname = f'{OUTPUT_DIR}/pickles/model_scores_times'
    m_data = [m_scores, m_times]
    m_scores, m_times = pickle_unpickle(m_data, fname)

    # Make some plots comparing classifiers
    for param in ['vis', 'cld']:
        plot_model_scores(m_scores[param], param)
        plot_model_times(m_times[param], param)

    print('Finished')


def get_clf(clf_models, X_train, all_y_train, X_test, all_y_test, plot_dir,
            bust_type, model_name, model, m_scores, m_times):
    """
    Creates classifier to predict busts.

    Args:
        clf_models (dict): Classifier dictionary
        X_train (pandas.DataFrame): Training input data
        y_train (pandas.Series): Training target data
        X_test (pandas.DataFrame): Testing input data
        y_test (pandas.Series): Testing target data
        all_y_test (pandas.DataFrame): Testing target data including bust text
        plot_dir (str): Directory to save plots to
        bust_type (str): Bust type
        model_name (str): Classifier name
        model (sklearn classifier): Classifier
        m_scores (dict): Model scores
        m_times (dict): Model processing times
    Returns:
        clf_models (dict): Classifier dictionary
        m_scores (dict): Updated model scores
        m_times (dict): Updated model processing times
    """
    # Create columns of class integers based on bust labels
    labels = list(pd.unique(all_y_train[bust_type]))
    if 'no_bust' in labels:
        labels.remove('no_bust')
    lab_dict = dict({'no_bust': 0},
                    **{label: ind + 1 for ind, label in enumerate(labels)})

    # Get class integers based on lab_dict
    y_train = all_y_train[bust_type].map(lab_dict)
    y_test = all_y_test[bust_type].map(lab_dict)

    # Ensure k_neighbors is smaller than the smallest class
    class_counts = y_train.value_counts()
    k_neighbors = min([min(class_counts) - 1, 5])

    # Oversample minority class using SMOTE and clean using Tomek links
    smt = SMOTETomek(random_state=8, smote=SMOTE(k_neighbors=k_neighbors))
    X_train, y_train = smt.fit_resample(X_train, y_train)

    # Define classifier, train and make predictions, timing it
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'\nElapsed time for {model_name}: {elapsed_time:.2f} seconds')

    # Shortened/simplified names for fnames, etc
    mf_name = model_name.replace(' ', '_').lower()
    bf_name = bust_type.split('_')[0]

    # Get required scores
    for score_name, score_func in SCORES.items():
        m_score = score_func(y_test, y_pred, average='macro')
        print(score_name, m_score)
        m_scores[bf_name]['Classifier'].append(model_name)
        m_scores[bf_name]['Evaluation Metric'].append(score_name)
        m_scores[bf_name]['Score'].append(m_score)

    # Plot confusion matrix
    fname = f'{plot_dir}/cm_{bf_name}_{mf_name}.png'
    plot_confusion_matrix(lab_dict, y_test, y_pred, fname)

    # Add classifier to dictionary
    bf_name = bust_type.split('_')[0]
    clf_models[f'{bf_name}_{mf_name}'] = model
    clf_models[f'{bf_name}_{mf_name}_label_dict'] = lab_dict

    return clf_models, m_scores, m_times


def get_xy(t_data):
    """
    Concatenates dataframe and separates into X/y.

    Args:
        t_data (list): List of various data
    Returns:
        X (pandas.DataFrame): X data
        all_y (pandas.DataFrame): All y data
    """
    # Concatenate dataframes in list
    tdf = pd.concat([tlist[0] for tlist in t_data], ignore_index=True)

    # Also need the whole dataset for any bust models
    X = tdf[co.PARAM_COLS]
    X = X.apply(pd.to_numeric)
    all_y = tdf[co.BUST_COLS]

    return X, all_y


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

    # Add labels (make nice if possible)
    ax.set_xlabel('Predicted label', fontsize=18)
    ax.set_ylabel('True label', fontsize=18)
    try:
        ax.set_xticklabels(nice_labels)
        ax.set_yticklabels(nice_labels)
    except:
        pass

    # Save plot
    plt.tight_layout()
    fig.savefig(fname)
    plt.close()


def plot_model_scores(m_scores, param):
    """
    Plots classifier evaluation metrics.

    Args:
        m_scores (dict): Model scores
        param (str): Parameter
    Returns:
        None
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 6))

    # Create box plot separating by day period if necessary
    sns.boxplot(data=m_scores, x='Evaluation Metric', y='Score',
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
    fig.savefig(f'{OUTPUT_DIR}/ml_plots/classifier_scores_{param}.png',
                bbox_inches = "tight")
    plt.close()


def plot_model_times(m_times, param):
    """
    Plots classifier processing times.

    Args:
        m_times (dict): Model times
        param (str): Parameter
    Returns:
        None
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create box plot separating by day period if necessary
    sns.boxplot(data=m_times, x='Classifier', y='Time', showfliers=False)

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
    fig.savefig(f'{OUTPUT_DIR}/ml_plots/classifier_times_{param}.png',
                bbox_inches = "tight")
    plt.close()


def split_data(icao_data):
    """
    Splits data into training and testing sets.

    Args:
        icao_data (list): TAF data for airport
    Returns:
        X_train (pandas.DataFrame): Training input data
        all_y_train (pandas.DataFrame): Training target data
        X_test (pandas.DataFrame): Testing input data
        all_y_test (pandas.DataFrame): Testing target data
        test_data (list): All testing data
    """
    # Split into training, validating and testing datasets
    train_data, test_data = train_test_split(icao_data, test_size=0.2)

    # Concatenate dataframes in each dataset
    X_train, all_y_train = get_xy(train_data)
    X_test, all_y_test = get_xy(test_data)

    return X_train, all_y_train, X_test, all_y_test, test_data


if __name__ == "__main__":

    # Run main function and time it
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Finished in {elapsed_time:.2f} minutes")
