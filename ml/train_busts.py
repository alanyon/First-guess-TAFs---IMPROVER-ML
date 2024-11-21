"""
Module to create classifier models used to predict when and how IMPROVER
generated TAFs will go bust.

Functions:
    main: Main function.
    get_clf: Creates classifier to predict bust labels.
    get_xy: Concatenates dataframes and separates into X/y.
    pickle_unpickle: Pickles and unpickles data.
    plot_confusion_matrix: Plots confusion matrix.
    plot_model_scores: Plots classifier evaluation metrics.
    plot_model_times: Plots classifier processing times.
    split_data: Splits data into training and testing sets.

Written by Andre Lanyon
"""
import os
import numpy as np
from numpy import sort
import pickle
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
import optuna
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize, dummy_minimize, BayesSearchCV
from skopt.plots import plot_convergence
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import warnings

import common.configs as co
import ml.bust_adjust as ba

# Import environment variables
OUTPUT_DIR = os.environ['OUTPUT_DIR']
FAKE_DATE = os.environ['FAKE_DATE']

# Other constants
SCORES = {'F1 score': f1_score, 'Recall': recall_score, 
          'Precision': precision_score}
CLASSIFIERS = ['XGBoost', 'Random Forest']

# Seaborn settings
sns.set_style('darkgrid')

# Suppress FutureWarnings
warnings.filterwarnings("ignore")


def main():
    """
    Main function to build classifiers to predict bust labels.
    """
    # Turn off pandas 'chained' warning
    pd.options.mode.chained_assignment = None

    # Get icao from date icao dictionary
    icao = co.DATE_ICAOS[FAKE_DATE]

    print('icao', icao)

    # To collect best features
    best_features = {}

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
    icao_best_features = {}
    for bust_type in co.BUST_COLS:

        # Loop through all classifiers to test
        for model_name in CLASSIFIERS:

            # Get models to predict bust/no bust and bust type
            clf_models = get_clf(
                clf_models, X_train, all_y_train, X_test, all_y_test,
                plot_dir, bust_type, model_name, get_features=True, 
                optimise=True, compare_models=False
            )

    bl_data = [test_data, clf_models]
    bl_fname = f'{OUTPUT_DIR}/pickles/clfs_data_{icao}'

    # Pickle files including bust label classifier models
    with open(bl_fname, 'wb') as f_object:
        pickle.dump(bl_data, f_object)


def balance_data(X_train, y_train):
    """
    Balances data using SMOTE and Tomek links.

    Args:
        X_train (pandas.DataFrame): Training input data
        y_train (pandas.Series): Training target data
    Returns:
        X_train (pandas.DataFrame): Balanced training input data
        y_train (pandas.Series): Balanced training target data
    """
    # Get minimum class count and class with minimum count
    class_counts = y_train.value_counts()
    min_class_count = min(class_counts)

    # Use simple method on classes with 3 samples or less
    if min_class_count <= 3:
        oversampler = RandomOverSampler(sampling_strategy='minority')
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    # Ensure k_neighbors is smaller than the smallest class
    class_counts = y_train.value_counts()
    k_neighbors = min([min(class_counts) - 1, 5])

    # Oversample minority class using SMOTE and clean using Tomek links
    smt = SMOTETomek(random_state=8, smote=SMOTE(k_neighbors=k_neighbors))
    X_train, y_train = smt.fit_resample(X_train, y_train)

    return X_train, y_train


def best_features_plot(best_features, bust_type):
    """
    Plots best features for bust type.

    Args:
        best_features (dict): Best features
        bust_type (str): Bust type
    Returns:
        None
    """
    # Get all features chosen for bust type
    b_type_features = [list(best_features[icao][bust_type]) 
                       for icao in best_features]

    # Flatten list of lists
    all_features = [feature for sublist in b_type_features 
                    for feature in sublist]

    # Count occurrences of each feature
    feature_counts = {feature: all_features.count(feature) 
                      for feature in all_features}

    # Calculate percentage of occasions each feature appears
    total_occasions = len(b_type_features)
    feature_occasions = {feature: all_features.count(feature) 
                         / total_occasions * 100 
                         for feature in all_features}

    # Sort dictionary by value
    feature_occasions = dict(sorted(feature_occasions.items(), 
                                    key=lambda item: item[1], reverse=True))

    # Rearrange for seaborn plot
    perc_features = {'Feature': list(feature_occasions.keys()),
                     'Percentage': list(feature_occasions.values())}

    # Add in features that weren't chosen
    for feature in co.PARAM_COLS:
        if feature not in perc_features['Feature']:
            perc_features['Feature'].append(feature)
            perc_features['Percentage'].append(0)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot feature importances
    sns.barplot(data=perc_features, x='Percentage', y='Feature', ax=ax)

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/ml_plots/best_features_{bust_type}.png')
    plt.close()


def get_best_features(default, X_train, y_train, feat_fname, model_name):
    """
    Gets best features for classifier.

    Args:
        default (sklearn classifier): Default classifier
        X_train (pandas.DataFrame): Training input data
        y_train (pandas.Series): Training target data
        feat_fname (str): File path for saving plot
        model_name (str): Classifier name
    Returns:
        best_features (list): Best features
    """
    # Indices of sorted feature importances
    sorted_idx = default.feature_importances_.argsort()

    # Get sorted features
    sorted_features = X_train.columns[sorted_idx]

    # Make plot of feature importances
    fig, ax = plt.subplots()
    ax.barh(sorted_features, default.feature_importances_[sorted_idx])
    ax.set_xlabel("Feature Importance")
    plt.tight_layout()
    fig.savefig(feat_fname)
    plt.close()

    # Define default variables to update if necessary
    best_features = sorted_features.copy()

    # Get thresholds for feature selection and loop through them
    thresholds = sort(default.feature_importances_)

    # Loop through thresholds
    for ind, thresh in enumerate(thresholds):

        # select features using threshold
        selection = SelectFromModel(default, threshold=thresh, prefit=True)

        # Get subset of X_train using features
        select_X_train = selection.transform(X_train)

        # Use Stratified K-Fold for cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Loop through folds and get mean precision score
        scores = []
        for tr_ind, te_ind in skf.split(select_X_train, y_train):

            # Split the data
            X_train_fold = select_X_train[tr_ind] 
            X_test_fold =  select_X_train[te_ind]
            y_train_fold = y_train[tr_ind]
            y_test_fold = y_train[te_ind]

            # Balance data, fit, make predictions, calc precision
            X_tr_b, y_tr_b = balance_data(X_train_fold, y_train_fold)

            # train model
            if model_name == 'XGBoost':
                selection_model = XGBClassifier(random_state=42)
            elif model_name == 'Random Forest':
                selection_model = RandomForestClassifier(random_state=42)
            selection_model.fit(X_tr_b, y_tr_b)

            # Make predictions and score
            y_pred = selection_model.predict(X_test_fold)
            prec = precision_score(y_test_fold, y_pred, average='macro',
                                   zero_division=0)
            scores.append(prec)

        # Get mean score
        sel_prec = np.mean(scores)

        # Print score
        print(f'Thresh={thresh}, n={select_X_train.shape[1]} '
              f'Precision={sel_prec}')

        # Update best features score increased
        if ind == 0:
            best_precision = sel_prec
        elif sel_prec > best_precision:
            best_features = sorted_features[-select_X_train.shape[1]:]
            best_precision = sel_prec

    print(f'Best features: {best_features}')
    print(f'Best precision: {best_precision}')

    return best_features


def get_clf(clf_models, X_train, all_y_train, X_test, all_y_test, plot_dir,
            bust_type, model_name, get_features=False, optimise=False, 
            compare_models=False):
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
        m_scores (dict): Model scores
        m_times (dict): Model processing times
    Returns:
        clf_models (dict): Classifier dictionary
        m_scores (dict): Updated model scores
        m_times (dict): Updated model processing times
    """
    # Shortened/simplified names for fnames, etc
    mf_name = model_name.replace(' ', '_').lower()
    bf_name = bust_type.split('_')[0]

    # Get label dictionary
    lab_dict = get_label_dict(all_y_train[bust_type])

    # Get class integers based on lab_dict
    y_train = all_y_train[bust_type].map(lab_dict)
    y_test = all_y_test[bust_type].map(lab_dict)

    # Oversample minority class using SMOTE and clean using Tomek links
    X_train_b, y_train_b = balance_data(X_train, y_train)

    # Define default model  
    if model_name == 'XGBoost':
        default = XGBClassifier(random_state=42)
    elif model_name == 'Random Forest':
        default = RandomForestClassifier(random_state=42)
    elif model_name == 'Decision Tree':
        default = DecisionTreeClassifier(random_state=42)
    elif model_name == 'Gradient Boosting':
        default = GradientBoostingClassifier(random_state=42)

    # Train model and get time      
    start = time.time()
    warnings.filterwarnings("ignore")
    default.fit(X_train_b, y_train_b)
    end = time.time()
    elapsed = end - start
    print(f'\nElapsed time for {model_name}: {elapsed:.2f} seconds')

    # Get default precision score
    default_y_pred = default.predict(X_test)
    default_precision = precision_score(y_test, default_y_pred, 
                                        average='macro', zero_division=0)
    # f1 = f1_score(y_test, default_y_pred, average='macro')
    print('Score before optimisation:', default_precision)

    # Plot confusion matrix
    cm_fname = f'{plot_dir}/cm_{bf_name}_{mf_name}_default.png'
    plot_confusion_matrix(lab_dict, y_test, default_y_pred, cm_fname)

    # To add model comparison stats to if required
    m_scores = {'Classifier': model_name, 'Param': bf_name}

    # Collect stats for comparing models if required
    if compare_models:

        # Add model time to dictionary
        m_scores['Time'] = elapsed

        # Get required scores
        for score_name, score_func in SCORES.items():
            m_score = score_func(y_test, default_y_pred, average='macro')
            m_score_minorities = score_func(y_test, default_y_pred, 
                                            labels=[1, 2], average='micro')
            print(score_name, m_score, m_score_minorities)
            m_scores[score_name] = m_score
            m_scores[f'{score_name} (minorities)'] = m_score_minorities

    # Plot feature importance and select best features
    if get_features:
        feat_fname = f'{plot_dir}/feature_importance_{bf_name}_{mf_name}.png'
        best_features = get_best_features(
            default, X_train, y_train, feat_fname, model_name
        )
        X_train, X_train_b = X_train[best_features], X_train_b[best_features]
        X_test = X_test[best_features]

    else:
        best_features = X_train.columns

    # Optimise hyperparameters
    if optimise:
        fname = f'{plot_dir}/convergence_{bf_name}_{mf_name}.png'
        opt_model = optimise_hypers(X_train, y_train, X_train_b, y_train_b,
                                    model_name)
    else:
        if model_name == 'XGBoost':
            opt_model = XGBClassifier(random_state=42)
        elif model_name == 'Random Forest':
            opt_model = RandomForestClassifier(random_state=42)
        elif model_name == 'Decision Tree':
            opt_model = DecisionTreeClassifier(random_state=42)
        elif model_name == 'Gradient Boosting':
            opt_model = GradientBoostingClassifier(random_state=42)
        opt_model.fit(X_train_b, y_train_b)

    # Define optimal classifier and print score
    y_pred_opt = opt_model.predict(X_test)
    opt_precision =  precision_score(y_test, y_pred_opt, average='macro',
                                     zero_division=0)
    # f1 = f1_score(y_test, y_pred_opt, average='macro')
    print('Score after optimisation:', opt_precision)

    # Print which model got the better score
    if opt_precision > default_precision:
        print('Optimised model better')
    else:
        print('Default model better')

    # Plot confusion matrix
    fname = f'{plot_dir}/cm_{bf_name}_{mf_name}_opt.png'
    plot_confusion_matrix(lab_dict, y_test, y_pred_opt, fname)

    # Add classifier to dictionary
    bf_name = bust_type.split('_')[0]
    clf_models[f'{bf_name}_{mf_name}'] = opt_model
    clf_models[f'{bf_name}_{mf_name}_label_dict'] = lab_dict
    clf_models[f'{bf_name}_{mf_name}_features'] = best_features
    clf_models[f'{bf_name}_{mf_name}_model_scores'] = m_scores

    return clf_models


def get_label_dict(bust_labels):

    labels = list(pd.unique(bust_labels))
    if 'no_bust' in labels:
        labels.remove('no_bust')
    lab_dict = dict({'no_bust': 0},
                    **{label: ind + 1 for ind, label in enumerate(labels)})

    return lab_dict


def get_prec(X_train, y_train, mod):

    # Use Stratified K-Fold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for tr_ind, te_ind in skf.split(X_train, y_train):

        # Split the data
        X_train_fold, X_test_fold = X_train.iloc[tr_ind], X_train.iloc[te_ind]
        y_train_fold, y_test_fold = y_train.iloc[tr_ind], y_train.iloc[te_ind]

        # Balance data, fit, make predictions, calc precision
        X_tr_b, y_tr_b = balance_data(X_train_fold, y_train_fold)

        # Fit model
        mod.fit(X_tr_b, y_tr_b)

        # Make predictions and score
        y_pred = mod.predict(X_test_fold)
        prec = precision_score(y_test_fold, y_pred, average='macro',
                               zero_division=0)
        scores.append(prec)

    mean_score = np.mean(scores)

    return mean_score


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
    score = precision_score(y, y_pred, labels=[1, 2], average='micro',
                            zero_division=0)

    return score, y_pred


def oob_precision(y_true, y_pred):

    return precision_score(y_true, y_pred, labels=[1, 2], average='micro',
                           zero_division=0)


def optimise_hypers(X_train, y_train, X_train_b, y_train_b, model_name):
    """
    Optimises hyperparameters for classifier.

    Args:
        X_train (pandas.DataFrame): Training input data
        y_train (pandas.Series): Training target data
        model_name (str): Classifier name
    Returns:
        model (sklearn classifier): Optimised classifier
    """
    # For XGBoost
    if model_name == 'XGBoost':

        # Define objective function
        def objective(trial):

            # Define search space
            param = {
                'random_state': 42,
                'verbosity': 0,
                'n_estimators': trial.suggest_int("n_estimators", 100, 500,
                                                  step=100),
                'max_depth': trial.suggest_categorical('max_depth', [None] + 
                                                       list(range(2, 11))),
                'learning_rate': trial.suggest_float('learning_rate', 0.1, 
                                                     0.5),
                'min_child_weight': trial.suggest_int("min_child_weight", 1, 
                                                      6), 
                'n_jobs': -1
            }

            # Define classifier
            mod = XGBClassifier(**param)

            # Use cross validation to get mean macro precision score
            mean_prec_score = get_prec(X_train, y_train, mod)

            return mean_prec_score

        # Define default model and get precision score
        default_mod = XGBClassifier(random_state=42, verbosity=0)
        default_score = get_prec(X_train, y_train, default_mod)

        # Run optimisation
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        trial = study.best_trial

        # Choose best model
        if trial.value > default_score:
            best_clf = XGBClassifier(**trial.params)
        else:
            best_clf = default_mod

    # For Random Forest
    elif model_name == 'Random Forest':

        def objective(trial):

            param = {
                'random_state': 42,
                'n_estimators': trial.suggest_int("n_estimators", 100, 500,
                                                  step=100),
                'max_depth': trial.suggest_categorical("max_depth", [None] + 
                                                       list(range(2, 11))),
                'min_samples_split': trial.suggest_int("min_samples_split", 2, 
                                                       20, step=2),
                'max_features': trial.suggest_int("max_features", 3, 15),
                'n_jobs': -1
            }

            # Define classifier
            mod = RandomForestClassifier(**param)

            # Use cross validation to get mean macro precision score
            mean_prec_score = get_prec(X_train, y_train, mod)

            return mean_prec_score

        # Define default model and get precision score
        default_mod = RandomForestClassifier(random_state=42)
        default_score = get_prec(X_train, y_train, default_mod)

        # Run optimisation                         
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        trial = study.best_trial
        
        # Choose best model
        if trial.value > default_score:
            best_clf = RandomForestClassifier(**trial.params)
        else:
            best_clf = default_mod

    # Fit optimised model on balanced data
    best_clf.fit(X_train_b, y_train_b)

    return best_clf


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
    fname = f'{OUTPUT_DIR}/ml_plots/model_scores/classifier_scores_{param}.png'
    fig.savefig(fname, bbox_inches = "tight")
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
    fname = f'{OUTPUT_DIR}/ml_plots/model_scores/classifier_times_{param}.png'
    fig.savefig(fname, bbox_inches = "tight")
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
