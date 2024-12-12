"""
Module to compare model performance across different classifiers.

Functions:

    main: Main function.
    plot_model_scores: Plots classifier performance metrics.
    plot_model_times: Plots classifier processing times.

Written by: Andre Lanyon
"""
import time
import os
import copy
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_score, recall_score
import common.configs as co
import ml.train_busts as tr
import ml.test_scores as ts

# Define constants
OUTPUT_DIR = os.environ['OUTPUT_DIR']
PARAMS = {'vis': 'Visibility', 'cld': 'Cloud'}
MODELS = {'xgboost': 'XGBoost', 'random_forest': 'Random Forest',
          'gradient_boosting': 'Gradient Boosting', 
          'decision_tree': 'Decision Tree'}
EVALS = ['F1 score', 'F1 score (minorities)', 'Precision', 
         'Precision (minorities)', 'Recall', 'Recall (minorities)']
NUM_ICAOS = len(co.ML_ICAOS)


def main():
    """
    Main function.
    """
    # To collect classifier scores and processing times
    m_scores = {'Classifier': [], 'Performance Metric': [], 'Score': [],}
    m_times = {'Classifier': [], 'Time': []}

    # Get dictionary mapping ICAOs to airport names
    icao_dict = ts.get_icao_dict()

    # loop through all ICAOs
    for icao in co.ML_ICAOS:
        
        # Unpickle data
        with open(f'{OUTPUT_DIR}/pickles/clfs_data_models_{icao}', 
                  'rb') as f_object:
            unpickler = pickle.Unpickler(f_object)
            test_data, clf_models = unpickler.load()

        # Get model scores and times
        for param, model in itertools.product(PARAMS, MODELS):
            model_scores = clf_models[f'{param}_{model}_model_scores']

            # Update scores
            for eval_met in EVALS:

                # Ignore minority scores
                if 'minorities' in eval_met:
                    continue

                # Add details to dictionary
                m_scores['Classifier'].append(MODELS[model])
                m_scores['Performance Metric'].append(eval_met)
                m_scores['Score'].append(model_scores[eval_met])

            # Update times
            m_times['Classifier'].append(MODELS[model])
            m_times['Time'].append(model_scores['Time'])

    # Plot model scores/times
    plot_model_scores(m_scores)
    plot_model_times(m_times)


def plot_model_scores(m_scores):
    """
    Plots classifier performance metrics.

    Args:
        m_scores (dict): Model scores
        param (str): Parameter
    Returns:
        None
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 6))

    # Create box plot separating by day period if necessary
    sns.boxplot(data=m_scores, x='Performance Metric', y='Score',
                hue='Classifier')

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
    ax.set_xlabel('Performance Metric', fontsize=18, weight='bold')

    # Add vertical lines to separate scores
    # ax.axvline(-0.5, color='k', linestyle='-', linewidth=1, alpha=0.3)
    for x_loc in range(len(set(m_scores['Performance Metric']))):
        ax.axvline(x_loc + 0.5, color='white', linestyle='-', linewidth=1)

    # Save and close plot
    plt.tight_layout()
    fname = f'{OUTPUT_DIR}/ml_plots/model_scores/classifier_scores.png'
    fig.savefig(fname, bbox_inches = "tight")
    plt.close()
    

def plot_model_times(m_times):
    """
    Plots classifier processing times.

    Args:
        m_times (dict): Model times
        param (str): Parameter
    Returns:
        None
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create box plot separating by day period if necessary
    sns.boxplot(data=m_times, x='Classifier', y='Time', hue='Classifier')

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
    fname = f'{OUTPUT_DIR}/ml_plots/model_scores/classifier_times.png'
    fig.savefig(fname, bbox_inches = "tight")
    plt.close()


if __name__ == "__main__":

    # Run main function and time it
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Finished in {elapsed_time:.2f} minutes")