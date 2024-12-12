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
from sklearn.metrics import precision_score, recall_score, f1_score
import common.configs as co
import ml.train_busts as tr

# Define constants
OUTPUT_DIR = os.environ['OUTPUT_DIR']
PARAMS = {'vis': 'Visibility', 'cld': 'Cloud'}
MODELS = {'xgboost': 'XGBoost', 'random_forest': 'Random\nForest'}
NUM_ICAOS = len(co.ML_ICAOS)


def main():
    """
    Main function...
    """
    # For collecting feature stats
    scores = {
        'vis': {'Airport': [], 'XGBoost\nRecall': [], 'XGBoost\nPrecision': [], 
                'XGBoost\nF1 Score': [], 'Random\nForest\nRecall': [], 
                'Random\nForest\nPrecision': [], 'Random\nForest\nF1 Score': []},
        'cld': {'Airport': [], 'XGBoost\nRecall': [], 'XGBoost\nPrecision': [], 
                'XGBoost\nF1 Score': [], 'Random\nForest\nRecall': [], 
                'Random\nForest\nPrecision': [], 'Random\nForest\nF1 Score': []}
    }

    # Get dictionary mapping ICAOs to airport names
    icao_dict = get_icao_dict()

    # loop through all ICAOs
    for icao in co.ML_ICAOS:

        # Only get stats from required ICAOs
        if icao not in co.ML_ICAOS:
            continue
        
        # Unpickle data
        with open(f'{OUTPUT_DIR}/pickles/clfs_data_{icao}', 'rb') as f_object:
            unpickler = pickle.Unpickler(f_object)
            test_data, clf_models = unpickler.load()

        # Concatenate dataframes in each dataset
        X_test, all_y_test = tr.get_xy(test_data)

        # Loop through each parameter
        for param in PARAMS:

            # Update scores with airport name
            scores[param]['Airport'].append(icao_dict[icao])

            # Loop through each model
            for model in MODELS:

                # Get model
                clf = clf_models[f'{param}_{model}']

                # Get label dictionary
                lab_dict = clf_models[f'{param}_{model}_label_dict']

                # Get class integers based on lab_dict
                y_test = all_y_test[f'{param}_bust_label'].map(lab_dict)

                # Just use best features
                X_best = X_test[clf_models[f'{param}_{model}_features']]

                # Get predictions
                y_pred = clf.predict(X_best)

                # Get scores
                recall = recall_score(y_test, y_pred, average='macro')
                precision = precision_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')

                # Update scores
                scores[param][f'{MODELS[model]}\nRecall'].append(recall)
                scores[param][f'{MODELS[model]}\nPrecision'].append(precision)
                scores[param][f'{MODELS[model]}\nF1 Score'].append(f1)

    # Pickle scores
    with open(f'{OUTPUT_DIR}/pickles/scores', 'wb') as f_object:
        pickler = pickle.Pickler(f_object)
        pickler.dump(scores)
    # Unpickle scores
    with open(f'{OUTPUT_DIR}/pickles/scores', 'rb') as f_object:
        unpickler = pickle.Unpickler(f_object)
        scores = unpickler.load()

    # Create heatmap table of scores
    vis_scores = score_table(scores, 'vis')
    cld_scores = score_table(scores, 'cld')

    # Create boxplot of scores
    make_boxplot(vis_scores, cld_scores)


def get_icao_dict():
    """
    Creates a dictionary mapping ICAO codes to airport names.

    Args:
        None
    Returns:
        icao_dict (dict): Dictionary mapping ICAO codes to airport names
    """
    # Load in airport info
    airport_info = pd.read_csv('taf_info.csv', header=0)

    # Create dictionary mapping ICAO codes to airport names
    icao_dict = pd.Series(airport_info.airport_name.values,
                          index=airport_info.icao).to_dict()

    return icao_dict


def score_table(scores, param):

    # Get parameter scores and convert to dataframe
    scores_df = pd.DataFrame(scores[param])

    # Set the Airport column as the index so we can display it nicely
    scores_df.set_index('Airport', inplace=True)

    # Create a custom colormap (red to green)
    clrs = [(0.8, 0.2, 0.2), (1, 0, 0), (0.95, 0.95, 0.95), (0.6, 1, 0.6),
            (0, 0.5, 0)]
    custom_cmap = LinearSegmentedColormap.from_list('custom_red_green', clrs)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 12))

    # Create heatmap with color-coded P-values and T-statistics
    sns.heatmap(scores_df, annot=scores_df,  cmap=custom_cmap, center=0.5,
                fmt='.2f', linewidths=0.5, cbar=False, 
                annot_kws={"fontsize":12})

    # Edit axes labels
    labels = ax.get_xticklabels()
    # for label in labels:
    #     # Insert \n in score labels
    #     label.set_text(label.get_text().replace(' F1',
    #                                             '\nF1').replace(' Pr', '\nPr'))
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

    # Remove x ticks but not labels
    ax.tick_params(axis='x', length=0)

    # Format the plot
    ax.set_ylabel('')
    plt.xticks(rotation=90)
    ax.xaxis.tick_top()

    # Save and close figure
    fig.savefig(f'{OUTPUT_DIR}/ml_plots/icao_scores/{param}_scores.png',
                bbox_inches='tight')
    plt.close()

    # Add parameter to scores_df
    scores_df['Parameter'] = PARAMS[param]

    return scores_df


def make_boxplot(vis_scores, cld_scores):

    # Combine visibility and cloud scores
    scores_df = pd.concat([vis_scores, cld_scores])

    # Rearrange for plotting
    box_scores = {'Performance Metric': [], 'Model Type': [], 'Score': []}
    for _, row in scores_df.iterrows():
        box_scores['Performance Metric'].append('F1 Score')
        box_scores['Model Type'].append(f'XGBoost ({row.Parameter})')
        box_scores['Score'].append(row['XGBoost\nF1 Score'])
        box_scores['Performance Metric'].append('F1 Score')
        box_scores['Model Type'].append(f'Random Forest ({row.Parameter})')
        box_scores['Score'].append(row['Random\nForest\nF1 Score'])
        box_scores['Performance Metric'].append('Precision')
        box_scores['Model Type'].append(f'XGBoost ({row.Parameter})')
        box_scores['Score'].append(row['XGBoost\nPrecision'])
        box_scores['Performance Metric'].append('Precision')
        box_scores['Model Type'].append(f'Random Forest ({row.Parameter})')
        box_scores['Score'].append(row['Random\nForest\nPrecision'])
        box_scores['Performance Metric'].append('Recall')
        box_scores['Model Type'].append(f'XGBoost ({row.Parameter})')
        box_scores['Score'].append(row['XGBoost\nRecall'])
        box_scores['Performance Metric'].append('Recall')
        box_scores['Model Type'].append(f'Random Forest ({row.Parameter})')
        box_scores['Score'].append(row['Random\nForest\nRecall'])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create boxplot of scores
    sns.boxplot(data=box_scores, x='Performance Metric', y='Score',
                hue='Model Type', ax=ax)

    # Remove legend title and put outside axis
    handles, labels = ax.get_legend_handles_labels()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.8),
              fontsize=15, title='Model Type',
              title_fontproperties={'size':18, 'weight':'bold'})

    # Set font sizes on axes
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel('Score', fontsize=18, weight='bold')
    ax.set_xlabel('Performance Metric', fontsize=18, weight='bold')

    # Add vertical lines to separate scores
    # ax.axvline(-0.5, color='k', linestyle='-', linewidth=1, alpha=0.3)
    for x_loc in range(len(set(box_scores['Performance Metric']))):
        ax.axvline(x_loc + 0.5, color='white', linestyle='-', linewidth=1)

    # Save and close figure
    fig.savefig(f'{OUTPUT_DIR}/ml_plots/icao_scores/boxplot.png',
                bbox_inches='tight')
    plt.close()
  

if __name__ == "__main__":

    # Run main function and time it
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Finished in {elapsed_time:.2f} minutes")