import time
import os
import copy
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import common.configs as co

# Define constants
OUTPUT_DIR = os.environ['OUTPUT_DIR']
PARAMS = {'vis': 'Visibility', 'cld': 'Cloud'}
MODELS = {'xgboost': 'XGBoost', 'random_forest': 'Random Forest'}
NUM_ICAOS = len(co.ML_ICAOS)

# Seaborn settings
sns.set_style('darkgrid')

def main():
    """
    Main function...
    """
    # # For collecting feature stats
    # best_features = {'xgboost_vis': {feat: 0 for feat in co.PARAM_COLS},
    #                  'xgboost_cld': {feat: 0 for feat in co.PARAM_COLS},
    #                  'random_forest_vis': {feat: 0 for feat in co.PARAM_COLS},
    #                  'random_forest_cld': {feat: 0 for feat in co.PARAM_COLS}}

    # # For collecting hyperparameter stats
    # xg_template = {'n_estimators': [], 'max_depth': [], 'learning_rate': [], 
    #                'min_child_weight': []}
    # rf_template = {'n_estimators': [], 'max_depth': [], 
    #                'min_samples_split': [], 'max_features': []}
    # hyperparams = {'xgboost_vis': copy.deepcopy(xg_template),
    #                'xgboost_cld': copy.deepcopy(xg_template),
    #                'random_forest_vis': copy.deepcopy(rf_template),
    #                'random_forest_cld': copy.deepcopy(rf_template)}

    # # loop through all ICAOs
    # for icao in co.ML_ICAOS:
        
    #     # Unpickle data
    #     with open(f'{OUTPUT_DIR}/pickles/clfs_data_{icao}', 'rb') as file_object:
    #         unpickler = pickle.Unpickler(file_object)
    #         test_data, clf_models = unpickler.load()

    #     # Update stats for model and parameter
    #     for param, model in itertools.product(PARAMS, MODELS):

    #         # Update feature stats
    #         for feature in clf_models[f'{param}_{model}_features']:
    #             best_features[f'{model}_{param}'][feature] += 1

    #         # Update hyperparameter stats
    #         for key, val in hyperparams[f'{model}_{param}'].items():
    #             hyperparams[f'{model}_{param}'][key].append(
    #                 clf_models[f'{param}_{model}'].get_params()[key])

    # # Pickle stats
    # with open(f'{OUTPUT_DIR}/pickles/best_features', 'wb') as file_object:
    #     pickler = pickle.Pickler(file_object)
    #     pickler.dump(best_features)
    # with open(f'{OUTPUT_DIR}/pickles/hyperparams', 'wb') as file_object:
    #     pickler = pickle.Pickler(file_object)
    #     pickler.dump(hyperparams)

    # Unpickle stats
    with open(f'{OUTPUT_DIR}/pickles/best_features', 'rb') as file_object:
        unpickler = pickle.Unpickler(file_object)
        best_features = unpickler.load()
    with open(f'{OUTPUT_DIR}/pickles/hyperparams', 'rb') as file_object:
        unpickler = pickle.Unpickler(file_object)
        hyperparams = unpickler.load()

    # Plot stats
    for param in PARAMS:

        # Plot feature stats
        plot_features(best_features, param)

    for model in MODELS:

        # Plot hyperparameter stats
        plot_hyperparams(hyperparams, model)


def plot_features(features, param):

    # Get best features
    xg_features = features[f'xgboost_{param}']
    rf_features = features[f'random_forest_{param}']

    # Calculate percentage of occasions each feature was used
    perc_xg_features = {feat: (count / NUM_ICAOS) * 100 
                        for feat, count in xg_features.items()}
    perc_rf_features = {feat: (count / NUM_ICAOS) * 100
                        for feat, count in rf_features.items()}

    # Sort features by percentage
    sorted_xg_features = dict(sorted(perc_xg_features.items(), 
                                     key=lambda x: x[1], reverse=True))

    # Rearange data for seaborn plot
    sns_features = {'Feature': list(sorted_xg_features.keys()) ,
                    'Percentage': list(sorted_xg_features.values()),
                    'Classifier': ['XGBoost'] * len(sorted_xg_features)}
    sns_features['Feature'] += list(perc_rf_features.keys())
    sns_features['Percentage'] += list(perc_rf_features.values())
    sns_features['Classifier'] += ['Random Forest'] * len(perc_rf_features)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(9, 7))

    # Create bar plot
    sns.barplot(data=sns_features, x='Percentage', y='Feature', 
                hue='Classifier', ax=ax)

    # Add title
    ax.set_title(f'{PARAMS[param]} Model Feature Usage', fontsize=20,
                 weight='bold')

    # Increase font sizes of axes and tick labels
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel('Percentage of Models Using Feature (%)', fontsize=16, 
                  weight='bold') 
    ax.set_ylabel('Feature', fontsize=16, weight='bold')

    # Remove legend title and put outside axis
    handles, labels = ax.get_legend_handles_labels()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.9),
              fontsize=15, title='Classifier',
              title_fontproperties={'size':18, 'weight':'bold'})

    # Save and close plot
    plt.tight_layout()
    fname = f'{OUTPUT_DIR}/ml_plots/feature_plots/{param}_features.png'
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


def plot_hyperparams(hyperparams, model):

    # Get hyperparameters
    vis_hyperparams = hyperparams[f'{model}_vis']
    cld_hyperparams = hyperparams[f'{model}_cld']

    # Define colours
    colours = {'Visibility': 'green', 'Cloud': 'purple'}
    hue_order = ['Visibility', 'Cloud']

    # Create figure and axes
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    # Loop though hyperparameters, colurs and axes
    for ax, (vis_hyper, vis_values), (cld_hyper, cld_values) in zip(
        ax.flatten(), vis_hyperparams.items(), cld_hyperparams.items()):

        # Ensure same hyperparameter, printing warning message if not
        if vis_hyper != cld_hyper:
            print(f'Warning: {vis_hyper} does not match {cld_hyper}')
            exit()

        # Collect into dictionary
        hy_data = {vis_hyper: vis_values, 
                   'Parameter': ['Visibility'] * len(vis_values)}
        hy_data[vis_hyper] += cld_values
        hy_data['Parameter'] += ['Cloud'] * len(cld_values)

        # Convert None values to 'None' in hyper column
        hy_data[vis_hyper] = ['None' if val is None 
                              else val for val in hy_data[vis_hyper]]

        # Convert to dataframe
        hy_data = pd.DataFrame(hy_data)

        # All hyperparameters are categorical except learning rate
        if vis_hyper == 'learning_rate':

            # Create histogram
            sns.histplot(data=hy_data, x=vis_hyper, ax=ax, bins=10, 
                         hue='Parameter', hue_order=hue_order, palette=colours, 
                         alpha=0.5)
            ax.set_title(f'Histogram for {vis_hyper} distribution', 
                         fontsize=20, weight='bold')

        # For other categorical hyperparameters...
        else:

            # Sort rows by hyperparameter
            hy_data = hy_data.sort_values(
                by=vis_hyper, 
                key=lambda col: col.apply(lambda v: (isinstance(v, str), 
                                                     v if isinstance(v, str) 
                                                     else float(v)))
            )
             
            # Convert all values to strings
            hy_data[vis_hyper] = hy_data[vis_hyper].astype(str)

            # Create bar chart with red bars
            sns.countplot(data=hy_data, x=vis_hyper, ax=ax, hue='Parameter',
                          hue_order=hue_order, palette=colours,
                          order=hy_data[vis_hyper].unique())
            ax.set_title(f'Bar chart for {vis_hyper}', fontsize=20,
                         weight='bold')
        
        # Force y-axis to be integers
        ax.yaxis.get_major_locator().set_params(integer=True)

        # Increase font sizes of axes and tick labels
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel(vis_hyper, fontsize=16, weight='bold')
        ax.set_ylabel('Count', fontsize=16, weight='bold')

        # Only show one legend and put outside axes
        if vis_hyper == 'max_depth':
            handles, labels = ax.get_legend_handles_labels()
            box = ax.get_position()
            ax.legend(handles, labels, loc='center left', 
                      bbox_to_anchor=(0.6, 1.25), fontsize=18)
        else:
            ax.get_legend().remove()

    # Figure title
    plt.suptitle(f'{MODELS[model]} Model Hyperparameters', fontsize=20, 
                 weight='bold')

    # Save and close plot
    plt.tight_layout()
    fname = f'{OUTPUT_DIR}/ml_plots/hyperparam_plots/{model}_hyperparams.png'
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":

    # Run main function and time it
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Finished in {elapsed_time:.2f} minutes")