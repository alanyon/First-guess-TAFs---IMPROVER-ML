"""
Master script calling all other modules relating to TAF generation.

Functions:
    main: Main function.
"""
import os
import pickle

import common.configs as co
import data_extraction.extract_sort_data as ex
import generate.generate_taf as ge
import ml.verify_busts as vb
import ml.data_sorting as ds

# Define environment constants
TAF_START = os.environ['TAF_START']
VER_DIR = os.environ['VER_DIR']


def main():
    """
    Calls other modules to generate TAFs.
    """
    # Make output directory if needed
    if not os.path.exists(VER_DIR):
        make_dirs()

    # Define path that data will be stored in
    d_file = f'{co.TEST_DIR}/{TAF_START}'

    # If data already extracted, use that
    if os.path.exists(d_file):
        param_dfs_missing_times, airport_info, taf_dts = unpickle_file(d_file)

    # Otherwise, need to extract data from MASS
    else:

        # Extract all relevant IMPROVER data, and required TAF variables
        (param_dfs_missing_times,
         airport_info, taf_dts) = ex.get_imp_data(TAF_START)

        # Save as csv file for testing
        pickle_data([param_dfs_missing_times, airport_info, taf_dts], d_file)

    # Filter data for each airport and collect
    for _, site_info in airport_info.iterrows():

        # Ignore defence for now
        if site_info['rules'] == 'defence':
            continue

        # FOR TESTING
        # if site_info['icao'] != 'EGNT':
        #     continue

        # For info
        icao = site_info['icao']
        print(icao)

        # Get data for airport
        site_df = ex.get_site_data(param_dfs_missing_times, site_info,
                                   taf_dts)

        # If no data found, move to next airport
        if site_df.empty:
            continue

        # Generate TAF and write to txt file
        ver_taf = ge.taf_gen(site_df)
        txt_file = f'{VER_DIR}/tafs/{icao}_all_old.txt'
        with open(txt_file, 'a', encoding='utf-8') as t_file:
            t_file.write(ver_taf)

        # Change to format needed for bust labels
        taf = ver_taf[46:].split()

        # Get label for TAF based on whether it would have gone bust
        bust_labels = vb.get_bust_labels(taf, site_info)

        if not bust_labels:
            continue

        # Organise dataframe for use in ml training
        ml_df = ds.get_ml_df(site_df, bust_labels=bust_labels)

        # Define pickle file path
        icao_pickle = f'{VER_DIR}/pickles/pickle_{icao}'

        # If file already exists, add to it
        if os.path.exists(icao_pickle):
            ml_data = unpickle_file(icao_pickle)
            ml_data.append([ml_df, site_df])

        # Otherwise, create new list of data
        else:
            ml_data = [[ml_df, site_df]]

        # Pickle data 
        pickle_data(ml_data, icao_pickle)


def make_dirs():
    """
    Creates required directories.
    """
    # Create main directory
    os.system(f'mkdir {VER_DIR}')

    # Create sub directories
    for dir_name in ['pickles', 'tafs']:
        os.system(f'mkdir {VER_DIR}/{dir_name}')


def pickle_data(data, p_file):
    """
    Saves object as pickle file.
    """
    with open(p_file, 'wb') as f_object:
        pickle.dump(data, f_object)


def unpickle_file(p_file):
    """
    Unpickles a pickled file.
    """
    with open(p_file, 'rb') as file_object:
        unpickler = pickle.Unpickler(file_object)
        data = unpickler.load()

    return data


if __name__ == "__main__":
    main()
