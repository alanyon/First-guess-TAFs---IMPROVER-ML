import pickle
import time
import pandas as pd

import common.configs as co
import generate.generate_taf as ge

# Import environment variables
OUTPUT_DIR = os.environ['OUTPUT_DIR']
FAKE_DATE = os.environ['FAKE_DATE']

# Turn off pandas 'chained' warning
pd.options.mode.chained_assignment = None


def main():
    """
    Main function to build classifiers to predict bust labels.
    """
    # Get icao from date icao dictionary
    icao = co.DATE_ICAOS[FAKE_DATE]

    # Unpickle data
    i_pickle = f'{OUTPUT_DIR}/pickles/pickle_{icao}'
    with open(i_pickle, 'rb') as file_object:
        unpickler = pickle.Unpickler(file_object)
        taf_data = unpickler.load()

    # Unpack data
    for (ml_data, site_df) in taf_data:

        # Generate TAF and write to txt file
        taf = ge.taf_gen(site_df)
        txt_file = f'{OUTPUT_DIR}/tafs/{icao}_all_old.txt'
        with open(txt_file, 'a', encoding='utf-8') as t_file:
            t_file.write(taf)


if __name__ == "__main__":

    # Run main function and time it
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Finished in {elapsed_time:.2f} minutes")
