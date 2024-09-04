"""
Generates TAFs.

Functions:
    get_base_conditions: Determines appropriate base conditions.
    get_becmgs: Finds and collects BECMG group information.
    get_taf_options: Creates a list of possible TAFs.
    get_tempos: Finds TEMPO and PROB groups.
    taf_gen: Main function to generate TAF.
"""
from multiprocessing import Process, Queue
import random

import numpy as np

import bases_changes.bases as ba
import bases_changes.becmgs as be
import bases_changes.optimise as op
import bases_changes.tempos as te
import common.calculations as ca
import organise.write_taf as wt


def get_base_conditions(site_data):
    """
    Determines appropriate base conditions at the start of a TAF.

    Args:
        site_data (pandas.DataFrame): IMPROVER and airport data
    Returns:
        bases (dict): All base conditions
    """
    # Need 30th and 50th percentiles (preserving metadata in 50th)
    tdf_taf = site_data[site_data['taf_time'] == 'during']
    tdf_30 = tdf_taf[(tdf_taf['percentile'] == 30)]
    tdf_50 = tdf_taf[(tdf_taf['percentile'] == 50)]

    # To add base values to
    bases = {}

    # Get base wind direction and wind mean
    bases['wind_dir'], num_dirs = ba.get_base_dir(tdf_50)
    base_mean, num_means = ba.get_base_mean(tdf_50)

    # Subset dataframe using same number of values as used with base
    # mean or base direction, whichever is smaller
    end_index = min(num_means, num_dirs)
    s_tdf_50 = tdf_50[:end_index]
    bases['wind_mean'], bases['wind_gust'] = ba.get_base_gust(base_mean,
                                                              s_tdf_50)

    # Get base cloud base values
    bases['clds'], bases['cld_cat'] = ba.get_base_cld(tdf_30, tdf_50)
    bases['cb'] = 0

    # Get vis/sig wx base values
    (bases['vis'], bases['vis_cat'],
     bases['sig_wx'], bases['implied_sig_wx']) = ba.get_base_vis_wx(tdf_50, {})

    # Determine whether CAVOK/NSC required
    bases['cavok'] = ca.use_cavok(bases['vis'], bases['clds'], bases['sig_wx'],
                                  ['vis', 'cld'])

    # Add in extra identifying info to bases dictionary
    bases['change_type'] = 'base'
    bases['wx_changes'] = []
    bases['change_period'] = [tdf_50['time'].iloc[0], tdf_50['time'].iloc[0]]

    return bases


def get_becmgs(site_data, bases):
    """
    Finds and collects BECMG group information, returning list
    containing base conditions and BECMG groups.

    Args:
        site_data (pandas.DataFrame): IMPROVER and airport data
        bases (dict): Initial base conditions
    Returns:
        all_bases_becmgs (list): All base conditions and BECMG changes
    """
    # Get required IMPROVER data
    tdf_50 = site_data[(site_data['percentile'] == 50)]
    tdf_taf = tdf_50[tdf_50['taf_time'].isin(['during', 'after'])]

    # Start with one option containing initial base conditions
    becmg_options = [{'groups': [bases], 'data': tdf_taf,  'score': 0,
                      'finished': False}]
    keep_searching = True

    # Find options for BECMG groups
    while keep_searching:
        becmg_options, keep_searching = be.find_becmg(becmg_options)

    return becmg_options


def get_taf_options(becmg_options_chunk, site_data):
    """
    Creates a list of possible TAFs.

    Args:
        becmg_options_chunk (list): Chunk of BECMG options
        site_data (pandas.DataFrame): IMPROVER and airport data
    Returns:
        taf_options (list): TAF options
    """
    # Find PROB/TEMPO groups for each BECMG option and collect resulting
    # TAF options
    taf_options = []

    for option in becmg_options_chunk:

        bases_becmgs = option['groups']

        # Add TEMPO/PROB groups for each weather parameter
        all_groups, base_period = get_tempos(site_data, bases_becmgs)

        # Optimise BECMG/TEMPO/PROB groups
        all_groups = op.optimise_groups(all_groups, site_data)

        # Change base period
        all_groups[0]['change_period'] = base_period

        # FOR TESTING
        # wt.taf_text(site_data, all_groups)

        # Add to TAF options, including BECMG score
        taf_options.append({'score': option['score'], 'groups': all_groups})

    return taf_options


def get_tempos(site_data, all_groups):
    """
    Finds TEMPO and PROB groups for each weather type (wind, vis/wx and
    cloud) and adds to change groups list.

    Args:
        site_data (pandas.DataFrame): IMPROVER and airport data
        all_groups (dict): Base conditions and BECMG groups
    Returns:
        all_groups (list): Base conditions and all change groups
    """
    # Get required IMPROVER data (all percentiles during TAF period)
    tdf = site_data[site_data['taf_time'].isin(['during'])]

    # Get TEMPO/PROB groups for each type of weather
    all_groups = te.param_tempos(tdf, all_groups, 'wind')
    all_groups = te.param_tempos(tdf, all_groups, 'vis')
    all_groups = te.param_tempos(tdf, all_groups, 'cld')

    # For updating base period with correct dts after optimising
    base_period = [tdf['time'].iloc[0], tdf['time'].iloc[-1]]

    return all_groups, base_period


def taf_gen(site_data):
    """
    Main function to generate TAF.

    Args:
        site_data (pandas.DataFrame): IMPROVER and airport data
    """
    # Print out relevant data for testing and comparing to TAF
    # wt.print_data(site_data)

    # Get base conditions
    bases = get_base_conditions(site_data)

    # Get BECMG groups
    becmg_options = get_becmgs(site_data, bases)

    # Reduce number of options by only taking options with minimum 
    # number of groups
    min_groups = min(len(option['groups']) for option in becmg_options)
    becmg_options = [opt for opt in becmg_options 
                     if len(opt['groups']) == min_groups]

    # Bit of a hack to limit ridiculous number of options
    becmg_options = random.sample(becmg_options, min(len(becmg_options), 30))

    # Split into chunks for multi-processing
    print(f'Number of BECMG options: {len(becmg_options)}')
    chunks = np.array_split(becmg_options, min([len(becmg_options), 10]))

    # Define variables for multiprocessing
    queue = Queue()
    processes = []

    # Loop through becmg_options chunks, then find PROB/TEMPO groups for
    # each BECMG option and collect resulting TAF options
    for chunk in chunks:

        # Define arguments for multiprocessing
        args = (get_taf_options, [chunk, site_data], queue)

        # Append process for multiprocessing
        processes.append(Process(target=ca.mp_queue, args=args))

    # Start processes
    for process in processes:
        process.start()

    # Collect output from processes and close queue
    all_taf_options = [queue.get() for _ in processes]
    queue.close()

    # Flatten options
    all_taf_options = [option for options in all_taf_options
                       for option in options]

    # Get TAF options with least smallest number of change groups
    min_groups = min(len(option['groups']) for option in all_taf_options)
    shortest_tafs = [option for option in all_taf_options
                     if len(option['groups']) == min_groups]

    # If multiple short TAFs, pick the one with the lowest BECMG score
    best_option = min(shortest_tafs, key=lambda x: x['score'])

    # Write TAF for best option
    taf = wt.taf_text(site_data, best_option['groups'])

    return taf
