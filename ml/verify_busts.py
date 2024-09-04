"""
Script to count number of TAF busts, calculate statistics and output
plots and spreadsheets.

Functions:
    add_labels: Adds bust label to bust_labels dictionary.
    calc_labels: Calculates bust labels.
    get_bust_labels: Extracts TAFs/METARs and collects bust information.
    get_metars: Retireves METARs and SPECIs between two times.
    update_labels: Adds label to dictionary if necessary.

Written by Andre Lanyon.
"""
from datetime import timedelta

from taf_monitor.checking import CheckTafThread
from taf_monitor.data_retrieval import RetrieveObservations
from dateutil.rrule import HOURLY, rrule
from taf_monitor.time_functionality import ConstructTimeObject as ct

# Accepted first guess TAFs
FGT_FILE = ('/data/users/alanyon/tafs/verification/data_20220308_cld_'
            'vis/Output/acceptedTafs.csv')
# TAF terms
TAF_TERMS = ['BECMG', 'TEMPO', 'PROB30', 'PROB40']
# To convert heading into direction label (N, S, E or W)
NUM_TO_DIR = dict(zip(range(0, 370, 10),
                      list('NNNNNEEEEEEEEESSSSSSSSSWWWWWWWWWNNNNN')))
# Wind bust types and direction strings
B_TYPES = ['increase', 'decrease', 'dir']
DIRS = ['N', 'E', 'S', 'W', 'VRB']
# String names of lists and dictionaries used to collect data
NAMES = ['wind_info', 'wind_stats', 'fg_dirs', 'is_dirs', 'metar_dirs',
         'all_info', 'all_stats']


def add_labels(bust_labels, hours, busts, wx_type, w_dir):
    """
    Adds bust label to bust_labels dictionary.

    :param bust_labels: Dictionary to add bust labels to
    :type bust_labels: dict
    :param hours: List of datetimes at hourly intervals over TAF period
    :type hours: list
    :param busts: List of bust information
    :type busts: list
    :param wx_type: Type of weather
    :type wx_type: str
    :param w_dir: Indicator for if bust is due to directional windchange
    :type w_dir: bool

    :return: Updated dictionary of bust labels
    :rtype: dict

    """
    for bust in busts:

        # Unpack list
        bust_types, _, metar_time = bust

        # Closest hour in TAF period
        closest_taf_time = min(hours, key=lambda d: abs(d - metar_time))

        # Wind bust messages
        if wx_type == 'wind':

            # Get bust label from bust types and make label string
            for bust_type in bust_types:

                if bust_types[bust_type]:

                    # Only need last word of bust type
                    b_label = f'{wx_type}_{bust_type.split()[-1]}'

                    # Separate directional changes from strength changes
                    if any([w_dir and 'dir' in b_label,
                            not w_dir and 'dir' not in b_label]):

                        # Update bust_labels if necessary
                        bust_labels = update_labels(
                            bust_labels, b_label, closest_taf_time,
                            wx_type
                            )

        # Cloud and visibility bust messages
        else:

            # Make label
            b_label = f'{wx_type}_{bust_types}'

            # Update bust_labels if necessary
            bust_labels = update_labels(bust_labels, b_label,
                                        closest_taf_time, wx_type)

    return bust_labels


def calc_labels(busts_dict, start, end, wxs, w_dir=False):
    """
    Calculates bust labels.

    :param busts_dict: Dictionary of bust information
    :type busts_dict: dict
    :param start: Start time to of TAF period
    :type start: datetime.datetime
    :param end: End time of TAF period
    :type end: datetime.datetime
    :param wxs: List of weather types to check for
    :type wxs: list
    :param w_dir: Indicator for whether directional busts required,
                  defaults to False
    :type w_dir: bool, optional

    :return: Dictionary of bust labels
    :rtype: dict
    """
    # Dictionary of hourly datetimes during TAF period
    hours = rrule(freq=HOURLY, dtstart=start, until=end)
    bust_labels = {hour: [] for hour in hours}

    # Get relevant parts of busts dictionary based on wxs
    wx_busts_dict = {key: val for key, val in busts_dict.items() if key in wxs}

    # Add labels, if any, to bust labels dictionary
    for wx_str, busts in wx_busts_dict.items():
        bust_labels = add_labels(bust_labels, hours, busts, wx_str, w_dir)

    # Convert lists of labels to strings
    bust_labels = {tdt: (' '.join(bust_labels[tdt]) if bust_labels[tdt]
                         else 'no_bust')
                   for tdt in bust_labels}

    return bust_labels


def get_bust_labels(taf, site_info):
    """
    Extracts TAFs and METARs and compares them, collecting bust
    information.

    :param taf: TAF, separated into its components
    :type taf: List
    :param site_info: Site information
    :type site_info: pandas.DataFrame

    :return: Dictionaries containing bust labels
    :rtype: (dict, dict, dict, dict)
    """
    # Get info from site_df
    valid_dt, icao = site_info['taf_start'], site_info['icao']

    # Get first guess TAF validity times as python datetime objects
    start, end = ct(taf[2], valid_dt.day, valid_dt.month, valid_dt.year).TAF()

    # Add 1 hour to end time to capture last hour of TAF
    end = end + timedelta(hours=1)

    # Get relevant METARs and SPECIs
    metars = get_metars(icao, start, end)
    if not metars:
        return False

    # Verify TAF against METARs and collect bust information
    bust_summaries = CheckTafThread(icao, start, end, taf, metars).run()

    # Make dictionaries mapping types of busts to strings
    # (not using sig wx)
    busts_dict = {'wind': bust_summaries['wind'],
                  'vis': bust_summaries['visibility'],
                  'cld': bust_summaries['cloud']}

    # Get a label for each hour of the TAF
    wind_labels = calc_labels(busts_dict, start, end, ['wind'])
    dir_labels = calc_labels(busts_dict, start, end, ['wind'], w_dir=True)
    vis_labels = calc_labels(busts_dict, start, end, ['vis'])
    cld_labels = calc_labels(busts_dict, start, end, ['cld'])

    return wind_labels, dir_labels, vis_labels, cld_labels


def get_metars(icao, s_time, e_time):
    """
    Retireves METARs and SPECIs between two times for an ICAO code.

    :param icao: ICAO airport identifier
    :type icao: str
    :param s_time: Start time to search for METARs
    :type s_time: datetime.datetime
    :param e_time: End time to search for METARs
    :type e_time: datetime.datetime

    :return: List of METARs
    :rtype: list
    """
    # Get METARs and SPECIs
    try:
        GetObs = RetrieveObservations
        metars = GetObs(icao, "M", s_time, latest_only=False,
                        start_time=s_time, end_time=e_time).operation()
        specis = GetObs(icao, "S", s_time, latest_only=False,
                        start_time=s_time, end_time=e_time).operation()
    except:
        return False

    # Collect METARs and SPECIs into single list
    metars += specis

    # Remove METARs and SPECIs recorded as 'NoRecord'
    metars = [metar for metar in metars if metar != "NoRecord"]

    # Remove duplicates (e.g. for METARs with trends)
    new_metars = []
    for ind, metar in enumerate(metars):
        if ind == 0 or metar[1] == current_metar[1]:
            current_metar = metar
        else:
            new_metars.append(current_metar)
            current_metar = metar
        if ind == len(metars) - 1:
            new_metars.append(current_metar)

    # Remove AUTO term from METARs and SPECIs as it has no value
    new_metars = [[ele for ele in metar if ele != 'AUTO']
                  for metar in new_metars]

    # Sort list so SPECIs in time order with METARs
    new_metars.sort(key=lambda x: x[1])

    return new_metars


def update_labels(bust_labels, b_label, closest_taf_time, wx_type):
    """
    Adds label to dictionary if deemed necessary.

    :param bust_labels: Dictionary containing bust labels
    :type bust_labels: dict
    :param b_label: Bust label to check
    :type b_label: str
    :param closest_taf_time: Time closest to bust time in TAF
    :type closest_taf_time: datetime.datetime
    :param wx_type: Weather type
    :type wx_type: str

    :return: Updated bust label dictionary
    :rtype: dict
    """
    # Get existing labels at closest TAF time
    current_labels = bust_labels[closest_taf_time]

    # Only append if label not already in list
    if b_label not in current_labels:

        # If increased and decreased bust reported in same hour, just
        # use decreased bust for label
        if f'{wx_type}_increase' in current_labels:
            inc_ind = current_labels.index(f'{wx_type}_increase')
            bust_labels[closest_taf_time][inc_ind] = b_label
        elif f'{wx_type}_decrease' not in current_labels:
            bust_labels[closest_taf_time].append(b_label)

        # Ensure list in alphabetical order
        bust_labels[closest_taf_time].sort()

    return bust_labels
