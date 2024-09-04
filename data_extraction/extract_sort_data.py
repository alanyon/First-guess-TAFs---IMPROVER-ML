"""
Extracts and sorts data ready for TAF generation.

Functions:
    cld_cat_row: Determines cloud TAF category based on cloud values.
    day_season: Determines day type and season at start of TAF period.
    estimate_precip: Estimates precip code based on other parameters.
    estimate_sig_wx: Estimates sig wx code.
    extract_data: Extracts relevant data from MASS.
    fill_in_sig_wxs: Fills in missing sig wx values.
    get_imp_data: Collects required IMPROVER data from files.
    get_open_taf_hours: Gets valid TAF datetimes based on airport hours.
    get_site_data: Filters IMPROVER data to get airport-specific data.
    get_start_end_dts: Finds start and end times for subsetting data.
    get_taf_hrs: Gets all possible TAF hours based on longest TAF.
    load_filter_data: Loads in IMPROVER data for a weather parameter.
    round_dir: Rounds wind direction to nearest 10 degrees.
    round_vis_row: Rounds vis appropriately on row of dataframe.
    update_sig_wx: Converts sig wx codes to TAF-specific strings.
    update_values: Updates IMPROVER values.
    update_vis: Checks visibility lines up with cloud, wind and sig wx.
    vis_cat_row: Determines visibility TAF category based on vis value.
"""
import glob
import math
import os
import queue as queue_module
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import Process, Queue
from zoneinfo import ZoneInfo

import holidays
import iris
import pandas as pd
from dateutil.rrule import HOURLY, rrule
from iris.pandas import as_data_frame
from iris.util import equalise_attributes, promote_aux_coord_to_dim_coord

import common.calculations as ca
import common.checks as ch
import common.configs as co

# To stop pandas warnings
iris.FUTURE.pandas_ndim = True
pd.options.mode.chained_assignment = None


def cld_cat_row(row):
    """
    Determines cloud base TAF category based on info in row of
    dataframe.

    Args:
        row (pandas.Series): Row of dataframe
    Return:
        cld_cat (float): Cloud base category
    """
    # Get cloud cat using values from row
    cld_cat = ca.get_cld_cat(row['cld_3'], row['cld_5'], row['rules_col'])

    return cld_cat


def day_season(row):
    """
    Determines the day type and season at the start of the TAF period
    for row of airport_info - needed to determine opening and closing
    times of airports.

    Args:
        row (pandas.Series): Airport information
    Returns:
        season_day_type (str): Season and day type
    """
    if row['taf_start'] in holidays.CountryHoliday('GB',
                                                   subdiv=row['country_code']):
        day_type = 'ph'
    elif row['taf_start'].strftime('%A') == 'Saturday':
        day_type = 'sat'
    elif row['taf_start'].strftime('%A') == 'Sunday':
        day_type = 'sun'
    else:
        day_type = 'week'

    # Determine if summer or winter (based on if in British Summertime)
    tz_info = ZoneInfo('Europe/London')
    if tz_info.dst(row['taf_start']) == timedelta(0):
        season = 'w'
    else:
        season = 's'

    # Combine into single string
    season_day_type = f'{season}_{day_type}'

    return season_day_type


def estimate_precip(row):
    """
    Estimates precip code based on other parameters.

    Args:
        row (pandas.Series): Row of dataframe
    Return:
        wx_str (str): Sig wx code
    """
    # 50th percentile sig wx and temp are currently code currently
    # assigned to sig_wx
    sig_wx_50, temp_50 = row['sig_wx']

    # If precip in 50th percentile sig wx, assume same type here
    if 'TS' in sig_wx_50:
        precip_type = 'TS'
    elif 'SH' in sig_wx_50:
        precip_type = 'SH'
    elif 'DZ' in sig_wx_50:
        precip_type = 'DZ'

    # Otherwise, assume dynamic rain
    else:
        precip_type = 'RA'

    # Put everything together to get sig wx code, using either SN or RA
    if row['temp'] < 0:
        if precip_type in ['TS', 'SH']:
            wx_str = f'{precip_type}SN'
        else:
            wx_str = 'SN'
    elif precip_type in ['TS', 'SH']:
        wx_str = f'{precip_type}RA'
    elif precip_type == 'DZ':
        wx_str = 'DZ'
    elif 'SN' in sig_wx_50 and row['temp'] <= temp_50:
        wx_str = 'RASN'
    else:
        wx_str = 'RA'

    # Ensure appropiate precip rate is used (if necessary)
    wx_str = ch.check_rate(row['precip_rate'], row['vis'], row['temp'],
                           wx_str)

    return wx_str


def estimate_sig_wx(row):
    """
    Estimates sig wx code based on visibility, precip rate and screen
    temp. IMPROVER only outputs weather codes at the 50th percentile (as
    it is hard to define what the x percentile weather code would be),
    so codes are estimated here based on other IMPROVER parameters.

    Args:
        row (pandas.Series): Row of dataframe
    Return:
        wx_str (str): Sig wx code
    """
    # Leave 50th percentiles unchanged
    if row['percentile'] == 50:
        return row['sig_wx']

    # If precip signalled, try to match to 50th percentile sig wx
    if row['precip_rate'] > 0.1:

        # # Get an appropriate precip wx code
        wx_str = estimate_precip(row)

    # If precip code not appropriate, leave empty for now (BR/FG
    # checked later)
    else:
        wx_str = ''

    # Ensure appropriate mist/fog code used
    wx_str = ch.check_mist_fog(row['vis'], row['temp'], row['rules_col'], wx_str)

    return wx_str


def extract_data(blend_str):
    """
    Extracts relevant data from MASS.

     Args:
        blend_str (str): String containing blend date.
    """
    # Extract tar file from MASS
    moo_cmd = subprocess.run(
        ['moo', 'get', f'{co.MASS_DIR}/mix_suite_{blend_str}00Z/spot.tar',
         f'{co.DATA_DIR}/{blend_str}00Z_spot.tar'],
         check=False, capture_output=True, encoding="utf-8"
    )

    # Exit if file not on MASS
    if moo_cmd.returncode not in [0]:
        print(moo_cmd.returncode)
        raise FileNotFoundError("archive not found on mass")

    # Get list of filenames from tar file
    list_files_cmd = subprocess.run(
        ['tar', '-tf', f'{co.DATA_DIR}/{blend_str}00Z_spot.tar'], check=True,
         capture_output=True, encoding="utf-8")
    fnames_in_tar = list_files_cmd.stdout.splitlines()

    # Get list of filenames in tar file needed for TAF generation
    fnames_to_extract = []

    # Loop through required parameters and determine filenames
    param_fnames = {}
    for param, param_configs in co.IMPROVER_PARAMETERS.items():

        # Define start of filename to extract data from
        fname_start = param_configs['fname_start']
        alt_fname_start = param_configs['fname_start_alt']
        fname_pattern = f"spot/{fname_start}.*B.*Z-{param}.nc"
        alt_fname_pattern = f"spot/{alt_fname_start}.*B.*Z-{param}.nc"

        # List to add filenames to
        param_fnames_arch = []

        # Find all files in tar that match filename pattern
        for file in fnames_in_tar:
            matches = re.findall(fname_pattern, file)
            if matches:
                param_fnames_arch.extend(matches)

        # Use alternative filename pattern if no matches
        if not param_fnames_arch:
            param_fnames[param] = alt_fname_start
            for file in fnames_in_tar:
                matches = re.findall(alt_fname_pattern, file)
                if matches:
                    param_fnames_arch.extend(matches)
        else:
            param_fnames[param] = fname_start

        # if no matching files than raise error
        if not param_fnames_arch:
            raise FileNotFoundError(f'No files for {param} found in tar '
                                    'archive')

        # Add to list of required files
        fnames_to_extract.extend(param_fnames_arch)

    # Make directory to put files in
    dest = f'{co.DATA_DIR}/{blend_str}00Z'
    if not os.path.exists(dest):
        os.makedirs(dest)

    # Write files to query file
    fetch_files_file = f'{co.DATA_DIR}/{blend_str}00Z_query_file'
    with open(fetch_files_file, "w", encoding="utf-8") as f:
        f.write("\n".join(fnames_to_extract))

    # Untar all files
    tar_cmd = subprocess.run(
        ['tar', '-xC', dest, '-f', f'{co.DATA_DIR}/{blend_str}00Z_spot.tar',
         f'--files-from={fetch_files_file}',
         '--strip-components=1'],
         check=False, capture_output=True, encoding="utf-8"
    )

    # Print out errors and output if untarring fails
    if tar_cmd.stderr or tar_cmd.stdout:
        print(f'/nTAR ERRORS\n{tar_cmd.stderr}/nTAR OUTPUT\n{tar_cmd.stdout}')
    tar_cmd.check_returncode()

    return param_fnames


def fill_in_sig_wxs(site_df):
    """
    Fills in missing sig wx values.

    Args:
        site_df (pandas.DataFrame): Dataframe containing IMPROVER and
                                    airport data
    Return:
        site_df (str): Updated dataframe
    """
    # First, set non-50th percentile sig wxs to contain 50th percentile
    # sig wxs and temps
    # (There's probably a better way to do this)
    for ind, row in site_df.iterrows():
        if not isinstance(row['sig_wx'], str):
            row_50 = site_df[(site_df['time'] == row['time']) &
                             (site_df['percentile'] == 50)]
            site_df.at[ind, 'sig_wx'] = [row_50['sig_wx'].values[0],
                                         row_50['temp'].values[0]]

    # Now update these values using other weather elements in each row
    site_df['sig_wx'] = site_df.apply(estimate_sig_wx, axis=1)

    return site_df


def get_imp_data(taf_start):
    """
    Collects required IMPROVER data from files.

    Args:
        taf_start (str): TAF start string
    Returns:
        param_dfs_missing_times (tuple): Dataframes containing IMPROVER
                                         data and missing times
        airport_info (pandas.DataFrame): Airport information
        taf_dts (list): Datetimes relevant to TAF period
    """
    # Get taf start time as datetime object
    taf_start_dt = datetime.strptime(taf_start, '%Y%m%d%H')

    # Assume blend time is 3 hours before TAF start
    blend_dt = taf_start_dt - timedelta(hours=3)
    blend_str = blend_dt.strftime('%Y%m%dT%H')

    # Load in airport info
    airport_info = pd.read_csv(co.AIRPORT_INFO_FILE, header=0)

    # Get datetimes to cover all TAFs (max 30 hours)
    airport_info, taf_dts = get_taf_hrs(airport_info, taf_start_dt)

    # Define constraints for loading data
    sites = [f'{site:08d}' for site in list(airport_info['site_number'])]
    sites_con = iris.Constraint(met_office_site_id=sites)
    perc_con = iris.Constraint(percentile=co.PERCENTILES)

    # Extract data from MASS, collecting fnames for each parameter
    param_fnames = extract_data(blend_str)

    # Define variables for multiprocessing
    queue = Queue()
    processes = []

    # Get required IMPROVER data from extracted files
    for param, fname in param_fnames.items():

        # Define arguments for multiprocessing
        args = (load_filter_data,
                [param, sites_con, perc_con, taf_dts, fname, blend_str],
                queue)

        # Append process for multiprocessing
        processes.append(Process(target=ca.mp_queue, args=args))

    # Start processes
    for process in processes:
        process.start()

    # Collect output from processes and check for errors
    param_dfs_missing_times = []
    error_occurred = False

    # Use a loop with timeout to periodically check for queue messages
    while processes:
        try:
            # Try to get the result from the queue without blocking
            result = queue.get_nowait()

            if result == co.ERROR_SENTINEL:
                error_occurred = True
                break
            param_dfs_missing_times.append(result)

        except queue_module.Empty:
            # No messages yet, wait a bit and continue checking
            time.sleep(0.1)

        # Check if any process has finished
        for process in processes.copy():
            if not process.is_alive():
                processes.remove(process)

    # If an error occurred, terminate all running processes
    if error_occurred:
        print('An error occurred in one of the loading data processes. '
              'Terminating all processes.', file=sys.stderr)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()  # Ensure the process is terminated
        queue.close()
        sys.exit(1)  # Exit with a non-zero status indicating an error
    else:
        # Wait for all remaining processes to complete normally
        for process in processes:
            process.join()

    queue.close()

    # Remove blend time data files
    os.system(f'rm -r {co.DATA_DIR}/{blend_str}*')

    return param_dfs_missing_times, airport_info, taf_dts


def get_open_taf_hours(site_info, taf_dts):
    """
    Gets valid TAF datetimes based on airport opening hours.

    Args:
        site_info (pandas.Seriers): Series containing airport info
        taf_dts (list): Datetimes relevant to TAF period
    Returns:
        site_dts (dict): TAF period datetimes relevant to airport
    """
    # Get taf info from site_info
    taf_start = site_info['taf_start']
    taf_len = int(site_info['taf_len'])

    # TAFs not needed for airports that receive TAFs every 6 hours if
    # TAF start is 03Z, 09Z, 15Z or 21Z
    if taf_start.hour in [3, 9, 15, 21] and site_info['taf_freq'] == 6:
        return None

    # Get opening hours based on day type and season
    open_hr = site_info[f'{site_info["season_day"]}_start']
    close_hr = site_info[f'{site_info["season_day"]}_end']

    # Do not need to generate TAF if airport not open
    if math.isnan(open_hr):
        return None

    # Ensure open/close hours are integers
    open_hr, close_hr = int(open_hr), int(close_hr)

    # 99s indicate TAF is open 24 hours so use all hours (with 3 hours
    # before and after TAF)
    if open_hr == 99 and close_hr == 99:
        start, end = taf_dts[0], taf_dts[taf_len + 3]

    # Remainder of cases are for non-24hr airports that are open at the
    # start of the TAF period
    else:

        # Get start and end dts (including 3 hours before and after TAF)
        start, end = get_start_end_dts(taf_start, open_hr, close_hr, taf_len)

    # Return None if no start dt found
    if start is None:
        return None

    # Define site dts for airport, incluing info as whether dts are
    # before, during or after TAF period
    site_dts = {}

    for taf_dt in taf_dts:
        if taf_dt < start or taf_dt > end:
            continue
        if taf_dt > end - timedelta(hours=3):
            site_dts[taf_dt] = 'after'
        else:
            site_dts[taf_dt] = 'during'

    # Civil TAFs need to be at least 2 hours long, defence TAFs at
    # least 3 hours long (including 3 hours before and after TAF)
    if site_info['rules'] == 'defence'and len(site_dts) < 7:
        return None
    if site_info['rules'] != 'defence' and len(site_dts) < 6:
        return None

    return site_dts


def get_site_data(param_dfs_missing_times, site_info, taf_dts):
    """
    Filters IMPROVER data to obtain data relevant to airport.

    Args:
        param_dfs_missing_times (tuple): Dataframes containing IMPROVER
                                         data and missing times
        site_info (pandas.Series): Series containing airport info
        taf_dts (list): Datetimes relevant to TAF period
    Returns:
        site_df (pandas.DataFrame): Dataframe containing IMPROVER and
                                    airport data
    """
    # Get TAF datetimes based on airport opening hours
    site_dts = get_open_taf_hours(site_info, taf_dts)

    # Return empty dataframe valid TAF datetimes found
    if not site_dts:
        print('TAF not due')
        return pd.DataFrame()

    # Loop through all parameter dataframes, take out revevant data and
    # merge into single dataframe
    for ind, (param_df, missing_times) in enumerate(param_dfs_missing_times):

        # Check for any missing data
        if any(m_time in site_dts for m_time in missing_times):
            print('Missing data')
            return pd.DataFrame()

        # Subset data using opening hours of airport and site number
        param_df = param_df[param_df['time'].isin(site_dts)]
        param_df = param_df[param_df['site'] == site_info['site_number']]

        # Don't need site column anymore
        param_df.drop(['site'], axis=1, inplace=True)

        # For first iteration, set site dataframe to parameter dataframe
        if ind == 0:
            site_df = param_df

        # Otherwise merge parameter dataframe with site dataframe
        else:
            site_df = site_df.merge(param_df, how='outer',
                                    on=['time', 'percentile'])

    # Abandon if dataframe is empty
    if site_df.empty:
        print('No data')
        return pd.DataFrame()

    # Add time info to dataframe
    site_df['taf_time'] = site_df['time'].map(site_dts)

    # Add taf type to as column for applying functions to rows
    site_df['rules_col'] = site_info['rules']

    # Update values - e.g. ensure correct units, round values if
    # necessary and ensure internal consistency between visibility,
    # cloud and sig wx
    site_df = update_values(site_df)

    # Add required airport info as attributes to site_df
    vrbs = ['rules', 'airport_name', 'icao', 'taf_issue', 'taf_start', 'bench']
    site_df.attrs = {vrb: site_info[vrb] for vrb in vrbs}

    return site_df


def get_start_end_dts(taf_start, open_hr, close_hr, taf_len):
    """
    Finds start and end datetimes to use for subsetting IMPROVER data.

    Args:
        taf_start (datetime.datetime): Start of TAF period
        open_hr (int): Opening hour of airport
        close_hr (int): Closing hour of airport
        taf_len (int): Length of TAF in hours
    Returns:
        start (datetime.datetime): Start time to extract data
        end (datetime.datetime): End time to extract data
    """
    # Define opening and closing datetimes
    if all([taf_start.hour > close_hr, close_hr > open_hr]):
        open_dt = (taf_start + timedelta(days=1)).replace(hour=open_hr)
    else:
        open_dt = taf_start.replace(hour=open_hr)
    if close_hr < open_hr:
        close_dt = (open_dt + timedelta(days=1)).replace(hour=close_hr)
    else:
        close_dt = open_dt.replace(hour=close_hr)

    # Cases in which TAF period start in opening hours of airport
    if open_dt <= taf_start <= close_dt:
        start = taf_start

    # If TAF period starts outside of opening hours, generate TAF if
    # airport opens in 2 hours or less
    elif 0 < (open_dt - taf_start).total_seconds() <= 7200:
        start = open_dt

    # In all other cases, do not generate TAF
    else:
        return None, None

    # End index defined by length of TAF or closing time of
    # airport, whichever is earlier
    if close_dt < (taf_start + timedelta(hours=taf_len)):
        end = close_dt + timedelta(hours=3)
    else:
        end = taf_start + timedelta(hours=taf_len + 3)

    return start, end


def get_taf_hrs(airport_info, taf_start_dt):
    """
    Gets all possible TAF hours based on longest TAF (30 hours). Also
    returns strings representing the type of day and season in which the
    TAF period starts.

    Args:
        airport_info (pandas.DataFrame): Airport information
        taf_start_dt (datetime.datetime): TAF start time
    Returns:
        airport_info (pandas.DataFrame): Airport information
        all_hours (list): Datetimes relevant to TAF period
    """
    # Add TAF time variables to airport info df
    airport_info['taf_start'] = taf_start_dt
    airport_info['taf_end'] = taf_start_dt + timedelta(hours=30)

    # Assume TAF will be issued 1 hour before TAF start period
    airport_info['taf_issue'] = taf_start_dt - timedelta(hours=1)

    # Get strings representing day type and season at TAF start
    airport_info['season_day'] = airport_info.apply(day_season, axis=1)

    # Get relevant hours to all TAFs - also need data 3 hours after TAF
    # period
    all_hours = list(rrule(HOURLY, interval=1, dtstart=taf_start_dt, count=34))

    return airport_info, all_hours


def load_filter_data(param, sites_con, perc_con, taf_dts, fname, blend_str):
    """
    Loads in all required IMPROVER data for a single weather parameter,
    sorting as necessary, converting to a dataframe and returning as
    dictionary in order to identify parameter data refers to.

    Args:
        param (str): Weather parameter used for IMPROVER file name
        sites_con (iris.Constraint): Constraint for site numbers
        perc_con (iris.Constraint): Constraint for percentiles
        taf_dts (list): List of datetimes revevant for TAF period
        fname (str): String with start of fname
        blend_str (str): Blend time string
    Returns:
        param_df (pandas.DataFrame): Dataframe containing IMPROVER data
        missing_times (list): List of any times missing from IMPROVER
    """
    # Cube list to append cubes to
    param_cube_list = iris.cube.CubeList([])

    # Collect any times of missing files
    missing_times = []

    # Loop through all required taf dts
    for tdt in taf_dts:

        # Convert TAF dt to string
        tdt_str = tdt.strftime('%Y%m%dT%H%MZ')

        # Indicator for whether file found in archive
        file_found = False

        # Get latest blend time possible
        for minutes in ['45', '30', '15', '00']:

            # Define file and test if it exists
            tdt_file = (f'{co.DATA_DIR}/{blend_str}00Z/{fname}{tdt_str}'
                        f'-B{blend_str}{minutes}Z-{param}.nc')
            if not os.path.exists(tdt_file):
                continue

            # Indicate that file has been found
            file_found = True

            # Load cube - how cube is loaded depends on type
            if co.IMPROVER_PARAMETERS[param]['data_type'] == 'percentiles':
                tdt_cube = iris.load_cube(tdt_file, sites_con & perc_con)
            elif co.IMPROVER_PARAMETERS[param]['data_type'] == 'deterministic':
                tdt_cube = iris.load_cube(tdt_file, sites_con)
            else:
                tdt_cube = iris.load_cube(tdt_file, sites_con)

            # Break for loop if suitable cube found
            break

        # Add to cube list (if suitable file found)
        if file_found:
            param_cube_list.append(tdt_cube)

        # If no file, add to missing times
        else:
            missing_times.append(tdt)

    # Merge cubes
    equalise_attributes(param_cube_list)
    param_cube = param_cube_list.merge_cube()

    # Convert to required units if necessary
    if co.IMPROVER_PARAMETERS[param]['units']:
        param_cube.convert_units(co.IMPROVER_PARAMETERS[param]['units'])

    # Make met_office_site_id a dimension coordinate so that it is
    # recognised when converting to pandas dataframe
    mo_points = param_cube.coord('met_office_site_id').points.astype(int)
    param_cube.coord('met_office_site_id').points = mo_points
    promote_aux_coord_to_dim_coord(param_cube, "met_office_site_id")

    # Convert to pandas dataframe and change param column to short name
    param_df = as_data_frame(param_cube)
    param_df.columns = [co.IMPROVER_PARAMETERS[param]['short_name']]

    # Add columns for site and time from index values
    param_df.reset_index(inplace=True)
    if 'percentile' not in param_df.columns:
        param_df['percentile'] = 50

    # Change site column name to more user-friendly name
    param_df.rename(columns={'met_office_site_id': 'site'}, inplace=True)

    return param_df, missing_times


def round_dir(wdir):
    """
    Rounds wind direction to nearest 10 degrees.

    Args:
        wdir (float): Raw wind direction
    Return:
        rounded_dir (float): Rounded wind direction
    """
    # Ignore nans
    if math.isnan(wdir):
        rounded_dir = wdir

    # Round directions to nearest 10
    else:
        rounded_dir = int(10 * round(float(wdir) / 10))

    # Change any 360s to 0
    if rounded_dir == 360:
        rounded_dir = 0

    return rounded_dir


def round_vis_row(row):
    """
    Rounds visibility appropriately on row of dataframe.

    Args:
        row (pandas.Series): Row of dataframe
    Return:
        rounded_vis (int): Rounded visibility value
    """
    # Round vis using values from row
    rounded_vis = ca.round_vis(row['vis'], row['rules_col'])

    return rounded_vis


def update_sig_wx(row):
    """
    Converts sig wx codes to TAF-specific strings.

    Args:
        row (pandas.Series): Row of dataframe
    Return:
        wx_str (str): Sig wx code
    """
    # Convert to sig wx string if not already done
    if not isinstance(row['sig_wx'], str):

        # Ignore if nan
        if math.isnan(row['sig_wx']):
            return row['sig_wx']

        # Convert to TAF-specific codes
        wx_str = co.SIG_WX_DICT[row['sig_wx']]

    # Otherwise, get wx string already calculated
    else:
        wx_str = row['sig_wx']

    # Ensure appropiate precip rate is used (if necessary)
    wx_str = ch.check_rate(row['precip_rate'], row['vis'], row['temp'],
                           wx_str)

    # Ensure appropriate mist/fog code used
    wx_str = ch.check_mist_fog(row['vis'], row['temp'], row['rules_col'],
                               wx_str)

    return wx_str


def update_values(site_df):
    """
    Updates IMPROVER values, ensuring internal consistency and rounding
    values appropriately.

    Args:
        site_df (pandas.DataFrame): Dataframe containing IMPROVER and
                                    airport data
    Return:
        site_df (pandas.DataFrame): Updated dataframe
    """
    # For cloud parameters, change nans (i.e. no cloud above okta
    # threshold) to high cloud value - 5,000ft
    site_df[['cld_3', 'cld_5']] = site_df[['cld_3', 'cld_5']].fillna(5000)

    # Also, ensure cloud values are all above zero
    for cld in ['cld_3', 'cld_5']:
        site_df[cld][site_df[cld] < 0] = 0

    # Round cloud and visibility values to those generally used in TAFs
    for cld_param in ['cld_3', 'cld_5']:
        site_df[cld_param] = site_df[cld_param].apply(ca.round_cld)
    site_df['vis'] = site_df.apply(round_vis_row, axis=1)

    # Round wind values
    site_df = site_df.round({'wind_mean': 0, 'wind_gust': 0})
    site_df['wind_dir'] = site_df['wind_dir'].apply(round_dir)

    # As no non-50th percentile values for wind dirs, make all
    # percentiles the same as the 50th percentile values
    dirs_50 = site_df.loc[site_df['percentile'] == 50, 'wind_dir'].values
    for perc in co.PERCENTILES:
        site_df.loc[site_df['percentile'] == perc, 'wind_dir'] = dirs_50

    # For really low visibilities, force cloud to be on the surface
    site_df.loc[site_df['vis'] <= 500, ['cld_3', 'cld_5']] = 0

    # Convert sig wx codes and adjust to give TAF-appropriate values
    site_df['sig_wx'] = site_df.apply(update_sig_wx, axis=1)

    # Need to estimate sig wx for 30th, 40th, 60th and 70th percentiles
    # (not available in IMPROVER)
    site_df = fill_in_sig_wxs(site_df)

    # Adjust visibility based on cloud base, wind and sig wx
    site_df['vis'] = site_df.apply(update_vis, axis=1)

    # Update wx again with new visibilities
    site_df['sig_wx'] = site_df.apply(update_sig_wx, axis=1)

    # Get visibility and cloud TAF categories
    site_df['vis_cat'] = site_df.apply(vis_cat_row, axis=1)
    site_df['cld_cat'] = site_df.apply(cld_cat_row, axis=1)

    return site_df


def update_vis(row):
    """
    Checks how visibility lines up with cloud, wind and sig wx forecasts
    and adjusts visibility (always downwards) if necessary.

    Args:
        row (pandas.Series): Row of dataframe
    Return:
        vis (int): Updated visibility value
    """
    # Check against cloud and wind
    vis = ch.check_vis_cld_wind(row['vis'], row['cld_5'], row['wind_mean'])

    # Check against sig wx code
    vis = ch.check_vis_sig_wx(row['sig_wx'], vis)

    return vis


def vis_cat_row(row):
    """
    Determines visibility TAF category based on info in row of
    dataframe.

    Args:
        row (pandas.Series): Row of dataframe
    Return:
        vis_cat (float): Visibility category
    """
    # Get vis cat using visibility and TAF rules from row
    vis_cat = ca.get_vis_cat(row['vis'], row['sig_wx'], row['rules_col'])

    return vis_cat
