"""
Functions to write and save TAF.

Functions:
    get_cld_strs: Creates TAF strings from cloud info.
    get_taf_str: Creates text for base conditions and change groups.
    get_vis_wx_strs: Creates TAF strings from vis and sig wx info.
    get_wind_str: Creates TAF string from wind info.
    nice_format: Organises TAF text into easily read format.
    order_groups: Puts change groups in correct order.
    print_data: Prints out IMPROVER data in easy to read format.
    taf_text: Creates TAF in text form
"""
import os
from datetime import timedelta

import pandas as pd

import common.configs as co

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)


def get_cld_strs(group):
    """
    Creates TAF strings from cloud info.

    Args:
        group (dict): Change group (or base conditions info)
    Returns:
        cld_strs (list): List of cloud strings
    """
    # If NSC indicated, just need this
    if 'NSC' in group['cavok']:
        return ['NSC']

    # If visibility is less than 500m and cloud on surface, assume sky
    # obscured
    if group['clds']['cld_5'] == 0 and group['vis'] < 500:
        return ['VV///']

    # Otherwise, collect relevant cloud strings
    cld_strs = []

    # Add FEW layer if needed
    if group['clds']['cld_1'] != group['clds']['cld_3']:
        cld_strs.append(f'FEW{int(group["clds"]["cld_1"] / 100):03d}')

    # Add SCT layer if needed (only if FEW not already added)
    if not cld_strs and group['clds']['cld_3'] != group['clds']['cld_5']:
        cld_strs.append(f'SCT{int(group["clds"]["cld_3"] / 100):03d}')

    # Add BKN layer if needed
    if group['clds']['cld_5'] < group['clds']['cld_8']:
        cld_strs.append(f'BKN{int(group["clds"]["cld_5"] / 100):03d}')

    # Add OVC layer if needed
    if group['clds']['cld_8'] < 5000:
        cld_strs.append(f'OVC{int(group["clds"]["cld_8"] / 100):03d}')

    # Add CB if indicated (to highest layer reported)
    if group['cb']:
        cld_strs[-1] += 'CB'
        cld_strs[-1] = cld_strs[-1].replace('OVC', 'BKN')

    return cld_strs


def get_taf_str(groups, site_data):
    """
    Converts base conditions and change groups to text to use in TAF.

    Args:
        groups (list): List of base conditions and change groups
        site_data (pandas.DataFrame): IMPROVER data and airport info
    Returns:
        taf_str (str): String with base conditions and change groups
    """
    # Start with empty list to add to
    changes = []

    # Loop through base conditions and all change groups
    for group in groups:

        # Define start and end times for change period
        start, end = group['change_period']

        # Convert to strings of correct format
        start_str = f'{start:%d%H}'

        # for end of period, midnight needs to be 24, not 00
        if end.hour == 0:
            end_str = f'{end - timedelta(hours=1):%d}24'
        else:
            end_str = f'{end:%d%H}'

        # Add change group type and change period if not base conditions
        if group['change_type'] != 'base':

            # Add change type
            changes.append(f'{group["change_type"]}')

        # Add the base (whole TAF period) or change group time period
        changes.append(f'{start_str}/{end_str}')

        # Add wind group if necessary
        if group['change_type'] == 'base' or any('wind' in wx for wx
                                                 in group['wx_changes']):
            changes.append(get_wind_str(group))

        # If CAVOK indicated, just need that for cloud/wx/vis
        if 'CAVOK' in group['cavok']:
            changes.append('CAVOK')

        # Otherwise, consider vis/wx and cloud separately
        else:

            # Add vis/wx group if necessary
            if group['change_type'] == 'base' or 'vis' in group['wx_changes']:
                changes += get_vis_wx_strs(group)

            # Add cloud group if necessary
            if group['change_type'] == 'base' or 'cld' in group['wx_changes']:
                changes += get_cld_strs(group)

    # Convert changes list to one long string
    chs = ' '.join(changes)

    # Add in airport info bits
    taf_str = f'{site_data.attrs["taf_issue"]:%d%H%MZ} {chs}'

    return taf_str


def get_vis_wx_strs(group):
    """
    Creates TAF strings from visibility and significant weather info.

    Args:
        group (dict): Change group (or base conditions info)
    Returns:
        vs_strs (list): List of visibility and sig wx strings
    """
    # Start with visibility
    vs_strs = [f'{group["vis"]:04d}']

    # Add in wx if necessary
    if 'NSW' in group['cavok']:
        vs_strs.append('NSW')
    elif group['sig_wx']:
        vs_strs.append(group['sig_wx'])

    return vs_strs


def get_wind_str(group):
    """
    Creates TAF string from wind info.

    Args:
        group (dict): Change group (or base conditions info
    Returns:
        wind_str (str): Wind string to use in TAF
    """
    # Get direction and mean for wind string, adding gust if necessary
    wind_str = f'{group["wind_dir"]:03d}{group["wind_mean"]:02d}'
    if group["wind_gust"] != group["wind_mean"]:
        wind_str += f'G{group["wind_gust"]:02d}'
    wind_str += 'KT'

    return wind_str


def nice_format(taf_str, site_data):
    """
    Organises TAF text into easily read format, with each change group
    on a different line.

    Args:
        taf_str: (str): Raw TAF text
        site_data (pandas.DataFrame): IMPROVER data and airport info
    Returns:
        nice_taf: (str): Nicely formatted TAF text.
    """
    # Start string with airport name
    nice_taf = (f'\n{site_data.attrs["airport_name"]}:\n\nTAF '
                f'{site_data.attrs["icao"]}')

    # Split remainder of TAF string and loop through elements
    taf_split = taf_str.split()
    for ind, term in enumerate(taf_split):

        # Start new line when change group term found
        if term in ['BECMG', 'TEMPO', 'PROB30', 'PROB40']:

            # This is for 'PROBX TEMPO' cases - ensures these two terms
            # stay on the same line when they follow each other
            if 'PROB' in taf_split[ind - 1]:
                nice_taf = f'{nice_taf} {term}'

            # In all other cases, new line should be started
            else:
                nice_taf = f'{nice_taf}\n     {term}'

        # For start of TAF, do not need space
        elif term == 'TAF':
            nice_taf = f'{nice_taf}{term}'

        # Otherwise, continue on same line
        else:
            nice_taf = f'{nice_taf} {term}'

    # Add an '=' at the end (not sure what this is but it is at the end
    # of manual TAFs)
    nice_taf += '='

    return nice_taf


def order_groups(bases_changes):
    """
    Puts change groups in order. Ordered first by start time, then by
    change group priority...
        0, base
        1. BECMG
        2. TEMPO
        3. PROB40
        4. PROB40 TEMPO
        5. PROB30
        6. PROB30 TEMPO
    ...then by length of change group period

    Args:
        bases_changes: (list): Base conditions and change groups
    Returns:
        bases_changes: (list): Ordered base conditions and change groups
    """
    # Sort groups, by start of change, then by type, then by length
    bases_changes.sort(
        key=lambda x: [x['change_period'][0],
                       co.PRIORITY_DICT[x['change_type']],
                       x['change_period'][1] - x['change_period'][0]]
    )

    return bases_changes


def print_data(sdf):
    """
    Prints out relevant IMPROVER data in easy to read format.

    Args:
        sdf (pandas.DataFrame): IMPROVER and airport data
    """
    # Get times during TAF period, drop unnecessary columns and reorder
    sdf = sdf[sdf['taf_time'] == 'during']
    sdf = sdf[['time', 'percentile', 'wind_dir', 'wind_mean', 'wind_gust',
               'vis', 'sig_wx', 'cld_3', 'cld_5', 'temp', 'precip_rate',
               'lightning']]

    # Seaparate by percentile
    for _, gb_obj in enumerate(sdf.groupby(by='percentile')):

        # Get df, rename columns and reset index before printing
        perc_df = gb_obj[1]
        cols = perc_df.drop(['time', 'percentile'], axis=1).columns
        perc_df.rename({col: f'{col}_{int(gb_obj[0])}' for col in cols},
                       axis=1, inplace=True)
        perc_df.drop(['percentile'], axis=1, inplace=True)
        perc_df.reset_index(drop=True, inplace=True)

        # Print dataframe
        print(perc_df)


def taf_text(site_data, bases_changes):
    """
    Creates a TAF in string form.

    Args:
        site_data: (pandas.DataFrame): IMPROVER data and airport info
        bases_changes (list): Base conditions and change groups
    Returns:
        taf (str): TAF text
    """
    # Order change groups
    ordered_groups = order_groups(bases_changes)

    # Get TAF string
    taf_str = get_taf_str(ordered_groups, site_data)

    # Change to easily read format
    nice_taf = nice_format(taf_str, site_data)

    # Add some extra lines at the end to equally space TAFs - number of
    # spaces depends on number of change groups
    num_spaces = 12 - len(ordered_groups)
    nice_taf += '\n' * num_spaces

    print(nice_taf)

    # Create verification-friendly TAF
    ver_taf = ver_format(taf_str, site_data)

    # Return TAF in format needed for verification
    return ver_taf


def update_html(date):
    """
    Updates html file displaying TAF output.

    :param date: date of the start of the TAF
    :type date: str
    """
    # File name of html file
    fname = f'{OUT_DIR}/html/taf_output_all.html'

    # Read in existing file, getting 2 lists of lines from the file, split
    # where an extra line is required
    with open(fname, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    first_lines = lines[:-81]
    last_lines = lines[-81:]

    # Edit html file and append/edit the required lines
    first_lines[-1] = first_lines[-1].replace(' selected="selected"', '')
    first_lines.append('                  <option selected="selected" '
                       f'value="{date}">{date}</option>\n')
    last_lines[-15] = last_lines[-15].replace(last_lines[-15][84:92], date)
    last_lines[-25] = last_lines[-25].replace(last_lines[-25][84:92], date)
    last_lines[-35] = last_lines[-35].replace(last_lines[-35][84:92], date)

    # Concatenate the lists together
    new_lines = first_lines + last_lines

    # Re-write the lines to a new file
    with open(fname, 'w', encoding='utf-8') as o_file:
        for line in new_lines:
            o_file.write(line)


def ver_format(taf_str, site_data):
    """
    Organises TAF text into format used for verification.

    Args:
        taf_str: (str): Raw TAF text
        site_data (pandas.DataFrame): IMPROVER data and airport info
    Returns:
        ver_taf: (str): Verification formatted TAF text.
    """
    # Get required strings from dataframe attributes
    issue_dt = site_data.attrs['taf_issue']
    issue_str = issue_dt.strftime('%H%MZ %d/%m/%y')
    sent_str = issue_dt.strftime('%H05')
    icao = site_data.attrs['icao']

    # Start string with required header info
    ver_taf = (f'T {issue_str}                EGRR {sent_str} 0 0 {icao} '
               f'{taf_str}\n')

    return ver_taf
