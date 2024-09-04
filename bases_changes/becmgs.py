"""
Functions used for determining BECMG groups.

Functions:
    all_wind_changes: Determines if change for multiple wind types.
    cld_change_row: Determines if significant cloud change.
    dir_change_row: Determines if significant wind direction change.
    find_becmg: Finds a BECMG group.
    get_becmg_period: Determines BECMG group period.
    get_changes: Determines indices of significant changes.
    get_consistent_index: Checks for consistent change.
    get_first_index: Finds first index of significant change.
    get_new_bases: Determines new base conditions after a BECMG group.
    get_wind_vals: Collects wind data used in base conditions calcs.
    gust_change_row: Determines if significant wind gust change.
    mean_change_row: Determines if significant wind mean change.
    vals_dir: Determines if dir change valid for using IMPROVER data.
    vals_mean: Determines if mean change valid for using IMPROVER data.
    vals_gust: Determines if gust change valid for using IMPROVER data.
    vis_change_row: Determines if significant visibility change.
"""
import copy

import pandas as pd

import bases_changes.bases as ba
import common.calculations as ca
import common.checks as ch
import common.configs as co


def all_wind_changes(row):
    """
    Determines if significant change from base condition for several
    wind changes on row of dataframe.

    Args:
        row (pandas.Series): Row of dataframe
    Returns:
        change (bool): Indicator for whether change significant
    """
    # Define changes to find values for
    wxs = row['wx_changes'].split()

    # Change is True if change criteria met for all wx types
    change = all(row[f'{wx}_vals'] for wx in wxs)

    return change


def cld_change_row(row):
    """
    Determines if cloud change significant from base conditions on row
    of dataframe.

    Args:
        row (pandas.Series): Row of dataframe
    Returns:
        change (str): Indicator for whether change significant
    """
    # Check if cloud change significant
    change = ch.cld_change(row['base_cld_cat'], row['cld_cat'],
                           row['rules_col'])

    return change


def dir_change_row(row):
    """
    Determines if wind direction change significant from base conditions
    on row of dataframe.

    Args:
        row (pandas.Series): Row of dataframe
    Returns:
        change (bool): Indicator for whether change significant
    """
    # Check id direction change significant
    change = ch.dir_change((row['wind_dir'], row['base_wind_dir']),
                           (row['wind_mean'], row['base_wind_mean']),
                           row['rules_col'])

    return change


def find_becmg(becmg_options):
    """
    Finds BECMG group options (if any to find), updating IMPROVER data
    in each case for next time the function is called. IMPROVER data
    should only be at the 50th percentiles as BECMG groups are
    deterministic.

    Args:
        becmg_options (list): List containing options for BECMG groups
    Returns:
        becmg_options (list): Updated list of BECMG groups
        keep_searching (bool): Indication for whether to keep searching
                               for more BECMG group options
    """
    # To prevent for loop iterating through any added options, define
    # the length of becmg options now use to break for loop
    num_options = len(becmg_options)

    # Loop through each of the current BECMG group options, branching
    # out new ones if necessary (default that no new groups to find)
    keep_searching = False
    for ind, option in enumerate(becmg_options):

        # Break for loop if it has moved beyond the original options
        if ind == num_options:
            break

        # Move to next option if no more BECMG groups to be found
        if option['finished']:
            continue

        # Get latest bases and tdf from option
        bases = option['groups'][-1]
        tdf = option['data']

        # Find valid change indices
        bdf, valid_indices = get_changes(tdf, bases)

        # If no change indices found, no more BECMG groups are needed
        if not valid_indices:
            option['finished'] = True
            continue

        # If this point is reached, new base conditions will be created
        # so there is potential for more BECMG groups
        keep_searching = True

        # For wind changes, collect values to calculate base conditions
        wind_vals = get_wind_vals(bdf, valid_indices)

        # Determine next BECMG group period and type (if any)
        period_options, becmg_types = get_becmg_periods(valid_indices, tdf,
                                                        option)

        # Get new base values following BECMG for each option
        for ind, p_option in enumerate(period_options):

            # Determine new base condions following BECMG
            new_bases = get_new_bases(p_option['tdf'], p_option['period'],
                                      becmg_types, bases, wind_vals)

            # Add to current option or create new one if necessary
            if ind == 0:
                option['groups'].append(new_bases)
                option['data'] = p_option['tdf']
                option['score'] += p_option['score']
            else:
                new_option = p_option['option']
                new_option['groups'].append(new_bases)
                new_option['data'] = p_option['tdf']
                new_option['score'] += p_option['score']
                becmg_options.append(new_option)

    return becmg_options, keep_searching


def get_becmg_periods(change_inds, tdf, option):
    """
    Calculates an appropriate BECMG period, with length between accepted
    values of 2 to 4 hours.

    Args:
        change_inds (dict): Indices of BECMG changes by weather type
        tdf (pandas.DataFrame): IMPROVER and airport data
    Return:
        becmg_period (tuple): BECMG group period
        becmg_types (list): Weather types to include in BECMG group
        new_tdf (pandas.DataFrame): Updated IMPROVER data
    """
    # Get first index
    first_wx = min(change_inds, key=change_inds.get)
    becmg_indices = {first_wx: change_inds[first_wx]}

    # Include all changes within 4 hour window (maximum length of BECMG)
    for wx in change_inds:
        if wx != first_wx and change_inds[wx] - change_inds[first_wx] <= 4:
            becmg_indices[wx] = change_inds[wx]

    # Get list of BECMG group weather types
    becmg_types = list(becmg_indices.keys())

    # Find the range that the BECMG group has to cover
    min_index = min(becmg_indices.values())
    max_index = max(becmg_indices.values())

    # Get length of remaining dataframe to limit end of BECMG period
    remaining_taf = len(tdf[tdf['taf_time'] == 'during']) - 1

    # BECMG group length depends on gap between change indices -
    # generally 3 hour periods preferred but allows for rare cases of
    # 4 hour groups (2 hour groups also possible if changes fall at end
    # of TAF)
    if max_index - min_index == 4:
        period_length = 4
    else:
        period_length = 3

    # Get first and last possible BECMG indices based on change indices
    first_first = max(max_index - period_length, 0)
    last_first = min(min_index, remaining_taf)

    # Get all possibilities for BECMG group period
    period_options = []
    for first in range(first_first, last_first + 1):

        # Get end of BECMG period
        second = min(first + period_length, remaining_taf)

        # If BECMG period less than 2 hours, don't use
        if second - first < 2:
            continue

        # Get new dataframe starting from end of BECMG group
        new_tdf = tdf[second:]

        # Define period from chosen indices
        becmg_period = (tdf['time'].iloc[first], tdf['time'].iloc[second])

        # Create copy of options to be updated later
        new_option = copy.deepcopy(option)

        # Score the option based on whether the cange indices sit in the
        # middle of the BECMG group (preferable) or at the edge - lower
        # score is better (avoid dividing by zero by adding 1 to pads)
        start_pad = min_index - first
        end_pad = second - max_index
        max_pad, min_pad = max([start_pad, end_pad]), min([start_pad, end_pad])
        score = (max_pad + 1) / (min_pad + 1)

        # Add options to dictionary
        period_options.append({'period': becmg_period, 'tdf': new_tdf,
                               'option': new_option, 'score': score})

    return period_options, becmg_types


def get_changes(tdf, bases):
    """
    Determines indices at which a BECMG group is required for each
    weather type. tdf should only contain IMPROVER data at 50th
    percentile.

    Args:
        tdf (pandas.DataFrame): IMPROVER data
        bases (dict): Current base conditions
    Return:
        valid_indices (dict): Indices at which BECMG group required
        bdf (pandas.DataFrame): IMPROVER data with base conditions and
                                other columns added
    """
    # BECMG group has to be at least 2 hours so do not search for one if
    # less than 2 hours left in TAF
    hrs_remaining = len(tdf[tdf['taf_time'] == 'during']) - 1
    if hrs_remaining < 2:
        return None, None

    # Add required bases to copy of IMPROVER data dataframe
    bdf = tdf.copy()
    for base_type in ['wind_dir', 'wind_mean', 'wind_gust', 'vis_cat',
                      'cld_cat']:
        bdf[f'base_{base_type}'] = bases[base_type]

    # Reset index of dataframe
    bdf.reset_index(drop=True, inplace=True)

    # Find significant changes from base conditions
    change_funcs = {'wind_dir': dir_change_row, 'wind_mean': mean_change_row,
                    'wind_gust': gust_change_row, 'vis': vis_change_row,
                    'cld': cld_change_row}
    for wx, change_func in change_funcs.items():
        bdf[f'{wx}_changes'] = bdf.apply(change_func, axis=1)

    # For direction and mean, all changes covered in BECMG groups, so
    # use index of first change from base conditions
    dir_index = get_first_index(bdf, 'wind_dir', hrs_remaining)
    mean_index = get_first_index(bdf, 'wind_mean', hrs_remaining)

    # Gusts, visibility/sig_wx and cloud need change to be consistent
    # for 4 hours to allow a BECMG group
    gust_index = get_consistent_index(bdf, 'wind_gust', hrs_remaining)
    vis_index = get_consistent_index(bdf, 'vis', hrs_remaining)
    cld_index = get_consistent_index(bdf, 'cld', hrs_remaining)

    # Get all valid change indices
    change_indices = {'wind_dir': dir_index, 'wind_mean': mean_index,
                      'wind_gust': gust_index, 'vis': vis_index,
                      'cld': cld_index}
    valid_indices = {key: val for key, val in change_indices.items()
                     if val <= hrs_remaining}

    # Don't need to continue if no valid changes found
    if not valid_indices:
        return None, None

    # multiple change types can be combined into same BECMG group if
    # within 4 hours of each other
    valid_indices = {key: val for key, val in valid_indices.items()
                     if val <= min(valid_indices.values()) + 4}

    return bdf, valid_indices


def get_consistent_index(bdf, wx, hrs_remaining):
    """
    Finds index of a consistent significant change (at least 4 hours).

    Args:
        bdf (pandas.DataFrame): IMPROVER data with base conditions
        wx (str): Weather type
        hrs_remaining (int): Number of hours remaining in the TAF
    Return:
        change_index (int): Index of first consistent change
    """
    # Start with default of no consistent change by assigning high index
    change_index = hrs_remaining + 1

    # Loop through bools indicating significant changes and break for
    # if consistent change found
    for ind, change in enumerate(bdf[f'{wx}_changes']):

        # Check if next 4 values are of the same sign, indicating a
        # consistent change of the same type (increase or decrease)
        next_4_vals = bdf[f'{wx}_changes'][ind: ind + 4]
        if change and any([all(val > 0 for val in next_4_vals),
                           all(val < 0 for val in next_4_vals)]):
            change_index = ind
            break

    return change_index


def get_first_index(bdf, wx, hrs_remaining):
    """
    Finds first index of a significant change.

    Args:
        bdf (pandas.DataFrame): IMPROVER data with base conditions
        wx (str): Weather type
        hrs_remaining (int): Number of hours remaining in the TAF
    Return:
        change_index (int): Index of first change
    """
    # Find index first non-False value in changes column (if any)
    change_index = bdf.where(bdf[f'{wx}_changes']).first_valid_index()

    # Give high index if no changes (to avoid confusing None with 0)
    if not change_index:
        change_index = hrs_remaining + 1

    return change_index


def get_new_bases(site_data, period, wx_types, old_bases, wind_vals):
    """
    Determines new base conditions after a BECMG group.

    Args:
        site_data (pandas.DataFrame): IMPROVER data
        period (list); Start and end of BECMG group period
        wx_types (list): Weather types required
        old_bases (dict): Previous base conditions
        wind_vals (pd.DataFrame): Wind values to use
    Returns:
        bases (dict): New base conditions
    """
    # Need 30th and 50th percentiles
    tdf_taf = site_data[site_data['taf_time'] == 'during']
    tdf_30 = tdf_taf[(tdf_taf['percentile'] == 30)]
    tdf_50 = tdf_taf[(tdf_taf['percentile'] == 50)]

    # To add base values to
    bases = {}

    # If there are wind changes, use pre-determined wind values to
    # ensure consistency
    if not wind_vals.empty:

        # Get base wind direction and wind mean
        bases['wind_dir'], num_dirs = ba.get_base_dir(wind_vals)
        base_mean, num_means = ba.get_base_mean(wind_vals, becmg=True)

        # Subset wind values using same number of values as used with
        # base mean or base direction, whichever is smaller
        end_index = min(num_means, num_dirs)
        s_wind_vals = wind_vals[:end_index]
        bases['wind_mean'], bases['wind_gust'] = ba.get_base_gust(base_mean,
                                                                  s_wind_vals)

    # Get base cloud base values if needed
    if 'cld' in wx_types:
        bases['clds'], bases['cld_cat'] = ba.get_base_cld(tdf_30, tdf_50)

    # Get vis/sig wx base values if needed
    if 'vis' in wx_types:
        (bases['vis'], bases['vis_cat'], bases['sig_wx'],
         bases['implied_sig_wx']) = ba.get_base_vis_wx(tdf_50, old_bases)

    # Add in old base values that haven't changed
    for wx, val in old_bases.items():
        if wx not in bases:
            bases[wx] = val

    # Determine if CAVOK/NSC/NSW needed
    bases['cavok'] = ca.use_cavok(bases['vis'], bases['clds'],
                                  bases['implied_sig_wx'], wx_types,
                                  prev_wx = old_bases['implied_sig_wx'])

    # Add in BECMG group info
    bases['change_type'] = 'BECMG'
    bases['change_period'] = period
    bases['wx_changes'] = wx_types

    return bases


def get_wind_vals(bdf, valid_indices):
    """
    Whenever new base conditions are calculated for any wind type
    (direction, mean or gust), all wind types are considered for base
    conditions even if only one type of change (e.g. if a BECMG group is
    required for a change in mean and gust, but not direction, a new
    base direction also needs to be calculated). This function defines
    which IMPROVER wind data to use for the calculation of base
    conditions to ensure TAF consistency.

    Args:
        bdf (pandas.DataFrame): IMPROVER data with base conditions
        valid_indices (dict): Indices at which BECMG group required
    Return:
        change_vals (pandas.DataFrame): Wind IMPROVER data to be used
                                        for determining base conditions
    """
    # Get just wind change indices
    wind_indices = {wx: ind for wx, ind in valid_indices.items()
                    if wx in ['wind_dir', 'wind_mean', 'wind_gust']}

    # Return empty dataframe if no wind changes
    if not wind_indices:
        return pd.DataFrame([])

    # Otherwise, find indices where changes found for each wind type
    val_functions = {'wind_dir': vals_dir, 'wind_mean': vals_mean,
                     'wind_gust': vals_gust}
    for wx in wind_indices:
        bdf[f'first_{wx}'] = bdf[f'{wx}'].iloc[wind_indices[wx]]
        bdf[f'first_{wx}_mean'] = bdf['wind_mean'].iloc[wind_indices[wx]]
        bdf[f'{wx}_change_time'] = bdf['time'].iloc[wind_indices[wx]]
        bdf[f'{wx}_vals'] = bdf.apply(val_functions[wx], axis=1)

    # Find where criteria met for all weather types
    bdf['wx_changes'] = ' '.join(wind_indices.keys())
    bdf['all_vals'] = bdf.apply(all_wind_changes , axis=1)

    # It is preferrable to use values from later in the BECMG period so
    # base conditions reflect values after BECMG period
    max_ind = max(wind_indices.values())
    if bdf['all_vals'].iloc[max_ind]:
        change_vals = bdf[bdf['all_vals']][co.PARAM_NAMES['wind'] +
                                           ['rules_col']]

    # If not all changes are valid at the latest change index, this
    # implies earlier changes are short-lived - in this case, (which
    # should be rare), ignore earlier changes to make for a simpler TAF.
    # By taking just the values at the latest index, this will include a
    # wind group with some significant wind changes and other
    # non-significant changes - this is fine under TAF rules and may
    # cover any unreported significant changes. Keep all types in
    # valid_changes though to ensure BECMG period covers these.
    else:
        change_vals = bdf.iloc[max_ind][co.PARAM_NAMES['wind'] + ['rules_col']]

    # Ensure gusts are not reported if no changes found by making gusts
    # the same as means
    if not 'wind_gust' in wind_indices:
        change_vals['wind_gust'] = change_vals['wind_mean']

    # Ensure change_vals is DataFrame (can be a Series if single wind
    # values) - have to take transpose to keep column names
    if isinstance(change_vals, pd.Series):
        change_vals = change_vals.to_frame().T

    return change_vals


def gust_change_row(row):
    """
    Determines if gust change significant from base conditions on row
    of dataframe.

    Args:
        row (pandas.Series): Row of dataframe
    Returns:
        change (str or bool): Indicator for whether change significant
    """
    # Check if change significant
    change = ch.gust_change(row['base_wind_gust'], row['wind_gust'],
                            row['base_wind_mean'], row['wind_mean'],
                            becmg=True)

    return change


def mean_change_row(row):
    """
    Determines if wind mean change significant from base conditions on
    row of dataframe.

    Args:
        row (pandas.Series): Row of dataframe
    Returns:
        change (str or bool): Indicator for whether change significant
    """
    # Check if change significant
    change = ch.mean_change(row['base_wind_mean'], row['wind_mean'],
                            row['rules_col'])

    return change


def vals_dir(row):
    """
    Determines if wind direction is significantly different from base
    conditions but not significantly different from first significant
    change from base conditions and at or after first change.

    Args:
        row (pandas.Series): Row of dataframe
    Returns:
        change (bool): Indicator for whether to treat change as valid
    """
    # Get 3 bools - whether change from base conditions, change from
    # first change and at or after first change
    base_change = ch.dir_change((row['wind_dir'], row['base_wind_dir']),
                                (row['wind_mean'], row['base_wind_mean']),
                                row['rules_col'])
    first_change = ch.dir_change((row['wind_dir'], row['first_wind_dir']),
                                 (row['wind_mean'],
                                  row['first_wind_dir_mean']),
                                 row['rules_col'])
    after_first = row['time'] >= row['wind_dir_change_time']

    # Change is True if direction significantly different from base, not
    # significantly different from (and at or after) first change
    change = all([base_change, not first_change, after_first])

    return change


def vals_mean(row):
    """
    Determines if wind mean is significantly different from base
    conditions but not significantly different from first significant
    change from base conditions and at or after first change.

    Args:
        row (pandas.Series): Row of dataframe
    Returns:
        change (bool): Indicator for whether to treat change as valid
    """
    # Get 3 bools - whether change from base conditions, change from
    # first change and at or after first change
    base_change = ch.mean_change(row['base_wind_mean'], row['wind_mean'],
                                 row['rules_col'])

    first_change = ch.mean_change(row['first_wind_mean'], row['wind_mean'],
                                  row['rules_col'])
    after_first = row['time'] >= row['wind_mean_change_time']

    # Change is True if direction significantly different from base, not
    # significantly different from (and at or after) first change
    change = all([base_change, not first_change, after_first])

    return change


def vals_gust(row):
    """
    Determines if wind gust is significantly different from base
    conditions but not significantly different from first significant
    change from base conditions and at or after first change.

    Args:
        row (pandas.Series): Row of dataframe
    Returns:
        change (bool): Indicator for whether to treat change as valid
    """
    # Get 3 bools - whether change from base conditions, change from
    # first change and at or after first change
    base_change = ch.gust_change(row['base_wind_gust'], row['wind_gust'],
                                 row['base_wind_mean'], row['wind_mean'],
                                 becmg=True)
    first_change = ch.gust_change(row['first_wind_gust'], row['wind_gust'],
                                  row['first_wind_gust_mean'],
                                  row['wind_mean'], becmg=True)
    after_first = row['time'] >= row['wind_gust_change_time']

    # Change is True if direction significantly different from base, not
    # significantly different from (and at or after) first change
    change = all([base_change, not first_change, after_first])

    return change


def vis_change_row(row):
    """
    Determines if visibility change significant from base conditions on
    row of dataframe.

    Args:
        row (pandas.Series): Row of dataframe
    Returns:
        change (bool): Indicator for whether change significant
    """
    # Check if change significant
    change = ch.vis_change(row['base_vis_cat'], row['vis_cat'])

    return change
