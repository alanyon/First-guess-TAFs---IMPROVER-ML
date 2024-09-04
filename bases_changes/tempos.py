"""
Functions to create TEMPO and PROB groups.

Functions:
    cover_all_changes: Creates single change group covering all changes.
    define_cols: Defines change categories columns.
    extend_period: Extends all groups by 1 hour if possible.
    extend_possible: Determines if change group extension possible.
    get_chg_values: Determines forecast values for change group.
    get_param_bases: Extracts base values relevant to wx type.
    get_suitable_values: Chooses values to use in change group.
    one_percentile: Creates single percentile change group.
    optimal_changes: Finds best change group options.
    organise_groups: Organisise TEMPO/PROB groups into correct format.
    over_cats: Defines variables for overlapping groups.
    overlapping_groups: Finds option using overlapping change groups.
    param_tempos: Finds TEMPO/PROB groups for a given weather type.
    period_tempos: Finds TEMPO/PROB groups between periods.
    update_options: Performs tests, adds option to options dictionary.
    values_bases: Collects values and base conditions relevant to group.
"""
import copy
import itertools
from datetime import timedelta

import numpy as np
import pandas as pd

import bases_changes.bases as ba
import bases_changes.tempo_calcs as tc
import common.calculations as ca
import common.configs as co

# Significant change functions
CHANGE_FUNCS = {'wind': tc.gust_change_row, 'vis': tc.vis_change_row,
                'cld': tc.cld_change_row}


def cover_all_changes(options, perc, unq_changes, unq_cats, change_inds,
                      cat_weights, stdf):
    """
    Creates a change group option in which a single percentile is used
    to cover all category changes (either increased or decreased) by
    just forecasting the most extreme category. Updates options
    dictionary if necessary.

    Args:
        options (dict): Current change group options and their scores
        perc(str): Percentile being used to cover changes
        unq_changes (pandas.DataFrame): All non-duplicated changes
        unq_cats (np.array): Unique non-zero categories to cover
        change_inds (pandas.RangIndex): Indices where changes forecast
        cat_weights (dict): Category changes and their weights
        stdf (pandas.DataFrame): Subset of IMPROVER data
    Returns:
        options (dict): Updated change group options and scores
    """
    # Define relevant percentiles and most extreme change category,
    # depending on change type
    if perc in ['30', '40', '50d']:
        x_cat = np.min(unq_cats)
    elif perc in ['70', '60', '50i']:
        x_cat = np.max(unq_cats)

    # If changes are predicted and percentile is a relevant one...
    if change_inds.any():

        # Take copy of main changes dataframe and change all other
        # percentile values to 0
        x_changes = unq_changes.copy()
        x_changes[x_changes.drop('time', axis=1).columns]=0

        # Change all hours where changes are forecast to most extreme
        # category
        x_changes[perc].iloc[change_inds] = x_cat

        # Test if option passes TAF rule tests, adding to options if so
        options = update_options(options, unq_changes, x_changes, cat_weights,
                                 stdf)

    return options


def define_cols(change_type):
    """
    Defines the columns to use to find change categories.

    Args:
        change_type (str): Change type (increase or decrease)
    Returns:
        change_percs (list): Column names to use for PROB groups
        changes_det (str): Column name to use for deterministic group
                           (i.e. TEMPO group)
    """
    # Column_names to use for decreased changes
    if change_type == 'decrease':
        change_percs, changes_det = ['30', '40'], '50d'

    # Column_names to use for increased changes
    elif change_type == 'increase':
        change_percs, changes_det = ['70', '60'], '50i'

    return change_percs, changes_det


def extend_period(stdf, groups_df, unq_changes, direction, check_length=False):
    """
    Extends all PROB/TEMPO groups, either by 1 hour back in time or 1
    hours forward in time if possible.

    Args:
        stdf (pandas.DataFrame): IMPROVER (and other) data
        groups (dict): TEMPO/PROB groups
        groups_df (pandas.DataFrame): PROB/TEMPO groups category changes
        direction (str): Direction to extend group (forward/back)
        check_length (bool): Indicator for whether to ensure length of
                             group is at least 2 hours
    Returns:
        groups (list): Updated list of TEMPO/PROB groups
    """
    # Get groups based on dataframe
    groups = get_chg_values(stdf, groups_df, unq_changes)

    # Iterate through each change group
    for group_key, group in groups.items():

        # If period length being checked, move on if period >= 2 hours
        start, end = group['period']
        if check_length and (end - start).total_seconds() / 3600 >= 2:
            continue

        # Get new dt to test adding to dataframe (N.B. PROB/TEMPO
        # periods end 1 hour after latest change in dataframe so new dt
        # for extending forward is dt at the end of the current period)
        if direction == 'back':
            new_dt = group['period'][0] - timedelta(hours=1)
        elif direction == 'forward':
            new_dt = group['period'][1]

        # Can't extend if new dt is out of searching period
        if not groups_df['time'].isin([new_dt]).any():
            continue

        # Get stdf at time and group percentile and convert to Series
        t_stdf = stdf[stdf['time'] == new_dt]
        tp_stdf = t_stdf[t_stdf['percentile'] == int(group['perc'][:2])]
        tp_row = tp_stdf.squeeze()

        # Take copy of dataframe to avoid overwriting
        ext_df = groups_df.copy()

        # Can't extend if non-zero category already in new time position
        if ext_df.loc[ext_df.time == new_dt, group['perc']].values[0] != 0:
            continue

        # If new time category is zero, change category
        ext_df.loc[ext_df.time == new_dt, group['perc']] = group['cat']

        # Don't make changes if adding change creates an illegal TAF
        if not tc.changes_tests(ext_df):
            continue

        # If possible to extend, commit to changing groups df
        if extend_possible(group_key, group, groups, tp_row,
                           stdf.attrs['wx_type']):
            groups_df = ext_df

    return groups_df


def extend_possible(group_key, group, groups, tp_row, wx_type):
    """
    Compares values against base conditions to determine whether
    extension of change group period would be valid.

    Args:
        group_key (str): Key identifying change group
        group (dict): Change group info
        groups (list): all current change groups
        tp_row (pandas.Series): row of IMPROVER data/base conditions df
        wx_type (str): Weather type (wind, vis or cld)
    Returns:
        extend (bool): Indicator for whether extension is possible
    """
    # Weather parameters needed dependant on wx type
    params = co.PARAM_NAMES[wx_type]

    # Change values in series to those chosen in group
    for param in params:
        tp_row[param] = group['values'][param]

    # Loop through all groups
    for o_group_key, o_group in groups.items():

        # Move on if same group
        if group_key == o_group_key:
            continue

        # Look for other overlapping groups, adding as extra base values
        # if found
        o_bases = tc.get_other_bases(group, o_group)
        if o_bases:
            for param in params:
                tp_row[f'base_{param}_3'] = o_bases[f'{param}'][0]
            break

    # If group values still represent a significant change from base
    # conditions, it is possible to extend
    extend = CHANGE_FUNCS[wx_type](tp_row)

    return extend


def get_chg_values(stdf, groups_df, changes_df):
    """
    Determines suitable forecast values for a change group and collects
    TEMPO/PROB group information into dictionary.

    Args:
        stdf (pandas.DataFrame): Subset of IMPROVER data
        groups_df (pandas.DataFrame): Categories for change groups.
        changes_df (pandas.DataFrame): All unique changes.
    Returns:
        groups (list): List of TEMPO/PROB groups
    """
    # Get change groups from groups dataframe
    groups = tc.get_groups(groups_df)

    # loop through each group
    for group in groups.values():

        # Get relevant values and base conditions
        values, bases = values_bases(changes_df, stdf, group)

        # Choose suitable set of values to use
        group['values'] = get_suitable_values(values, bases, group['cat'],
                                              stdf)

    # For overlapping groups, ensure group with most extreme forecast is
    # significantly different from group with less extreme forecast -
    # only necessary for wind and vis as cloud purely category-driven
    for group, o_group in itertools.product(groups.values(), repeat=2):

        # Look for other bases
        o_bases = tc.get_other_bases(group, o_group)

        # If no extra base values found, move to next iteration
        if not o_bases:
            continue

        # Update values if necessary
        if stdf.attrs['wx_type'] == 'wind':
            group['values'] = tc.adjust_wind(group['values'],
                                             o_bases, group['cat'])
        elif stdf.attrs['wx_type'] == 'vis':
            group['values'] = tc.update_tempo_wx(group['values'],
                                                 group['cat'], o_bases,
                                                 stdf.attrs['rules'])

    # Now all values finalised, choose suitable change group type (i.e.
    # TEMPO or no TEMPO)
    for group in groups.values():
        group['change_type'] = tc.tempo_no_tempo(group, changes_df,
                                                 stdf.attrs['wx_type'])

    return groups


def get_param_bases(base_dict, wx_type):
    """
    Extracts base conditions relevant to wx type.

    Args:
        base_dict (dict): All base conditions
        wx_type (str): Weather type (wind, vis or cld)
    Returns:
        p_bases (dict): Base conditions revel=vant to wx type
    """
    # Return None if no base conditions
    if base_dict is None:
        return None

    # Cloud base conditions
    if wx_type == 'cld':
        p_bases = {'cld_3': base_dict['clds']['cld_3'],
                   'cld_5': base_dict['clds']['cld_5'],
                   'cld_cat': base_dict['cld_cat']}

    # Visibility and sig wx base conditions
    elif wx_type == 'vis':
        p_bases = {param: base_dict[param]
                   for param in co.PARAM_NAMES[wx_type][:2]}
        p_bases['sig_wx'] = base_dict['implied_sig_wx']

    # Wind base conditions
    elif wx_type == 'wind':
        p_bases = {param: base_dict[param]
                   for param in co.PARAM_NAMES[wx_type]}

    return p_bases


def get_suitable_values(values, bases, change_cat, stdf):
    """
    Finds appropriate values to use in change groups, tweaking to ensure
    valid changes if necessary.

    Args:
        values (dict): Possible values
        bases (dict): Base conditions
        change_cat (int): Change category
        stdf (pandas.DataFrame): Subset of IMPROVER data
    Returns:
        suitable_values (dict): Suitable values to use
    """
    # For wind groups
    if stdf.attrs['wx_type'] == 'wind':

        # Get appropriate values for each parameter
        best_values = {}
        for param in values:

            # Mean calculated differently for wind direction
            if param == 'wind_dir':
                gdf = pd.DataFrame(values)
                gdf.attrs['rules'] = stdf.attrs['rules']
                mean_dir, _ = ba.get_base_dir(gdf)
                best_values[param] = mean_dir

            # Otherwise, take mean and round
            else:
                best_values[param] = int(round(np.mean(values[param]), 0))

        # Ensure wind values still suitable change (i.e. after
        # means taken), then add to group dictionary
        suitable_values = tc.adjust_wind(best_values, bases, change_cat)

    # For vis groups
    elif stdf.attrs['wx_type'] == 'vis':

        # Get appropriate vis and sig wx values
        suitable_values = tc.get_vis_wx(values, bases, change_cat,
                                        stdf.attrs['rules'])

    # For cloud groups
    elif stdf.attrs['wx_type'] == 'cld':

        # Get appropriate cloud values
        suitable_values = tc.get_clds(values, change_cat, stdf.attrs['rules'])

    return suitable_values


def one_percentile(options, unq_changes, perc, cat_weights, stdf):
    """
    Creates change group option in which a single percentile's forecasts
    are used and updates options dictionary if necessary.

    Args:
        options (dict): Current change group options and their scores
        unq_changes (pandas.DataFrame): All non-duplicated changes
        perc (str): Percentile from which change group will be based on
        cat_weights (dict): Category changes and their weights
        stdf (pandas.DataFrame): Subset of IMPROVER data
    Returns:
        options (dict): Updated change group options and scores
    """
    # Take copy of main changes dataframe and change all other
    # percentile changes to zero (remove time column temporarily)
    changes_perc = unq_changes.copy()
    cols = [col for col in changes_perc.columns if col not in ['time', perc]]
    changes_perc.loc[:, cols] = 0

    # Test against TAF rules, calculate scores and add to options
    options = update_options(options, unq_changes, changes_perc, cat_weights,
                             stdf)

    return options


def optimal_changes(stdf):
    """
    Finds best change group options for representing forecast changes.

    Args:
        stdf (pandas.DataFrame): Subset of IMPROVER and airport data
    Returns:
        best_option (pandas.DataFrame): Best change group option
    """
    # Get smaller dataframe just containing changes at all percentiles
    unq_changes = tc.get_changes(stdf)

    # Take copy with time column removed
    unq_no_time = unq_changes.drop(['time'], axis=1)

    # Get category weights based on frequency of forecasts - used for
    # calculating scores later
    cat_weights, unq_cats = tc.get_weights(unq_no_time)

    # For adding change group options to
    options = []

    # Test against TAF rules, calculate scores and add to options
    option = unq_changes.copy()
    options = update_options(options, unq_changes, option, cat_weights, stdf)

    # Indices of decreased and increased changes
    dec_inds = unq_no_time[(unq_no_time < 0).any(axis=1)].index
    inc_inds = unq_no_time[(unq_no_time > 0).any(axis=1)].index

    # Find options that use a single percentile column
    for perc in unq_no_time.columns:

        # Create simple option just using changes from one percentile,
        # calculating scores and adding to options if necessary
        options = one_percentile(options, unq_changes, perc, cat_weights, stdf)

        # Create options that cover extreme changes using single
        # percentile if possible
        options = cover_all_changes(options, perc, unq_changes, unq_cats,
                                    dec_inds, cat_weights, stdf)
        options = cover_all_changes(options, perc, unq_changes, unq_cats,
                                    inc_inds, cat_weights, stdf)

    # Create options for overlapping groups (TEMPO and PROB) if possible
    options = overlapping_groups(options, unq_cats[unq_cats < 0], unq_changes,
                                 dec_inds, cat_weights, 'decrease', stdf)
    options = overlapping_groups(options, unq_cats[unq_cats > 0], unq_changes,
                                 inc_inds, cat_weights, 'increase', stdf)

    # If no viable options, return empty dictionary
    if not options:
        return {}

    # Choose option with highest overall score as best option
    best_groups = max(options, key=lambda x: x['score'])['groups']

    return best_groups


def organise_groups(groups, main_base, wx_type, end_dt):
    """
    Organises change groups into correct format for writing TAF later.

    Args:
        groups (dict): TEMPO/PROB groups
        main_base (dict): Main base conditions
        wx_type (str): Weather type (wind/vis/cld)
        end_dt (cftime): End of seach period time
    Returns:
        probs_tempos (list): Groups collected in required format
    """
    # To add PROB/TEMPO groups to
    probs_tempos = []

    # Loop through all found groups
    for group in groups.values():

        # Start with base values as default
        change_group = copy.deepcopy(main_base)

        # Update changed parameters
        for param in co.PARAM_NAMES[wx_type]:

            # Slightly different format for cld values
            if wx_type == 'cld' and param != 'cld_cat':
                change_group['clds'][param] = group['values'][param]
            else:
                change_group[param] = group['values'][param]

        # Also need implied wx for vis
        if wx_type == 'vis':
            change_group['implied_sig_wx'] = group['values']['implied_sig_wx']

        # Also need to add in 1 and 8 okta cloud for cloud - keep these
        # the same as default for now
        elif wx_type == 'cld':
            change_group['clds']['cld_1'] = group['values']['cld_3']
            change_group['clds']['cld_8'] = 5000

        # Add change period
        change_group['change_period'] = group['period']

        # Update wx changes
        change_group['wx_changes'] = [wx_type]

        # Update change group type
        change_group['change_type'] = group['change_type']

        # Add in end_dt to indicated time of next BECMG group/end of TAF
        change_group['end_dt'] = end_dt + timedelta(hours=1)

        # Determine if CAVOK/NSC/NSW needed
        change_group['cavok'] = ca.use_cavok(
            change_group['vis'], change_group['clds'],
            change_group['implied_sig_wx'], change_group['wx_changes'],
            prev_wx = main_base['implied_sig_wx'])

        # Append to probs_tempos list
        probs_tempos.append(change_group)

    return probs_tempos


def overlapping_groups(options, change_cats, unq_changes, change_inds,
                       cat_weights, change_type, stdf):
    """
    Finds change group option using overlapping groups (e.g. TEMPO and
    PROB30 TEMPO) and updates options dictionary if necessary.

    Args:
        options (dict): Current change group options and their scores
        change_cats (list): Change categories forecast
        unq_changes (pandas.DataFrame): Non-duplicated forecast changes
        change_inds (pandas.RangeIndex): Indices where changes forecast
        cat_weights (dict): Category changes and their weights
        change_type (str): Type of change (increase or decrease)
        stdf (pandas.DataFrame): Subset of IMPROVER data
    Returns:
        options (dict): Updated change group options and scores
    """
    # Overlapping groups only possible when at least 2 change categories
    if not change_inds.any() or len(change_cats) < 2:
        return options

    # Get all percentile columns (exclude time)
    p_cols = unq_changes.drop('time', axis=1).columns

    # Determine categories to use
    x_cat, l_cats_x_inds = over_cats(change_type, change_cats, change_inds,
                                     unq_changes, p_cols)

    # Define columns to use - depends on change type
    changes_percs, changes_det = define_cols(change_type)

    # Test overlapping with PROB30 and PROB40, covering range of less
    # extreme categories
    for perc, l_cat in itertools.product(changes_percs, l_cats_x_inds):

        # Only use l_cat if some forecasts in that category
        if not (unq_changes==l_cat).any(axis=None):
            continue

        # Create option for each extreme indices option
        for x_inds in l_cats_x_inds[l_cat]:

            # Get copy of unique changes filled with zeros
            changes_over = unq_changes.copy()
            changes_over[p_cols] = 0

            # Cover whole period with least extreme category
            changes_over[changes_det].iloc[change_inds] = l_cat

            # Change relevent indices to most severe category
            changes_over[perc].iloc[x_inds] = x_cat

            # If tests passed, add to options
            options = update_options(options, unq_changes, changes_over,
                                     cat_weights, stdf)

    return options


def over_cats(change_type, change_cats, change_inds, unq_changes, p_cols):
    """
    Determines categories to use for overlapping groups. There can be
    many possible options for this if there are many categories to
    cover. Here, the extreme category is fixed but there coan be
    multiple options for the less extreme category. For each of the less
    extreme category options, the indices used for the extreme category
    are not only those of the most extreme category, but also all of the
    intermediate categories so that all categories are covered. E.g. for
    a decreasing change where categories range from 2 to 5, x_cat would
    be 2 and l_cat possibilities would be 3, 4 and 5. For an l_cat of 5,
    x_inds would be all indices where the change category is 4 or lower.

    Args:
        change_type (str): Type of change (increase or decrease)
        change_cats (list): Change categories forecast
        change_inds (pandas.RangeIndex): Indices where changes forecast
        unq_changes (pandas.DataFrame): Non-duplicated forecast changes
        p_cols (list): Percentile columns
    Returns:
        x_cat (int): Most extreme category to use in overlapping PROB
                     group
        l_cats_x_inds (dict): Possible options for less extreme category
                              along with indices to use for extreme
                              category for each option
    """
    # Variables for decreased changes
    if change_type == 'decrease':

        # Extreme category and indices
        x_cat = np.min(change_cats)

        # Possible options for lower category
        l_cats = range(np.max(change_cats), x_cat, -1)

        # Possible options for indices of most extreme group for each
        # of the less extreme categories
        l_cats_x_inds = {}
        for l_cat in l_cats:

            # Collect possible indices for each category between x_cat
            # and l_cat
            x_inds_options = []
            for cat in range(x_cat, l_cat):

                # Collect indices for category being considered as well
                # as indices of more extreme categories
                x_inds = [ind for ind in change_inds
                          if unq_changes.iloc[ind][p_cols].min() <= cat]
                x_inds_options.append(x_inds)

            # Add to dictionary using l_cat as key and x_inds_options as
            # value
            l_cats_x_inds[l_cat] = x_inds_options

    # Variables for increased changes
    elif change_type == 'increase':

        # Extreme category and indices
        x_cat = np.max(change_cats)

        # Possible options for lower category
        l_cats = range(np.min(change_cats), x_cat)

        # Possible options for indices of most extreme group for each
        # of the less extreme categories
        l_cats_x_inds = {}
        for l_cat in l_cats:
            x_inds_options = []
            for cat in range(x_cat, l_cat, -1):

                # Collect indices for category being considered as well
                # as indices of more extreme categories
                x_inds = [ind for ind in change_inds
                          if unq_changes.iloc[ind][p_cols].max() >= cat]
                x_inds_options.append(x_inds)

            # Add to dictionary using l_cat as key and x_inds_options as
            # value
            l_cats_x_inds[l_cat] = x_inds_options

    return x_cat, l_cats_x_inds


def param_tempos(tdf, change_groups, wx_type):
    """
    Finds TEMPO or PROB groups for a weather type (wind, vis/wx, gust).

    Args:
        tdf (pandas.DataFrame): IMPROVER data
        change_groups (list): Base conditions and change groups
        wx_type: (str): Weather type of TEMPO/PROB groups to look for
    Returns:
        change_groups (list): Updated base conditions and change groups
    """
    # Add weather type as attribute to dataframe
    tdf.attrs['wx_type'] = wx_type

    # Get base conditions and BECMG groups relevant to weather type
    wx_changes = [grp for grp in change_groups if grp['change_type'] == 'base'
                  or any(wx_type in chg for chg in grp['wx_changes'])]

    # Iterate through wx change groups
    for ind, change in enumerate(wx_changes):

        # If not the last group, need to consider following BECMG group
        if ind != len(wx_changes) - 1:

            # Define change period and following change period
            change_period = change['change_period']
            next_change_period = wx_changes[ind + 1]['change_period']

            # Special case if main base is initial base conditions and
            # BECMG group starts at start of TAF - no room for
            # PROB/TEMPO groups here so move to next iteration
            if change_period[0] == next_change_period[0]:
                continue

            # End of period to look for PROB/TEMPO groups is 1 hour
            # before start of next BECMG period
            end_dt = next_change_period[0] - timedelta(hours=1)

        # If no following BECMG groups, end dt 1 hour before end of TAF
        else:
            end_dt = tdf['time'].unique()[-2]

        # Main base conditions defined in change group - take copy to
        # avoid overwriting later
        main_base = copy.deepcopy(change)

        # If BECMG group, need to consider earlier base conditions to
        # allow groups to overlap with BECMG group if necessary
        if ind != 0:
            early_base = copy.deepcopy(wx_changes[ind - 1])

        # If no more BECMG groups follow, ignore early_base
        else:
            early_base = None

        # Look for PROB/TEMPO groups over this period
        probs_tempos = period_tempos(main_base, early_base, end_dt, tdf)

        # Add in main base info for each group and fix probs for
        # overlapping groups (for merging groups later)
        for group in probs_tempos:

            # Add bases
            group['main_bases'] = change
            group['early_bases'] = early_base

            # Get bool indicating whether prob should be fixed (i.e. for
            # overlapping groups)
            group['fix_prob'] = tc.get_fix_prob(group, probs_tempos)

        # If any groups found, add to change_groups list
        change_groups += probs_tempos

    return change_groups


def period_tempos(main_base, early_base, end_dt, tdf):
    """
    Finds TEMPO or PROB groups between base/BECMG periods.

    Args:
        main_base (dict): Main base conditions
        early_base (dict): Base conditions before BECMG group
        end_dt (cftime): End of seach period time
        tdf (pandas.DataFrame): IMPROVER and airport data
    Returns:
        probs_tempos (list): Change groups found
    """
    # Get start of searching period
    start_dt = main_base['change_period'][0]

    # Get end of BECMG period, if any
    if early_base:
        becmg_end = main_base['change_period'][1]
    else:
        becmg_end = main_base['change_period'][0]

    # Subset IMPROVER data
    stdf = tdf.loc[(start_dt <= tdf['time']) & (tdf['time'] <= end_dt)]

    # Get required base conditions depending on wx type
    main_p_bases = get_param_bases(main_base, tdf.attrs['wx_type'])
    early_p_bases = get_param_bases(early_base, tdf.attrs['wx_type'])

    # Make new columns with base values
    for param in main_p_bases:

        # Add main base values
        stdf[f'base_{param}'] = main_p_bases[f'{param}']

        # Include second base means and gusts during BECMG period
        # if any - TEMPO/PROB group needs to be significantly different
        # from both sets of base conditions during this period
        if early_p_bases:
            new_col = stdf[f'base_{param}'].where(
                stdf.time >= becmg_end, other=early_p_bases[param])
        else:
            new_col = stdf[f'base_{param}']
        stdf[f'base_{param}_2'] = new_col

        # Add the possibility of 3rd set of base conditions for
        # overlapping TEMPO/PROB groups - make same as main base for now
        stdf[f'base_{param}_3'] = main_p_bases[f'{param}']

    # Find significant changes by comparing to base condtions
    stdf['change'] = stdf.apply(CHANGE_FUNCS[tdf.attrs['wx_type']], axis=1)

    # If no significant changes found, no TEMPO/PROB groups necessary
    if stdf['change'].sum() == 0:
        return []

    # Get dataframe of optimal changes for each percentile that will be
    # used to generate change groups (add weather type to stdf metadata)
    groups = optimal_changes(stdf)

    # Finally, organise change group info into correct format
    probs_tempos = organise_groups(groups, main_base, stdf.attrs['wx_type'],
                                   end_dt)

    return probs_tempos


def update_options(options, unq_changes, option, cat_weights, stdf):
    """
    Checks if change group option passes TAF rules tests and (if so)
    adds to options dictionary.

    Args:
        options (dict): Current change group options and their scores
        unq_changes (pandas.DataFrame): Non-duplicated forecast changes
        option (pandas.DataFrame): Change group option
        cat_weights (dict): Category changes and their weights
        stdf (pandas.DataFrame): Subset of IMPROVER data
    Returns:
        options (dict): Updated change group options and scores
    """
    # Take copy of option as it may be updated
    option_cp = option.copy()

    # Check if option passes tests
    taf_rules_check = tc.changes_tests(option_cp)

    # If tests failed, do not add to options
    if not taf_rules_check:
        return options

    # Extend change group periods if possible to create a safer TAF
    option_cp = extend_period(stdf, option_cp, unq_changes, 'back')
    option_cp = extend_period(stdf, option_cp, unq_changes, 'forward')

    # Try to extend further for groups < 2 hours
    option_cp = extend_period(stdf, option_cp, unq_changes, 'back',
                              check_length=True)
    option_cp = extend_period(stdf, option_cp, unq_changes, 'forward',
                              check_length=True)

    # If removing changes leaves all zeros, don't update options
    num_changes = np.count_nonzero(option_cp.drop('time', axis=1))
    if not num_changes:
        return options

    # Get groups again using updated option df
    groups = get_chg_values(stdf, option_cp, unq_changes)

    # Double check all tests still passed
    assert tc.changes_tests(option_cp), ' Tests not passed in adapted option'

    # Get performance score and add to options
    score = tc.get_score(unq_changes, option_cp, cat_weights, groups)
    options.append({'score': score, 'groups': groups})

    return options


def values_bases(changes_df, stdf, group):
    """
    Collects values relevant to group. Also collects relevant base
    conditions.

    Args:
        changes_df (pandas.DataFrame): Forecast changes
        stdf (pandas.DataFrame): Subset of IMPROVER data
        group (dict): Change group info
    Returns:
        values (dict): Possible values to use in change group
        bases (dict): Base conditions
    """
    # To add values to
    values = {param: [] for param in co.PARAM_NAMES[stdf.attrs['wx_type']]}
    bases = {param: [] for param in co.PARAM_NAMES[stdf.attrs['wx_type']]}

    # Find values for each point in IMPROVER data for which category
    # in group is forecast
    for perc_str, col in changes_df.items():

        # Move on if no forecasts of change group category
        if col[col == group['cat']].empty:
            continue

        # Get data at percentile and reset index
        perc_df = stdf[stdf['percentile'] == int(perc_str[:2])]
        perc_df.reset_index(drop=True, inplace=True)

        # Get values and base conditions at indices at which change
        # category is predicted
        for ind, _ in col[col == group['cat']].items():

            # Add required values and bases
            for param in bases:

                # Add IMPROVER values
                values[param].append(perc_df.loc[ind][param])

                # Add base values (could be more than 1 for each
                # parameter if group overlaps with BECMG period)
                for base_col in [f'base_{param}', f'base_{param}_2']:
                    if perc_df.loc[ind][base_col] not in bases[param]:
                        bases[param].append(perc_df.loc[ind][base_col])

    return values, bases
