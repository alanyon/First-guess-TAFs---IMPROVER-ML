"""
Functions used in TEMPO and PROB group calculations.

Functions:
    adjust_wind: Checks wind values and adjusts if necessary.
    calc_close_score: Calculates closeness score.
    calc_safe_score: Calculates TAF safety score.
    calc_simple_score: Calculates TAF simplicity score.
    changes_tests: Determines whether change group option is viable.
    choose_wx: Chooses suitable sig wx code.
    cld_change_row: Determines whether cloud change significant.
    get_changes: Gets DataFrame just with significant change indicators.
    get_clds: Determines suitable cloud values for TEMPO/PROB group.
    get_fix_prob: Determines when probs in a group should be fixed.
    get_groups: Gets change groups from dataframe.
    get_other_bases: Finds extra base conditions.
    get_score: Calculates performance score for change group option.
    get_vis_wx: Gets suitable vis/sig wx values for TEMPO/PROB group.
    get_weights: Weights categories by how often they are predicted.
    gust_change_row: Determines whether gust change significant.
    remove_duplicates: Removes duplicate forecasts.
    tempo_change: Determines if change significant for PROB/TEMPO group.
    tempo_no_tempo: Determines whether TEMPO suitable for change group.
    update_tempo_wx: Checks sig wx valid and updates if necessary.
    vis_change_row: Determines whether vis change significant.
"""
import itertools
from datetime import timedelta

import numpy as np
import pandas as pd

import common.calculations as ca
import common.checks as ch
import common.configs as co


def adjust_wind(vals, bases, cat):
    """
    Compares wind values against base conditions and adjusts them if
    necessary to ensure valid changes.

    Args:
        vals (dict): Wind values to be used in TEMPO PROB group
        bases: (dict): Base conditions to compare against
        cat (int): Change category.
    Returns:
        vals (dict): Updated wind values to be used in TEMPO PROB group
    """
    # Positive category implies an increase in wind strength
    if cat > 0:

        # Ensure new mean is at least 3kt more than base means(s) -
        # not a technical requirement but best practice
        new_mean = max([vals['wind_mean'],
                        *[base_mean + 3 for base_mean in bases['wind_mean']]])

        # Ensure mean is at least 15kt
        new_mean = max(new_mean, 15)

        # Ensure new_gust is at least 10kt more than new mean
        new_gust = max([vals['wind_gust'], new_mean + 10])

        # Ensure new gust is at least 10kt more than base gust(s)
        new_gust = max([new_gust,
                        *[base_gust + 10 for base_gust in bases['wind_gust']]])

    # Negative category implies a decrease in wind strength
    if cat < 0:

        # Ensure new mean is at least 3kt less than base means(s) -
        # not a technical requirement but best practice
        new_mean = min([vals['wind_mean'],
                        *[base_mean - 3 for base_mean in bases['wind_mean']]])

        # On the majority of occasions when a decrease in gusts is
        # forecast, the gusts become insignificant - even if this is not
        # the case, removing the gusts still creates a legal and safer
        # TAF (more conservative) TAF
        new_gust = new_mean

    # Update values dictionary
    vals['wind_mean'] = new_mean
    vals['wind_gust'] = new_gust

    return vals


def calc_close_score(changes, option, cat_weights):
    """
    For each category predicted by the model, calculates a score that
    represents how close the option is in terms of the percentile chosen
    and in terms of forecast time. Scores for each category are weighted
    by their prominence in full model forecast. Maximum (best) score is
    1, minimum (worst) score is 0.

    Args:
        changes (pandas.DataFrame): All non-duplicated changes
        option: (pandas.DataFrame): Option for change groups
        cat_weights (dict): Category changes and their weights
    Returns:
        score (float): Score
    """
    # Start with default score of zero and update with score for each
    # category
    score = 0

    # Find score for each category
    for cat in cat_weights:

        # If option does not contain category, score is zero so move on
        # to next iteration without adding to overall score.
        if cat not in option.values:
            continue

        # Find fraction of forecast times in which category predicted in
        # option is the same as that in changes
        c_inds = [ind for ind in changes.index
                  if cat in changes.iloc[ind].values]
        o_inds = [ind for ind in option.index
                  if cat in option.iloc[ind].values]

        matched_inds = [ind for ind in c_inds if ind in o_inds]
        cat_score = len(matched_inds) / len(c_inds)

        # If category predicted at the same time, penalise if
        # percentiles different
        prob_diffs = []
        for ind in matched_inds:
            c_perc = changes.iloc[ind][changes.iloc[ind] == cat].index
            o_perc = option.iloc[ind][option.iloc[ind] == cat].index

            # Should only be one percentile predicting category so check
            assert (len(c_perc) == 1
                    and len(o_perc) == 1), 'Multiple percentiles'

            # Maximum probability difference is 20 (e.g. 70th vs 50th
            # percentiles). Increased and decreased changes are treated
            # separately so lower and higher percentiles (e.g. 30th vs
            # 70th) will not be compared
            prob_diff = abs(int(c_perc[0][:2]) - int(o_perc[0][:2]))
            prob_diffs.append(prob_diff)

        # Get the mean probability difference and apply penalty to score
        # by dividing by 100 - this means that the maximum penalty is
        # multiplying the score by 0.8
        if not prob_diffs:
            mean_prob_diff = 0
        else:
            mean_prob_diff = np.mean(prob_diffs)
        cat_score *= (1 - mean_prob_diff / 100)

        # Weight score by category prominence and add to overall score -
        # this ensures that the eventual score will be between 0 and 1
        weighted_cat_score = cat_score * cat_weights[cat]
        score += weighted_cat_score

    return score


def calc_safe_score(changes, option):
    """
    This score provides a measure for how safe the eventual TAF will be
    in terms of its likelihood of going bust. It is framed in terms of
    categories covered. If all categories forecast in the model are
    covered by the option, a score of 1 will be given. If no categories
    are covered a score of 0 will be given.

    Args:
        changes (pandas.DataFrame): All non-duplicated changes
        option: (pandas.DataFrame): Option for change groups
    Returns:
        score (float): Score
    """
    # Drop time columns
    no_time_option = option.drop(['time'], axis=1)
    no_time_changes = changes.drop(['time'], axis=1)

    # Find rows with a decrease and rows with an increase
    rows_w_dec = no_time_changes.min(axis=1)<0
    rows_w_inc = no_time_changes.max(axis=1)>0

    # Create dfs of bools indicating if decrease/increase covered.
    dec_covered = (no_time_option[rows_w_dec].min(axis=1)
                   <= no_time_changes[rows_w_dec].min(axis=1))
    inc_covered = (no_time_option[rows_w_inc].max(axis=1)
                   >= no_time_changes[rows_w_inc].max(axis=1))
    cats_covered = pd.concat([inc_covered, dec_covered])

    # Take mean of rows with categories covered (if any)
    if cats_covered.empty:
        score = 0
    else:
        score = cats_covered.mean()

    # Cap score at ~0.5 for options in which not all categories covered
    # - this rewards options in which all categories are covered
    if score < 1:
        score *= 0.5

    return score


def calc_simple_score(groups):
    """
    This score provides a measure of simplicity based on how many change
    groups the option will add to the TAF. Simplicity is rewarded, so
    the fewer the number of change grroups the higher the score, with 1
    change group given the maximum score of 1.

    Args:
        groups (list): List of potential change groups
    Returns:
        score (float): Score
    """
    # Score is based on number of groups, but scale the score between
    # 0 and 1, with infinity change groups giving a score of 0 and 1
    # change group giving score of 1
    score = 2 / (1 + len(groups))

    # Scale it down even more to further penalise multiple groups (still
    # ranges from 0 to 1)
    score *= score

    # Apply extra penalty if overlapping groups cover same period - this
    # is technically allowed but generally discouraged
    periods = [group['period'] for group in groups.values()]
    period_combs = itertools.combinations(periods, 2)
    same_periods = [comb for comb in period_combs if comb[0] == comb[1]]
    score *= 0.2 ** len(same_periods)

    return score


def changes_tests(option):
    """
    Performs a series of tests to determine whether the the dataframe of
    changes would lead to a legal TAF that would also pass best practice
    guidelines (e.g. no simultaneous forecasts of increasing and
    decreasing conditions).

    Args:
        option (pandas.DataFrame): Option for change groups
    Returns:
        (Test result) (bool): Indicator for whether tests passed
    """
    # Take copy of dataframe and drop time column
    just_percs = option.drop(['time'], axis=1)

    # No point keeping if all zeros
    if just_percs.astype(bool).sum().sum() == 0:
        return False

    # Loop though rows (forecast times) of dataframe
    for _, row in just_percs.iterrows():

        # Number of overlapping groups at this time
        overlaps = len(row[row != 0])

        # Can't have more than 2 overlapping prob groups
        if overlaps > 2:
            return False

        # Can have 2 overlapping groups under certain conditions
        if overlaps == 2:

            # Can't have increasing and decreasing conditions forecast
            # at the same time
            if not row[row > 0].empty and not row[row < 0].empty:
                return False

            # The two overlapping groups can't be forecasting the same
            # category
            if len(row[row != 0].unique()) == 1:
                return False

            # Otherwise, only case when overlapping groups is when one
            # is a TEMPO and one is a PROB - i.e. if no changes in the
            # 50th percentile, do not allow overlap
            if not (row['50d'] == 0) ^ (row['50i'] == 0):
                return False

    # If False not returned, all tests have passed, so return True
    return True


def choose_wx(all_wxs, wxs, cat, base_wx=None):
    """
    Chooses worst/best weather depending on whether increasing or
    decreasing change.

    Args:
        all_wxs (list): All relevant weather codes
        wxs (list): Possible weather codes to use
        cat (int): Change category from base values
        base_wx (str): Base sig wx to consider (if any)
    Returns:
        wx_choice (str): Chosen wx code
    """
    # Get relevant wx codes (precip or non-precip)
    relevant_wxs = [wx for wx in wxs if wx in all_wxs]

    # Add in base wx (if any)
    if base_wx:
        relevant_wxs.append(base_wx)

    # If any relevant code found, choose appropriate one
    if relevant_wxs:

        # For decreasing change pick first in all_wxs list (i.e. worst)
        if cat < 0:
            wx_choice = all_wxs[min(all_wxs.index(wx) for wx in relevant_wxs)]

        # For increasing change pick last in all_wxs list (i.e. best)
        elif cat > 0:
            wx_choice = all_wxs[max(all_wxs.index(wx) for wx in relevant_wxs)]

    # If no relevant codes, make wx choice None
    else:
        wx_choice = None

    return wx_choice


def cld_change_row(row):
    """
    Determines if cloud change significant from base conditions on row
    of dataframe. Negative values indicate decreased changes, positive
    values indicate increased changes, zeros indicate no significant
    changes.

    Args:
        row (pandas.Series): Row of dataframe
    Returns:
        change (int): Indicator for whether change significant
    """
    # Check if change significant compared to main base values
    main_change = ch.cld_change(row['base_cld_cat'], row['cld_cat'],
                                row['rules_col'])

    # Check if change significant compared to later base values
    early_change = ch.cld_change(row['base_cld_cat_2'], row['cld_cat'],
                                 row['rules_col'])

    # Check if change significant compared to any overlapping group
    overlap_change = ch.cld_change(row['base_cld_cat_3'], row['cld_cat'],
                                   row['rules_col'])

    # Direction of BECMG change (increasing or decreasing conditions)
    becmg_change = row['base_cld_cat'] - row['base_cld_cat_2']

    # Check if change is significantly different from both bases
    change = tempo_change(main_change, early_change, overlap_change,
                          becmg_change, row['percentile'])

    return change


def get_changes(stdf):
    """
    Filters and re-arranges big dataframe to create a dataframe just
    containing forecast changes for each percentile.

    Args:
        stdf (pandas.DataFrame): Subset of IMPROVER data
    Returns:
        changes (pandas.DataFrame): Small dataframe containing changes
    """
    # Collect changes into smaller dataframe with percentiles as columns
    # and row for each time
    changes_percs = stdf[['time', 'percentile', 'change']]
    for ind, gb_obj in enumerate(changes_percs.groupby(by='percentile')):
        perc_df = gb_obj[1]
        perc_df.rename({'change': int(gb_obj[0])}, axis=1, inplace=True)
        perc_df.drop(['percentile'], axis=1, inplace=True)
        perc_df.reset_index(drop=True, inplace=True)
        if ind == 0:
            changes = perc_df
        else:
            changes = changes.merge(perc_df)

    # For purposes of finding change groups, convert percentiles to
    # strings and split 50th percentile into decreasing (d) and
    # increasing (i) changes
    changes.columns = changes.columns.astype(str)

    changes['50d'] = changes.apply(lambda x: x['50'] if x['50'] < 0 else 0,
                                   axis=1)
    changes['50i'] = changes.apply(lambda x: x['50'] if x['50'] > 0 else 0,
                                   axis=1)
    changes.drop(['50'], axis=1, inplace=True)

    # Remove duplicate changes (e.g. if 30th and 40th percentiles both
    # predict category 1, just keep 40th change)
    unq_changes = changes.apply(remove_duplicates, axis=1)

    return unq_changes


def get_clds(values, change_cat, rules):
    """
    Gets suitable cloud values to use in TEMPO/PROB group.

    Args:
        values (dict): Possible cloud values to use
        change_cat (int): Change category from base conditions
        rules (str): Airport TAF rules (defence, civil or offshore)
    Returns:
        best_values (dict): Values to use in PROB/TEMPO group
    """
    # There should always be only 1 TAF category in values so assert
    # before picking first one in list
    int_cats = [int(cat) for cat in values['cld_cat']]
    assert len(np.unique(int_cats)) == 1, 'Check cld categories'
    cld_cat = values['cld_cat'][0]

    # Get worst/best cloud, depending on change cat
    if change_cat < 0:
        cld_3 = min(values['cld_3'])
        cld_5 = min(values['cld_5'])
    elif change_cat > 0:
        cld_3 = max(values['cld_3'])
        cld_5 = max(values['cld_5'])

    # Avoid two cloud groups with similar values (e.g. SCT035 BKN040)
    # - only allow two cloud groups if in different categories
    if cld_3 != cld_5:
        cat_3 = ca.get_cld_cat(cld_3, cld_3, rules)
        cat_5 = ca.get_cld_cat(cld_5, cld_5, rules)
        if cat_3 == cat_5:
            cld_3 = cld_5

    # Collect values into dictionary
    best_values = {'cld_3': cld_3, 'cld_5': cld_5, 'cld_cat': cld_cat}

    return best_values


def get_fix_prob(group, groups):
    """
    Determines when probabilities in the group should be fixed.
    Probabilities are fixed if the group is part of a pair of
    overlapping groups. This is required as overlapping groups should
    only be a TEMPO overlapping with a PROB.

    Args:
        groups (list): List of change groups
        group (dict): Single change group
    Returns:
        prob_fix (bool): Indicator for whether prob should be fixed
    """
    # Get dts covering group period (ommitting last dt as prob is only
    # fixed for overlapping groups and TEMPO 03/06 TEMPO 06/09 are not
    # overlapping)
    group_dts = ca.get_period_dts(group['change_period'])

    # Loop through other groups in list
    other_grps = [grp for grp in groups if grp != group]
    for other_grp in other_grps:

        # Get dts for this period
        other_grp_dts = ca.get_period_dts(other_grp['change_period'])

        # If any overlapping dts, set prob to fixed
        if set(group_dts).intersection(other_grp_dts):
            return True

    return False


def get_groups(cdf):
    """
    Gets change groups from a dataframe of changes. The dataframe must
    have passed the tests in the taf_rules_check() function for this
    function to work.

    Args:
        cdf (pandas.DataFrame): Dataframe of significant changes
    Returns:
        groups (dict): Change groups
    """
    # Find all change groups represented in chg_groups dataframe
    # Dictionary for counting change groups needed - need to keep track
    # of categories forecast and number of zeros to determine when a new
    # change group will be added at each hour in the period
    cats = {perc: 0 for perc in cdf.columns}
    zeros = {perc: 0 for perc in cdf.columns}
    perc_groups = {perc: 0 for perc in cdf.columns}
    end_times = {perc: None for perc in cdf.columns}
    groups = {}

    # Loop through each hour and add to dictionary
    for _, row in cdf.iterrows():

        # Drop time but keep for later
        time = row['time']
        row.drop(['time'], inplace=True)

        # Loop through all percentiles with non-zero entries
        for perc, cat in row.items():

            # Add to zeros if no category forecast
            if cat == 0:
                zeros[perc] += 1
                continue

            # If here is reached, category in row is not zero - new
            # change group needed if category differs from that in
            # current open group (if any or if number of zeros since
            # last same category is 5 or more)
            if any([not groups, cat != cats[perc], zeros[perc] >= 5]):

                # Add end time to previous group if any
                if perc_groups[perc]:
                    g_key = f'{perc}_{perc_groups[perc]}'
                    if len(groups[g_key]['period']) == 1:
                        groups[g_key]['period'].append(end_times[perc])

                # Create new group with new info - end time determined
                # when new group found or when all rows have been looked
                # at, but for now set end time to start time + 1 hour
                perc_groups[perc] += 1
                new_g_key = f'{perc}_{perc_groups[perc]}'
                groups[new_g_key] = {'perc': perc, 'cat': cat,
                                     'period': [time]}
                end_times[perc] = time + timedelta(hours=1)

                # Reset zeros and change to new cat
                zeros[perc] = 0
                cats[perc] = cat

            # Update perc end time and reset zeros if same category
            # and < 5 zeros
            else:
                end_times[perc] = time + timedelta(hours=1)
                zeros[perc] = 0

            # End other percentile groups if necessary
            for o_perc in cdf.columns:

                # Ignore if same percentile or if other percentile is 0
                if any([o_perc == perc, cats[o_perc] == 0]):
                    continue

                # If same category in other group, other group should be
                # ended. Also, if cats are different but both
                # percentiles are not 50 (i.e. for PROB groups), other
                # group should be ended - this is to prevent overlapping
                # PROB groups
                if cats[o_perc] == cat or (all('50' not in per
                                               for per in [perc, o_perc])):
                    cats[o_perc] = 0
                    g_key = f'{o_perc}_{perc_groups[o_perc]}'
                    groups[g_key]['period'].append(end_times[o_perc])

    # Finally, need to end any open groups
    for group in groups.values():
        if len(group['period']) == 1:
            perc = group['perc']
            group['period'].append(end_times[perc])

    return groups


def get_other_bases(group, o_group):
    """
    Compares two change groups and checks if values in o_group should be
    used as extra set of base conditions for group to consider. This
    happens if groups overlap each other. N.B. For overlapping
    comparison, end time is actually 1 hour earlier than period end time
    as groups can start and endat the same time without overlapping
    (e.g. TEMPO 00/04 ... TEMPO 04/08 ...)

    Args:
        group (dict): PROB/TEMPO group
        o_group (dict): Other PROB/TEMPO group
    Returns:
        o_bases (dict): Extra base conditions
    """
    # Get all change times for comparison
    g_dts = ca.get_period_dts(group['period'])
    o_dts = ca.get_period_dts(o_group['period'])

    # Need to check values if any change times overlap and if
    # group cat is more extreme forecast
    if all([[dt for dt in g_dts if dt in o_dts],
            abs(group['cat']) > abs(o_group['cat'])]):

        # Define 'bases' as values in the less extreme group
        o_bases = {param: [o_group['values'][param]]
                   for param in o_group['values']}

    # Otherwise, don't need to consider extra bases
    else:
        o_bases = {}

    return o_bases


def get_score(changes, option, cat_weights, groups):
    """
    Calculates three scores that measure different aspects of
    performance for potential change groups, then combines into a single
    score.

    Args:
        changes (pandas.DataFrame): All non-duplicated changes
        option: (pandas.DataFrame): Option for change groups
        cat_weights (dict): Category changes and their weights
        groups: TEMPO/PROB groups based on option
    Returns:
        score (float): Combined performance score
    """
    # Calculate closeness score
    closeness = calc_close_score(changes, option, cat_weights)

    # Calculate safety score
    safety = calc_safe_score(changes, option)

    # Calculate simplicity score
    simplicity = calc_simple_score(groups)

    # Combine all scores to get overall score
    total_score = closeness * safety * simplicity

    return total_score


def get_vis_wx(values, bases, change_cat, rules):
    """
    Gets suitable visibility and sig wx values to use in TEMPO/PROB
    group.

    Args:
        values (dict): Possible vis/wx values to use
        bases: (dict): Base values
        change_cat (int): Change category from base conditions
        rules (str): Airport TAF rules (defence, civil or offshore)
    Returns:
        best_values (dict): Values to use in PROB/TEMPO group
    """
    # There should always be only 1 TAF category in values so assert
    int_cats = [int(cat) for cat in values['vis_cat']]
    assert len(np.unique(int_cats)) == 1, 'Check vis categories'
    vis_cat = values['vis_cat'][0]

    # Get worst/best vis, depending on change cat
    if change_cat < 0:
        vis_v = min(values['vis'])
    elif change_cat > 0:
        vis_v = max(values['vis'])

    # Split up wx codes into separate components and get unique codes
    wxs_values = list(set(' '.join(values['sig_wx']).split()))
    wxs_bases = list(set(' '.join(bases['sig_wx']).split()))

    # Get worst/best weather, depending on change type
    non_precip_b = choose_wx(co.NON_PRECIP_CODES, wxs_bases, change_cat)
    precip_b = choose_wx(co.PRECIP_CODES, wxs_bases, change_cat)
    non_precip_v = choose_wx(co.NON_PRECIP_CODES, wxs_values, change_cat,
                             base_wx=non_precip_b)
    precip_v = choose_wx(co.PRECIP_CODES, wxs_values, change_cat,
                         base_wx=precip_b)

    # Colect wxs into lists
    wxs_v = [wx for wx in [non_precip_v, precip_v] if wx]
    wxs_b = [wx for wx in [non_precip_b, precip_b] if wx]

    # Get suitable reported and implied wx codes
    wx, implied_wx = ca.get_new_wxs(wxs_b, wxs_v, vis_v, rules)

    # Collect values into dictionary
    best_values = {'vis': vis_v, 'vis_cat': vis_cat, 'sig_wx': wx,
                   'implied_sig_wx': implied_wx, 'base_wxs': bases['sig_wx'],
                   'base_viss': bases['vis'], 'wxs': values['sig_wx']}

    return best_values


def get_weights(chgs):
    """
    Weights forecast categories by how often they are predicted.

    Args:
        chgs (pandas.DataFrame): All non-duplicated changes
    Returns:
        cat_weights (dict): Categories and their weights
        unq_cats (np.array): Unique categories forecast
    """
    # Count categories
    unq_cats, counts = np.unique(chgs, return_counts=True)

    # Find unique non-zero categories
    non_zero_idx = np.argwhere(unq_cats).flatten()
    unq_cats = unq_cats[non_zero_idx]

    # Weight by number of forecasts and collect in dictionary
    weights = counts[non_zero_idx]/counts[non_zero_idx].sum()
    cat_weights = dict(zip(unq_cats, weights))

    return cat_weights, unq_cats


def gust_change_row(row):
    """
    Determines if gust change significant from base conditions on row
    of dataframe. Negative values indicate decreased changes, positive
    values indicate increased changes, zeros indicate no significant
    changes.

    Args:
        row (pandas.Series): Row of dataframe
    Returns:
        change (int): Indicator for whether change significant
    """
    # Check if change significant compared to main base values
    main_change = ch.gust_change(row['base_wind_gust'], row['wind_gust'],
                                 row['base_wind_mean'], row['wind_mean'])

    # Check if change significant compared to later base values
    early_change = ch.gust_change(row['base_wind_gust_2'], row['wind_gust'],
                                  row['base_wind_mean_2'], row['wind_mean'])

    # Check if change significant compared to any overlapping group
    overlap_change = ch.gust_change(row['base_wind_gust_3'], row['wind_gust'],
                                  row['base_wind_mean_3'], row['wind_mean'])

    # Direction of BECMG change (increasing or decreasing conditions)
    becmg_change = int(row['base_wind_gust'] - row['base_wind_gust_2'])

    # Check if change is significantly different from both bases
    change = tempo_change(main_change, early_change, overlap_change,
                          becmg_change, row['percentile'])

    return change


def remove_duplicates(row):
    """
    Removes duplicate forecasts and keeps forecast that would lead to
    highest probability in the eventual TAF - e.g. if 60th and 70th
    percentiles both predict category 2 at the same time, the 70th
    percentile forecast is removed so that a PROB40 (60th percentile)
    will eventually be chosen.

    Args:
        row (pandas.Series): Row of DataFrame
    Returns:
        row (pandas.Series): Updated row
    """
    # If 30th and 40th percentile change the same, just keep 40th
    if row['40'] == row['30'] or row['50d'] == row['30']:
        row['30'] = 0

    # If 40th and 50th percentile change the same, just keep 50th
    if row['50d'] == row['40']:
        row['40'] = 0

    # If 70th and 60th percentile change the same, just keep 60th
    if row['60'] == row['70'] or row['50i'] == row['70']:
        row['70'] = 0

    # If 60th and 50th percentile change the same, just keep 50th
    if row['50i'] == row['60']:
        row['60'] = 0

    return row


def tempo_change(main_change, early_change, overlap_change, becmg_change,
                 percentile):
    """
    Determines if value constitutes a significant change from base
    conditions.
    Args:
        main_change (int): Change from main base conditions
        early_change (int): Change from earlier base conditions
        overlap_change (int): Change from overlapping TEMPO group
        becmg_change (int): BECMG change (increase or decrease or none)
        percentile (int): percentile of forecast value
    Returns:
        change (int): Indicator for whether change significant
    """
    # Collect changes into list, including BECMG change if any
    changes = [main_change, early_change, overlap_change]
    if becmg_change:
        changes.append(becmg_change)

    # Change is not significant if not significantly different from any
    # base values and overlapping PROB/TEMPO groups
    if not all(changes):
        return 0

    # Changes should all be of same type (increasing or decreasing
    # conditions) to avoid TAF being confusing - negative value below
    # indicates this is not the case so do not allow change
    if not (all(chg > 0 for chg in changes)
            or all(chg < 0 for chg in changes)):
        return 0

    # Additional condition for non-50th percentiles is that 60th/70th
    # percentiles are only concerned with increased changes and
    # 30th/40th percentiles are only concerned with decreased changes
    if any([percentile in [60, 70] and main_change < 0,
            percentile in [30, 40] and main_change > 0]):
        return 0

    # If this stage is reached, change can be considered significant, so
    # make change equal to main change
    change = main_change
    return change


def tempo_no_tempo(group, changes_df, wx_type):
    """
    Decides whether TEMPO is used - PROB groups can be a straight PROB
    or a TEMPO (e.g. PROB30 or PROB30 TEMPO).

    Args:
        group (dict): TEMPO/PROB group info
        changes_df (pandas.DataFrame): Significant changes
        wx_type (str): Weather type (wind, vis or cld)
    Returns:
        change_type (str): Full TAF term
    """
    # Always TEMPO for 50th percentile
    if '50' in group['perc']:
        change_type = 'TEMPO'
        return change_type

    # For PROB groups in weather types other than wind (always TEMPO),
    # consider frequency of changes over period
    if wx_type != 'wind':

        # Get change period length in hours
        start, end = group['period']
        period_length = (end - start).total_seconds() / 3600

        # Most periods get extended later but, for these purposes,
        # assume 1 extra hour (e.g, for a forecast period of 00Z to 03Z,
        # period length should be 4 hours, not 3, as 4 forecasts
        # possible - 00Z, 01Z, 02Z, 03Z)
        period_length += 1

        # Number of unique category forecasts over change period
        cat_forecasts = changes_df[changes_df == group['cat']].fillna(0)
        n_forecasts = cat_forecasts.astype(bool).sum().sum()

        # If forecast frequency less than or equal to 0.5, choose TEMPO
        if n_forecasts / period_length <= 0.5:
            tempo = True

        # Otherwise, generally don't choose TEMPO, but exception is when
        # precip is forecast
        elif wx_type == 'vis' and any(wx in co.PRECIP_CODES for wx in
                                      group['values']['sig_wx'].split()):
            tempo = True
        else:
            tempo = False

    # For wind, only gusts are covered by PROB/TEMPO groups so always
    # use TEMPO
    else:
        tempo = True

    # Choose prob
    if group['perc'] in ['30', '70']:
        change_type = 'PROB30'
    elif group['perc'] in ['40', '60']:
        change_type = 'PROB40'

    # Add in TEMPO if required
    if tempo:
        change_type = change_type + ' TEMPO'

    return change_type


def update_tempo_wx(values, change_cat, overlap, rules):
    """
    Ensures sig wx code still suitable when considering overlapping
    group and updates it if necessary.

    Args:
        values (dict): Possible values to use
        change_cat (int): Change category from base conditions
        overlap (dict): 'Base' values from overlapping group
        rules (str): Airport TAF rules (defence, civil or offshore)
    Returns:
        values (dict): Updated values
    """
    # Add to bases and get values to re-consider
    wxs_bases = values['base_wxs'] + overlap['sig_wx']
    wxs = values['wxs']

    # Split up wx codes into separate components and get unique codes
    wxs_values = list(set(' '.join(wxs).split()))
    wxs_bases = list(set(' '.join(wxs_bases).split()))

    # Get worst/best weather, depending on change type
    non_precip_b = choose_wx(co.NON_PRECIP_CODES, wxs_bases, change_cat)
    precip_b = choose_wx(co.PRECIP_CODES, wxs_bases, change_cat)
    non_precip_v = choose_wx(co.NON_PRECIP_CODES, wxs_values, change_cat,
                             base_wx=non_precip_b)
    precip_v = choose_wx(co.PRECIP_CODES, wxs_values, change_cat,
                         base_wx=precip_b)

    # Colect wxs into lists
    wxs_v = [wx for wx in [non_precip_v, precip_v] if wx]
    wxs_b = [wx for wx in [non_precip_b, precip_b] if wx]

    # Get suitable reported and implied wx codes
    wx, implied_wx = ca.get_new_wxs(wxs_b, wxs_v, values['vis'], rules)
    values['sig_wx'] = wx
    values['implied_sig_wx'] = implied_wx

    return values


def vis_change_row(row):
    """
    Determines if vis change significant from base conditions on row
    of dataframe. Negative values indicate decreased changes, positive
    values indicate increased changes, zeros indicate no significant
    changes.

    Args:
        row (pandas.Series): Row of dataframe
    Returns:
        change (int): Indicator for whether change significant
    """
    # Check if change significant compared to main base values
    main_change = ch.vis_change(row['base_vis_cat'], row['vis_cat'])

    # Check if change significant compared to later base values
    early_change = ch.vis_change(row['base_vis_cat_2'], row['vis_cat'])

    # Check if change significant compared to any overlapping group
    overlap_change = ch.vis_change(row['base_vis_cat_3'], row['vis_cat'])

    # Direction of BECMG change (increasing or decreasing conditions)
    becmg_change = int(row['base_vis_cat'] - row['base_vis_cat_2'])

    # Check if change is significantly different from both bases
    change = tempo_change(main_change, early_change, overlap_change,
                          becmg_change, row['percentile'])

    return change
