"""
Functions to simplify TAF by merging change groups.

Functions:
    add_cb: Adds CB to cloud where appropriate.
    check_periods: Checks if change periods allow for merging.
    choose_change_type: Chooses appropriate change type to use.
    cld_merge: Checks if cloud values groups allow merging.
    combine_consecutives:Combines consecutive change groups if possible.
    combine_overlaps: Combines overlapping change groups.
    consecutive: Checks if change periods in groups are consecutive.
    get_base_value_cats: Gets TAF categories in change groups and bases.
    get_change_type: Gets suitable change type from combo of groups.
    get_unq_wxs: Gets unique weather changes from change groups.
    optimise_groups: Combines change groups where possible.
    same_prob: Determines if change types represent same probability.
    sort_combs: Sorts combinations according to the group change types.
    vis_merge: Checks if visibility values groups allow merging.
    wind_merge: Checks if wind values groups allow merging.
    wxs_merge: Checks if group values are suitable for merging.
"""
import copy
import itertools
from datetime import timedelta

import numpy as np

import common.calculations as ca
import common.checks as ch
import common.configs as co


def add_cb(groups, site_data):
    """
    Adds indicators to allow CB to be added to cloud groups when writing
    out TAF later where appropriate.

    Args:
        groups (list): Base conditions and change groups
        site_data (pandas.DataFrame): IMPROVER data
    Returns:
        new_groups (list): Updated base conditions and change groups
    """
    # Separate PROB/TEMPO groups from base/BECMGS
    new_groups = [group for group in groups
                 if group['change_type'] in ['base', 'BECMG']]
    probs_tempos = [grp for grp in groups if grp not in new_groups]

    # Get lightning probs for hours of TAF
    tdf = site_data[site_data['taf_time'].isin(['during'])]
    lightning_50 = tdf[(tdf['percentile'] == 50)][['time', 'lightning']]

    # Loop through change groups
    for group in probs_tempos:

        # Only consider vis changes (which contain sig wx - inc. precip)
        if 'vis' not in group['wx_changes']:
            new_groups.append(group)
            continue

        # Get dts covering group period
        group_dts = ca.get_period_dts(group['change_period'])
        lightning_group = lightning_50[lightning_50['time'].isin(group_dts)]

        # Get separate sig wx codes from group
        wxs = group['implied_sig_wx'].split()

        # Only try to add CB to cloud if TS in sig wx or heavy previp is
        # forecast and lightning probs are at least 0.1
        if not any([any(wx in co.TS_CODES for wx in wxs),
                    all([any(wx in co.HVY_CODES for wx in wxs),
                         max(lightning_group['lightning']) >= 0.1])]):
            new_groups.append(group)
            continue

        # First try to add to add CB to existing cloud in group
        if all(['cld' in group['wx_changes'],
                any(group['clds'][cld] < 5000 for cld in group['clds'])]):

            # Add CB flag to cloud group
            group['cb'] = 1

            # Add higher cloud group if needed for realistic CB base
            sig_cld = {cld: group['clds'][cld] for cld in group['clds']
                       if group['clds'][cld] < 5000}
            if all(cld_base < 1000 for cld_base in sig_cld.values()):
                group['clds']['cld_8'] = 1500

        # Otherwise, try to add new cloud group depending on base
        # conditions
        else:

            # Collect base cloud categories
            base_cats = [group['main_bases']['cld_cat']]
            if group['early_bases']:
                base_cats.append(group['early_bases']['cld_cat'])

            # Look for any overlapping groups, add to base cats if any
            for ogroup in probs_tempos:

                # Get dts covering group
                o_dts = ca.get_period_dts(ogroup['change_period'])

                # Add cld cat to base cats if overlapping and different
                if all([list(set(o_dts).intersection(group_dts)),
                        'cld' in ogroup['wx_changes']]):
                    base_cats.append(ogroup['cld_cat'])

            # Add BKN014CB to group if different from base conditions
            cat_14 = ca.get_cld_cat(1400, 1400, site_data.attrs['rules'])
            if all(ch.cld_change(cld_cat, cat_14, site_data.attrs['rules']) < 0
                   for cld_cat in base_cats):
                group['wx_changes'].append('cld')
                for okta in [1, 3, 5, 8]:
                    group['clds'][f'cld_{okta}'] = 1400
                group['cld_cat'] = cat_14
                group['cb'] = 1

        # Add back into bb groups
        new_groups.append(group)

    return new_groups


def check_periods(comb):
    """
    Checks if change periods allow for merging, returning new change
    period if so.

    Args:
        comb (tuple): Combination of change groups
    Returns:
        new_period (list): New change period
    """
    # Get earliest and latest times from groups in comb to define
    # potential new period
    dts = list(itertools.chain(*[grp['change_period'] for grp in comb]))
    new_period = [min(dts), max(dts)]

    # Simplest case is when all groups are over exactly the same period,
    # in which case, merging is possible
    if all(grp['change_period'] == comb[0]['change_period'] for grp in comb):
        return new_period

    # Otherwise, need to check whether groups cover same base period
    base_start = max(grp['main_bases']['change_period'][1] for grp in comb)
    base_end = min(grp['end_dt'] for grp in comb)

    # Do not merge groups if any dts fall outside of base period
    if min(dts) < base_start or max(dts) > base_end:
        return None

    # If all groups in same base period, do not merge if their periods
    # differ by 3 hours or more
    period_diffs = [grp['change_period'][1] - grp['change_period'][0]
                    for grp in comb]
    min_length = min(diff.total_seconds() / 3600 for diff in period_diffs)
    new_period_length = (new_period[1] - new_period[0]).total_seconds() / 3600
    if min_length <= new_period_length - 3:
        return None

    return new_period


def choose_change_type(change_types):
    """
    Chooses appropriate change type to use.

    Args:
        change_types (list): List of change types
    Returns:
        new_change_type (str): Chosen change type
    """
    # Choose lowest prob
    if any('30' in change_type for change_type in change_types):
        new_terms = ['PROB30']
    elif any('40' in term for term in change_types):
        new_terms = ['PROB40']
    else:
        new_terms = []

    # Use TEMPO is used in any of the groups
    if any('TEMPO' in change_type for change_type in change_types):
        new_terms.append('TEMPO')

    # Turn into string
    new_change_type = ' '.join(new_terms)

    return new_change_type


def cld_merge(comb, rules, comb_dts, comb_grp):
    """
    Checks if cloud values in combination of groups allow merging,
    updating combined group values if so.

    Args:
        comb (tuple): Combination of change groups
        rules (str): Airport TAF rules (defence, civil or offshore)
        comb_dts (list): All dates/times covered if groups combined
        comb_grp (dict): Combined change group
    Returns:
        comb_grp (dict): Updated combined change group
    """
    # Get base conditions and forecast values from each group
    base_cats, value_cats = get_base_value_cats(comb, comb_dts, 'cld')

    # Check that all group values are significantly different from all
    # base conditions
    for base_cat, value_cat in itertools.product(base_cats, value_cats):
        if not ch.cld_change(base_cat, value_cat, rules):
            return None

    # Check all cloud values are within same category
    if len(set(value_cats)) != 1:
        return None

    # Add cloud values to combined group - just use first group values
    # to avoid averaging problems
    comb_grp['clds'] = comb[0]['clds']

    return comb_grp


def combine_consecutives(groups, rules):
    """
    Combines consecutive change groups of same weather type if possible.

    Args:
        groups (list): All change groups
        rules (str): Airport TAF rules (defence, civil or offshore)
    Returns:
        merged_groups (list): All change groups, merged where possible
    """
    # Start with bases and BECMGS, then add in merged PROB/TEMPO groups
    merged_groups = [group for group in groups
                     if group['change_type'] in ['base', 'BECMG']]

    # Get all TEMPO/PROB groups
    probs_tempos = [group for group in groups if group not in merged_groups]

    # Get list of unique wx changes
    unq_wxs = get_unq_wxs(probs_tempos)

    # Try to merge groups if they contain the same wx changes
    for wxs in unq_wxs:

        # Get groups with same wx changes
        wx_groups = [grp for grp in probs_tempos if grp['wx_changes'] == wxs]

        # Start by attempting to combine all groups, then incrementally
        # reduce number of groups to combine
        ignore_groups = []
        count = len(wx_groups)
        while count > 0 and wx_groups:

            # All possible combinations of change groups (number of
            # groups in each combination dictated by count)
            combinations = itertools.combinations(wx_groups, count)

            # Loop through all combos and attempt to merge into 1 group
            for comb in combinations:

                # Skip if any groups marked to ignore
                if any(group in ignore_groups for group in comb):
                    continue

                # Add to merged groups if only 1 group in comb
                if len(comb) == 1:
                    merged_groups.append(comb[0])
                    continue

                # Test for merge compatability by comparing to bases for
                # each weather type, creating combined group if possible
                comb_grp = wxs_merge(comb, wxs, rules)

                # If merge determined not possible, move to next combo
                if comb_grp is None:
                    continue

                # Get suitable PROB/TEMPO term (if possible)
                new_change_type, fix_prob = get_change_type(comb)
                if new_change_type is None:
                    continue
                comb_grp['change_type'] = new_change_type
                comb_grp['fix_prob'] = fix_prob

                # Add to merged_groups
                merged_groups.append(comb_grp)

                # Add pre-merged groups to ignore list so they are not
                # merged with any other groups
                for group in comb:
                    ignore_groups.append(group)

            # Decrease count so smaller combination of groups can be
            # considered
            count -= 1

    return merged_groups


def combine_overlaps(groups):
    """
    Combines overlapping change groups where sensible to do so.

    Args:
        groups (list): Non-merged change groups
    Returns:
        merged_groups (list): Change groups after merging
    """
    # Separate into BECMGs and TEMPO/PROB groups
    probs_tempos = [group for group in groups
                    if group['change_type'] not in ['base', 'BECMG']]

    # Start with bases and BECMGS, then add in merged PROB/TEMPO groups
    merged_groups = [group for group in groups
                     if group['change_type'] in ['base', 'BECMG']]

    # Start by attempting to combine all groups, then incrementally
    # reduce number of groups to combine
    ignore_groups = []
    count = len(probs_tempos)
    while count > 0 and probs_tempos:

        # All possible combinations of change groups (number of groups
        # in each combination dictated by count)
        combinations = itertools.combinations(probs_tempos, count)

        # Order combinations in terms of change types - consider
        # combinations with similar probs first
        sorted_combs = sort_combs(combinations)

        # Loop through all combinations and try to merge into 1 group
        for comb in sorted_combs:

            # Skip if any groups marked to ignore
            if any(group in ignore_groups for group in comb):
                continue

            # Add to merged groups if only 1 group in comb
            if len(comb) == 1:
                merged_groups.append(comb[0])
                continue

            # Skip if same weather in any combination of groups
            wxs = list(itertools.chain(*[grp['wx_changes'] for grp in comb]))
            unique_wxs = list(set(wxs))
            if len(wxs) != len(unique_wxs):
                continue

            # Check if periods of groups compatible for merging
            new_period = check_periods(comb)

            # Move to next combination if merging not possible
            if new_period is None:
                continue

            # Get appropriate TAF term (i.e. TEMPO, PROB30, etc)
            new_change_type, fix_prob = get_change_type(comb)

            # Move to next iteration if no new change type possible
            if new_change_type is None:
                continue

            # If merging is allowed, combine values into single group
            # Start with first group and update as necessary
            merged_group = copy.deepcopy(comb[0])

            # Use new change period, change type and fix_prob bool
            merged_group['change_period'] = new_period
            merged_group['change_type'] = new_change_type
            merged_group['fix_prob'] = fix_prob

            # Add in weather values from the other groups in comb
            for group in comb[1:]:
                for wx_type in group['wx_changes']:
                    for dkey in co.WX_KEYS[wx_type]:
                        merged_group[dkey] = group[dkey]

            # Include all weather types
            merged_group['wx_changes'] = unique_wxs

            # Add to merged_groups
            merged_groups.append(merged_group)

            # Add pre-merged groups to ignore list so they are not
            # merged with any other groups
            for group in comb:
                ignore_groups.append(group)

        # Decrease count so smaller combo of groups can be considered
        count -= 1

    return merged_groups


def consecutive(comb):
    """
    Checks if change periods in combination of groups are consecutive,
    returning the combined period and datetimes covered in the period if
    so.

    Args:
        comb (tuple): Combination of change groups
    Returns:
        comb_period (list): Combined group change period
        comb_dts (list): All dates/times covered in combined period
    """
    # Get dts from change periods of all groups and order
    grps_dts = []
    for grp in comb:
        for gdt in ca.get_period_dts(grp['change_period']):
            grps_dts.append(gdt)
    grps_dts.sort()

    # Get dts assuming all groups were consecutive
    comb_period = [min(grps_dts), max(grps_dts) + timedelta(hours=1)]
    comb_dts = list(ca.get_period_dts(comb_period))

    # If groups overlap or there are gaps between them, return None
    if grps_dts != comb_dts:
        return None, None

    # Otherwise, return combined period and dts
    return comb_period, comb_dts


def get_base_value_cats(comb, comb_dts, wx_type):
    """
    Gets TAF categories in all change groups and base conditions to
    compare.

    Args:
        comb (tuple): Combination of change groups
        comb_dts (list): All dates/times covered if groups combined
        wx_type (str): Type of weather change
    Returns:
        base_cats (list): Categories in base conditions
        value_cats (list): Categories in change groups
    """
    base_cats, value_cats = [], []
    for grp in comb:

        # Add forecast values
        value_cats.append(grp[f'{wx_type}_cat'])

        # Always need main base conditions
        base_cats.append(grp['main_bases'][f'{wx_type}_cat'])

        # Add in early base conditions if combined group period overlaps
        if grp['early_bases'] is None:
            continue
        early_base_dts = ca.get_period_dts(grp['early_bases']['change_period'])
        if list(set(early_base_dts).intersection(comb_dts)):
            base_cats.append(grp['early_bases'][f'{wx_type}_cat'])

    return base_cats, value_cats


def get_change_type(comb):
    """
    Gets suitable change type from combination of groups if possible.

    Args:
        comb (tuple): Combination of change groups
    Returns:
        new_change_type (str): Change type if any chosen
        fix_prob (bool): Indication whether prob is fixed in new group
    """
    # Start with term in first group
    change_type = comb[0]['change_type']
    fix_prob = comb[0]['fix_prob']

    # Otherwise, loop through other groups
    for group in comb[1:]:

        # If more than 1 group with fixed prob, only merge if probs are
        # the same
        change_types = [change_type, group['change_type']]
        if fix_prob:

            # If more than 1 group with fixed prob...
            if group['fix_prob']:

                # Can't merge if probs are different
                if not same_prob(change_types):
                    return None, None

                # Otherwise, update change type
                change_type = choose_change_type(change_types)

        elif group['fix_prob']:

            change_type = group['change_type']
            fix_prob = group['fix_prob']

        else:

            change_type = choose_change_type(change_types)

    return change_type, fix_prob


def get_unq_wxs(probs_tempos):
    """
    Gets unique weather changes from change groups.

    Args:
        probs_tempos (list): PROB/TEMPO groups
    Returns:
        unq_wxs (list): Unique weather changes
    """
    # Get wx changes from each group and add to list if unique
    unq_wxs = []
    for wx in [grp['wx_changes'] for grp in probs_tempos]:
        if wx not in unq_wxs:
            unq_wxs.append(wx)

    return unq_wxs


def optimise_groups(groups, site_data):
    """
    Combines as many change groups as possible without losing forecast
    information or over-simplifying.

    Args:
        groups (list): Non-merged change groups
        site_data (pandas.DataFrame): IMPROVER and airport data
    Returns:
        groups (list): Change groups after merging
    """
    # Combine consecutive groups of same weather and category
    groups = combine_consecutives(groups, site_data.attrs['rules'])

    # Combine overlapping groups if possible
    groups = combine_overlaps(groups)

    # Add in CB where appropriate
    groups = add_cb(groups, site_data)

    return groups


def same_prob(change_types):
    """
    Determines if list of string change types represent the same
    probability forecast.

    Args:
        change_types (list): String change types to be considered
    Returns:
        same (bool): Indicator for whether probs are the same
    """
    same = any([all('30' in term for term in change_types),
                all('40' in term for term in change_types),
                all(term == 'TEMPO' for term in change_types)])
    return same


def sort_combs(combs):
    """
    Sorts combinations according to the group change types. Combinations
    containing groups with similar probability change types are
    prioritised.

    Args:
        combs (itertools.combinations): Combinations of change groups
    Returns:
        sorted_combs (list): Sorted combinations of change groups
    """
    # Create list of lists with each nested list containing score for
    # combination based on similarity of probs and combination itself
    combs_list = []
    for comb in combs:
        probs = sorted([co.PROB_DICT[grp['change_type']] for grp in comb])
        sum_diffs = np.sum(np.diff(probs))
        combs_list.append([sum_diffs, comb])

    # Sort lists by prob similarity score and extract sorted combs
    sorted_combs_list = sorted(combs_list, key=lambda x: x[0])
    sorted_combs = [c_list[1] for c_list in sorted_combs_list]

    return sorted_combs


def vis_merge(comb, rules, comb_dts, comb_grp):
    """
    Checks if visibility values in combination of groups allow merging,
    updating combined group values if so.

    Args:
        comb (tuple): Combination of change groups
        rules (str): Airport TAF rules (defence, civil or offshore)
        comb_dts (list): All dates/times covered if groups combined
        comb_grp (dict): Combined change group
    Returns:
        comb_grp (dict): Updated combined change group
    """
    # Get base conditions and forecast values from each group
    base_cats, value_cats = get_base_value_cats(comb, comb_dts, 'vis')

    # Check that all group values are significantly different from all
    # base conditions
    for base_cat, value_cat in itertools.product(base_cats, value_cats):
        if base_cat == value_cat:
            return None

    # Check all vis values are within same category
    if len(set(value_cats)) != 1:
        return None

    # Only merge in simplest case when reported wx is the same
    if len({grp['sig_wx'] for grp in comb}) != 1:
        return None

    # Add suitable values to combined group - mean for visibility
    # N.B. sig wx and vis cat will be the same for all groups if this
    # stage is reached so do not need to update these - implied wx may
    # be different but not important at this stage
    comb_grp['vis'] = ca.round_vis(np.mean([grp['vis'] for grp in comb]),
                                   rules)

    return comb_grp


def wind_merge(comb, comb_dts, comb_grp):
    """
    Checks if wind values in combination of groups allow merging,
    updating combined group values if so.

    Args:
        comb (tuple): Combination of change groups
        comb_dts (list): All dates/times covered if groups combined
        comb_grp (dict): Combined change group
    Returns:
        comb_grp (dict): Updated combined change group
    """
    # Get base conditions and forecast values from each group
    bases, values = [], []
    for grp in comb:

        # Add forecast values
        values.append({'wind_dir': grp['wind_dir'],
                       'wind_mean': grp['wind_mean'],
                       'wind_gust': grp['wind_gust']})

        # Always need main base conditions
        bases.append({'wind_dir': grp['main_bases']['wind_dir'],
                      'wind_mean': grp['main_bases']['wind_mean'],
                      'wind_gust': grp['main_bases']['wind_gust']})

        # Add in early base conditions if combined group period overlaps
        if grp['early_bases']:
            early_change_period = grp['early_bases']['change_period']
            early_base_dts = ca.get_period_dts(early_change_period)
            if list(set(early_base_dts).intersection(comb_dts)):
                bases.append({'wind_dir': grp['early_bases']['wind_dir'],
                              'wind_mean': grp['early_bases']['wind_mean'],
                              'wind_gust': grp['early_bases']['wind_gust']})

    # Check that all group values are significantly different from base
    # conditions (for gusts only)
    for base_dict, value_dict in itertools.product(bases, values):
        if not ch.gust_change(base_dict['wind_gust'], value_dict['wind_gust'],
                              base_dict['wind_mean'], value_dict['wind_mean']):
            return None

    # Ensure all group values are not significantly different from each
    # other (again, gusts only)
    for (values_1, values_2) in itertools.combinations(values, 2):
        if ch.gust_change(values_1['wind_gust'], values_2['wind_gust'],
                          values_1['wind_mean'], values_2['wind_mean']):
            return None

    # Determine whether change increasing or decreasing
    if values[0]['wind_gust'] > bases[0]['wind_gust']:
        chg_type = 'increase'
    else:
        chg_type = 'decrease'

    # If merge possible, take most extreme values
    use_values = values[0]
    for value_dict in values[1:]:
        if any([all([chg_type == 'increase',
                     value_dict['wind_gust'] > use_values['wind_gust']]),
                all([chg_type == 'decrease',
                     value_dict['wind_gust'] < use_values['wind_gust']])]):
            use_values = value_dict

    # Add values to combined group
    comb_grp['wind_dir'] = use_values['wind_dir']
    comb_grp['wind_mean'] = use_values['wind_mean']
    comb_grp['wind_gust'] = use_values['wind_gust']

    return comb_grp


def wxs_merge(comb, wxs, rules):
    """
    Checks if group values are suitable for merging for each weather
    type and creates a combined change group if so.

    Args:
        comb (tuple): Combination of change groups
        wxs (list): Weather changes in change groups
        rules (str): Airport TAF rules (defence, civil or offshore)
    Returns:
        comb_grp (dict): Combined change group
    """
    # Check if groups are consecutive and collect combined dts if so
    comb_period, comb_dts = consecutive(comb)
    if comb_dts is None:
        return None

    # Start with copy of first group as template, adding change period
    comb_grp = copy.deepcopy(comb[0])
    comb_grp['change_period'] = comb_period

    # Loop through all unique weather types
    for wx in wxs:

        # Determine if merge possible for wx and update combined group
        # with suitable values if so
        if wx == 'vis':
            comb_grp = vis_merge(comb, rules, comb_dts, comb_grp)
        elif wx == 'cld':
            comb_grp = cld_merge(comb, rules, comb_dts, comb_grp)
        elif wx == 'wind':
            comb_grp = wind_merge(comb, comb_dts, comb_grp)

        # Break for loop if merge not possible
        if comb_grp is None:
            break

    return comb_grp
