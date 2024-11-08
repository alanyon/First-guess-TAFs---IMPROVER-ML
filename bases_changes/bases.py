"""
Determines base conditions.

Functions:
    better_vis_wx: Determines best vis and sig wx conditions from lists.
    get_base_cld: Gets suitable cloud base conditions.
    get_base_dir: Gets suitable wind direction base conditions.
    get_base_gust: Gets suitable wind gust base conditions.
    get_base_mean: Gets suitable wind mean base conditions.
    get_base_vis_wx: Gets suitable vis and sig wx base conditions.
    shorten_clds: Shortens cloud base values to those in same category.
    shorten_dirs: Shortens wind directions so no sig differences.
    shorten_gusts: Shortens wind gusts so no sig differences.
    shorten_means: Shortens wind means so no sig differences.
    update_wx: Updates significant weather value to be used.
"""
import itertools

import numpy as np

import common.calculations as ca
import common.checks as ch
import common.configs as co


def better_vis_wx(viss, sig_wxs, condition, rules=None):
    """
    Tries to find best conditions in terms of visibility and significant
    weather in the first 4 values. Creates a safer TAF by keeping worst
    conditions in PROB/TEMPO groups.

    Args:
        viss (list): Visbility values
        sig_wxs (list): Significant weather codes
        condition (str): Condition best weather defined by
        rules (str): Airport TAF rules (defence, civil or offshore)
    Returns:
        vis (int): Best visibility value
        sig_wx (str): Best significant weather code
    """
    # Loop through first 4 vis and sig wx values
    for vis, sig_wx in zip(viss, sig_wxs):

        # Define criteria from condition
        if condition == 'light_precip':
            criteria = all(['-' in sig_wx, vis >= viss[0]])
        elif condition == 'mod_precip':
            criteria = all(['+' not in sig_wx, vis >= viss[0]])
        elif condition == 'NSW':
            criteria = all([sig_wx == '', vis == 9999])
        elif condition == 'best':
            criteria = all([(ca.get_vis_cat(vis, sig_wx, rules) >
                             ca.get_vis_cat(viss[0], sig_wxs[0], rules)),
                            sig_wx not in co.PRECIP_CODES])

        # Return vis and sig wx values if criteria met
        if criteria:
            return vis, sig_wx

    # If criteria not met, return None
    return None, None


def get_base_cld(tdf_30, tdf_50):
    """
    Determines suitable value for cloud bases in base conditions.

    Args:
        tdf_30 (pandas.DataFrame): 30th percentile data
        tdf_50 (pandas.DataFrame): 50th percentile data
    Returns:
        base_clds (dict): FEW, SCT, BKN and OVC cloud base values
        base_cld_cat (int): Cloud base TAF category
    """
    # Get required data and shorten if necessary
    clds_3, clds_5, cat, cats_30 = shorten_clds(tdf_30, tdf_50)

    # For defence rules and for most civil categories, just take mean of
    # 3 and 5 okta cloud
    if tdf_50.attrs['rules'] == 'defence' or cat not in [5., 7.]:
        base_cld_3 = ca.round_cld(clds_3.mean())
        base_cld_5 = ca.round_cld(clds_5.mean())
        base_cld_1 = base_cld_3
        base_cld_8 = 5000

    # For civil rules, need to account for CAVOK
    else:

        # If 5 okta cld BKN between 1500 and 5000ft, force to be SCT -
        # this prevents TAF going bust from any CAVOK obs
        if cat == 5:
            base_cld_3 = ca.round_cld(clds_5.mean())
            base_cld_5 = 5000
            base_cld_1, base_cld_8  = base_cld_3, base_cld_5

        # For CAVOK category
        else:

            # Only report CAVOK if CAVOK categories also in percentiles
            # data and lasts for at least 4 hours
            if (cats_30.nunique() == 1
                and list(cats_30)[0] == 7.
                and len(cats_30) >= 4):

                # Set all cloud to 5000ft to indicate CAVOK
                base_cld_1, base_cld_3, base_cld_5, base_cld_8 = [5000] * 4

            # Otherwise, avoid CAVOK by setting FEW cloud to 4500ft
            else:
                base_cld_1 = 4500
                base_cld_3, base_cld_5, base_cld_8 = [5000] * 3

    # Avoid two cloud groups with similar values (e.g. SCT035 BKN040)
    # - only allow two cloud groups if in different categories
    if base_cld_3 != base_cld_5:
        cat_3 = ca.get_cld_cat(base_cld_3, base_cld_3, tdf_50.attrs['rules'])
        cat_5 = ca.get_cld_cat(base_cld_5, base_cld_5, tdf_50.attrs['rules'])
        if cat_3 == cat_5:
            base_cld_3 = base_cld_5
            base_cld_1 = base_cld_5

    # Collect cloud values into dictionary
    base_clds = {'cld_1': base_cld_1, 'cld_3': base_cld_3,
                 'cld_5': base_cld_5, 'cld_8': base_cld_8}

    # Finally, get cloud category of base cloud values - use 1 okta
    # cloud instead of 3 okta in case FEW cloud specified to avoid CAVOK
    base_cld_cat = ca.get_cld_cat(base_cld_1, base_cld_5,
                                  tdf_50.attrs['rules'])

    return base_clds, base_cld_cat


def get_base_dir(tdf):
    """
    Determines a suitable 'mean' wind direction based on a list of
    directions in degrees between 0 and 360. This is difficult due to
    the fact that the smallest difference between 2 directions could be
    clockwise or anticlockwise, and has to be calculated using modular
    arithmetic to 'wrap around' 360. Also, a mean value starts to make
    less sense with differences of more than 90 degrees (e.g. what is
    the mean of 360 and 180 degrees? Could be 90 or 270, depending which
    way you go). So in these cases, the first value in the list is used
    rather than the mean.

    Args:
        tdf (pandas.DataFrame): IMPROVER and airport data
    Returns:
        base_dir (int): Wind direction to be used in base conditions
        num_dirs (int): Number of wind directions used to get base value
    """
    # Get wind directions from dataframe, shorten if necessary and find
    # maximum directional difference in shortened dataframe
    wind_dirs, max_diff = shorten_dirs(tdf)

    # After wind directions have been shortened, the maximum difference
    # between directions is 60 degrees. However, the absolute difference
    # between directions can be more (e.g, 350 and 020) - in these
    # cases, the 'mean' is calculated by adding the mean difference from
    # the 'minimum' value to the 'minimum' value (in the following list
    # of directions - 020, 350, 010 - 350 would be treated as the
    # minimum value, which can be found by getting the maximum value of
    # the biggest difference combination (350, 020) in this example)
    if (max(wind_dirs) - min(wind_dirs)) > 180:
        diffs_from_min = [(w_dir - max(max_diff)) % 360 for w_dir in wind_dirs]
        mean_dir = (max(max_diff) + np.mean(diffs_from_min)) % 360

    # If the directions do not cross 360, we can just take a normal
    # mean of the directions
    else:
        mean_dir = wind_dirs.mean()

    # Round mean wind direction to nearest 10 to get sensible value to
    # use in TAF
    base_dir = int(round((mean_dir / 10), 0) * 10)

    # Make 360 if result is 0
    if base_dir == 0:
        base_dir = 360

    # Also need number of wind means used for base gust calculation
    num_dirs = len(wind_dirs)

    return base_dir, num_dirs


def get_base_gust(base_mean, tdf):
    """
    Determines suitable value for wind gust in base conditions.

    Args:
        base_mean (int): base wind mean value
        tdf (pandas.DataFrame): IMPROVER and airport data
    Returns:
        base_mean (int): Updated base wind mean value
        base_gust (int): Base gust value
    """
    # Get required data from dataframe
    wind_means = tdf['wind_mean']
    wind_gusts = tdf['wind_gust']

    # Only include gusts in base conditions if at least 4 consecutive
    # gusts significant with high enough means
    if all(gust >= 25 and gust >= base_mean + 10 and wmean >= 15
           for wmean, gust in zip(wind_means[:4], wind_gusts[:4])):

        # Shorten gusts if necessary
        wind_gusts = shorten_gusts(wind_gusts)

        # Take mean of all gusts
        base_gust = int(round(wind_gusts.mean(), 0))

        # Ensure base mean high enough with gust
        base_mean = max(base_mean, 15)

    # Otherwise, set base gust to be the same as base mean
    else:
        base_gust = base_mean

    return base_mean, base_gust


def get_base_mean(tdf, becmg=False):
    """
    Determines suitable value for wind mean in base conditions.

    Args:
        tdf (pandas.DataFrame): IMPROVER and airport data
        becmg (bool): Indicator for whether BECMG group
    Returns:
        base_mean (int): Wind mean to be used in base conditions
        num_means (int): Number of wind means used to get base value
    """
    # Shorten wind means collection if necessary
    wind_means = shorten_means(tdf)

    # Get mean wind mean
    base_mean = int(round(wind_means.mean(), 0))

    # Force base to 9kt or 19kt if just over - creates a safer TAF (do
    # not apply to BECMG groups)
    if not becmg:
        if 10 <= base_mean <= 13 and list(wind_means)[0] <= 16:
            base_mean = 9
        elif 20 <= base_mean <= 23 and list(wind_means)[0] <= 26:
            base_mean = 19

    # Also need number of wind means used for base gust calculation
    num_means = len(wind_means)

    return base_mean, num_means


def get_base_vis_wx(tdf_50, old_bases):
    """
    Determines suitable value for visibility and significant weather in
    base conditions.

    Args:
        tdf_50 (pandas.DataFrame): 50th percentile data
        old_bases (dict): Previous base conditions
    Returns:
        base_vis (int): Visibility value to be used in base conditions
        base_vis_cat (float): Vis category of base_vis
        base_wx (str): Significant weather to used to get base value
        implied_wx (str): Implied wx (can be different to base_wx if
                          base conditions used for BECMG group)
    """
    # Get (up to) first 4 hours of vis and sig wx data from dataframe
    if old_bases:
        viss, sig_wxs = [], []
        for ind, cat in enumerate(tdf_50['vis_cat'][:4]):
            if cat == list(tdf_50['vis_cat'])[0]:
                viss.append(list(tdf_50['vis'])[ind])
                sig_wxs.append(list(tdf_50['sig_wx'])[ind])
    else:
        viss, sig_wxs = list(tdf_50['vis'])[:4], list(tdf_50['sig_wx'])[:4]

    # Update for cases when precip consistently forecast - allow for
    # multiple wx in single sig wx entry (e.g. 'DZ FG')
    if all(any(wx in co.PRECIP_CODES for wx in sig_wx.split())
           for sig_wx in sig_wxs):
        # Look for light precip first
        base_vis, base_wx = better_vis_wx(viss, sig_wxs, 'light_precip')

        # If no light precip found, look for mod precip
        if not base_vis:
            base_vis, base_wx = better_vis_wx(viss, sig_wxs, 'mod_precip')

    # Update for cases when precip in first wx code but not consistent
    elif any(wx in co.PRECIP_CODES for wx in sig_wxs[0].split()):
        # First look for 9999 NSW
        base_vis, base_wx = better_vis_wx(viss, sig_wxs, 'NSW')

        # If not, just look for best conditions
        if not base_vis:
            base_vis, base_wx = better_vis_wx(viss, sig_wxs, 'best',
                                              rules=tdf_50.attrs['rules'])

    # If no precip, try to find 9999 NSW (avoids, e.g., 8000 BECMG 9999)
    else:
        base_vis, base_wx = better_vis_wx(viss, sig_wxs, 'NSW')

    # If base values updated with None, just use first value
    if base_vis is None:
        base_vis, base_wx = viss[0], sig_wxs[0]

    # Extra condition to avoid having showers in the base conditions
    new_base_wx = []
    for wx in base_wx.split():
        wx = wx.replace('SH', '')
        wx = wx.replace('TS', '')
        new_base_wx.append(wx)
    base_wx = ' '.join(new_base_wx)

    # Get implied wx (can be different to wx reported in BECMG groups)
    base_wx, implied_wx = update_wx(base_wx, base_vis, old_bases,
                                    tdf_50.attrs['rules'])

    # Finally, get visibility TAF category
    base_vis_cat = ca.get_vis_cat(base_vis, base_wx, tdf_50.attrs['rules'])

    return base_vis, base_vis_cat, base_wx, implied_wx


def shorten_clds(tdf_30, tdf_50):
    """
    Shortens collections of cloud base data to the point in which there
    are no significant differences.

    Args:
        tdf_30 (pandas.DataFrame): 30th percentile data
        tdf_50 (pandas.DataFrame): 50th percentile data
    Returns:
        clds_3 (pandas.Series): 50th percentile >= 3 okta cloud bases
        clds_5 (pandas.Series): 50th percentile >= 5 okta cloud bases
        cat (float): 50th percentile cloud category
        cats_30 (pandas.Series): 30th percentile cloud categories
    """
    # Get required 30th and 50th percentile data
    cats_30 = tdf_30['cld_cat']
    clds_3, clds_5, cats = tdf_50['cld_3'], tdf_50['cld_5'], tdf_50['cld_cat']

    # Start with default of sig difference and update as necessary
    sig_diff = True
    while sig_diff:

        # If all cloud cats the same, keep all values and end while loop
        if cats.nunique() == 1:

            # As all cats are the same, just use the first one
            cat = list(cats)[0]
            sig_diff = False

        # Otherwise, remove last values and move to next iteration
        else:
            clds_3, clds_5, cats = clds_3[:-1], clds_5[:-1], cats[:-1]
            cats_30 = cats_30[:-1]

    return clds_3, clds_5, cat, cats_30


def shorten_dirs(tdf):
    """
    Shortens collections of wind directions to the point in which there
    are no significant differences.

    Args:
        tdf (pandas.DataFrame): IMPROVER and airport data
    Returns:
        wind_dirs (Pandas.Series): Wind directions (shortened)
        max_diff (tuple): 2-element combination of wind directions with
                          biggest difference
    """
    # Get required data from dataframe
    wind_dirs = tdf['wind_dir']
    wind_means = tdf['wind_mean']

    # Start with default of sig difference and update as necessary
    sig_diff = True
    while sig_diff:

        # Do not do anything if 1 or 0 values
        if len(wind_dirs) <= 1:
            return wind_dirs, None

        # Combine directions and means
        dirs_means = list(zip(wind_dirs, wind_means))

        # Get every 2-element combination of directions/means, but force
        # wind means to be at least 10kt for civil or 15kt for defence -
        # this limits the differences between directions to no more than
        # 60 degrees but also allows the possiblity of limiting to 30
        # degrees if wind means are strong enough
        if tdf.attrs['rules'] == 'defence':
            lim = 15
        else:
            lim = 10
        combs = [{'dir_comb': (comb[0][0], comb[1][0]),
                  'mean_comb': (max(comb[0][1], lim), max(comb[1][1], lim)),
                  'diff': ca.diff_calc((comb[0][0], comb[1][0]))}
                 for comb in itertools.combinations(dirs_means, 2)]

        # Check if any significant directional differences
        sig_diff = any(ch.dir_change(comb['dir_comb'], comb['mean_comb'],
                                     tdf.attrs['rules'])
                       for comb in combs)

        # Shorten wind directions if necessary and move to next
        # iteration of while loop
        if sig_diff:
            wind_dirs = wind_dirs[:-1]

    # Get combination of directions with maximum difference for mean
    # diection calculations
    max_diff = combs[np.argmax([comb['diff'] for comb in combs])]['dir_comb']

    return wind_dirs, max_diff


def shorten_gusts(wind_gusts):
    """
    Shortens collections of wind gusts to the point in which there are
    no significant differences.

    Args:
        wind_gusts (Pandas.Series): Wind gusts
    Returns:
        wind_gusts (Pandas.Series): Shortened wind gusts
    """
    # Start with default of sig difference and update as necessary
    sig_diff = True
    while sig_diff:

        # Do not do anything if 1 or 0 values
        if len(wind_gusts) <= 1:
            return wind_gusts

        # Get wind mean differences between every pair of values
        max_diff = max(wind_gusts) - min(wind_gusts)

        # Remove last wind mean if max difference >= 10kt and move to
        # next iteration
        if max_diff >= 10:
            wind_gusts = wind_gusts[:-1]

        # Otherwise, end while loop
        else:
            sig_diff = False

    return wind_gusts


def shorten_means(tdf):
    """
    Shortens collections of wind means to the point in which there are
    no significant differences.

    Args:
        wind_means (Pandas.Series): Wind means
    Returns:
        wind_means (Pandas.Series): Shortened wind means
    """
    # Get wind means from dataframe
    wind_means = tdf['wind_mean']

    # Start with default of sig difference and update as necessary
    sig_diff = True
    while sig_diff:

        # Do not do anything if 1 or 0 values
        if len(wind_means) <= 1:
            return wind_means

        # Remove last wind mean if sig difference between max and min
        if ch.mean_change(max(wind_means), min(wind_means),
                          tdf.attrs['rules']):
            wind_means = wind_means[:-1]

        # Otherwise, end while loop
        else:
            sig_diff = False

    return wind_means


def update_wx(base_wx, base_vis, old_bases, rules):
    """
    Updates significant weather value to be used in change group and
    introduces 'implied_wx' as weather that is implied in the change
    group e.g. '4000 BR BECMG 2000' implies BR is still present even
    though it is not reported in the BECMG group.

    Args:
        base_wx (str): Significant weather in the base conditions
        old_bases (dict): Previous base conditions
    Returns:
        base_wx (str): Updated base significant weather
        implied_wx (str): Weather implied in base conditions
    """
    # If old bases exists, this implies BECMG base conditions are being
    # sought, so implied weather might be different to that reported
    if old_bases:

        # Get base and implied wx codes based on significant changes
        base_wx, implied_wx = ca.get_new_wxs(
            old_bases['implied_sig_wx'].split(), base_wx.split(), base_vis,
            rules
        )

    # If old_bases does not exist and base_wx is empty string, implied
    # wx is NSW
    elif not base_wx:
        implied_wx = 'NSW'

    # Otherwise, implied wx is the same as base wx
    else:
        implied_wx = base_wx

    return base_wx, implied_wx
