"""
Checking and adjusting functions used in TAF generation.

Functions:
    check_mean: Determines if significant change between 2 mean winds.
    check_mist_fog: Checks that appropriate mist/fog code used.
    check_rate: Checks that appropriate precip rate is used.
    check_vis_cld_wind: Adjusts visibility based on cloud and wind.
    check_vis_sig_wx: Adjusts visibility based on sig wx.
    cld_change: Checks for significant cloud changes.
    dir_change: Checks if difference in wind direction significant.
    gust_change: Checks for significant gust changes.
    mean_change: Checks for significant wind mean changes.
    vis_change: Checks for significant visibility changes.
"""
import common.calculations as ca
import common.configs as co


def check_mean(mean_1, mean_2, rules):
    """
    Determines if mean_2 is a significant change from mean_1.

    Args:
        mean_1 (int or float): First mean wind value
        mean_2 (int or float): Second mean wind value
        rules (str): Airport TAF rules (civil, offshore or defence)
    Returns:
        change (bool): Indication for whether mean speeds are
                       significantly different from each other
    """
    # Civil rules just requires a difference of more than 10kt
    change = abs(mean_2 - mean_1) >= 10

    # Extra condition for defence that mean should be >=15kts before or
    # after
    if rules == 'defence':
        change = change and max(mean_1, mean_2) >= 15

    return change


def check_mist_fog(vis, temp, rules, wx_str):
    """
    Checks that appropriate mist/fog code used for sufficiently low
    visibilities. For defence rules, a weather code is required for all
    visibilities < 9999 - this is currently defaulted to HZ in the
    absence of other weather codes.

    Args:
        vis (int): Visibility
        temp (float): Air temperature
        rules (str): TAF rules for airport (defence, civil or offshore)
        wx_str (str): Sig wx code
    Returns:
        wx_str (str): Updated sig wx code
    """
    # Appropriate fog code to use if necessary
    fog_code = 'FZFG' if temp < 0 else 'FG'

    # Get precip and non-precip codes from string
    precip_wx, non_precip_wx = '', ''
    for wx_code in wx_str.split():
        if wx_code in co.PRECIP_CODES:
            precip_wx += wx_code
        else:
            non_precip_wx += wx_code

    # Add fog if necessary when precip code present
    if all([vis < 1000, precip_wx, 'FG' not in non_precip_wx]):
        non_precip_wx = fog_code

    # Add mist if visibility less than 5000m and light precip
    if all([1000 <= vis <= 5000, '-' in precip_wx, 'BR' not in non_precip_wx]):
        non_precip_wx = 'BR'

    # For cases in which just non-precip wx in code
    if not precip_wx:

        # For visibilities less than 1000m, some kind of fog should be
        # added
        if vis < 1000:
            non_precip_wx = fog_code

        # Rules for mist visibilities
        elif vis <= 5000:

            # For defence, to avoid BR/HZ complications
            if rules == 'defence' and vis == 5000:
                non_precip_wx = 'HZ'

            # Otherwise, just use mist
            else:
                non_precip_wx = 'BR'

        # For defence, wx code needed for all vis<9999 (default to HZ)
        elif rules == 'defence' and vis < 9999:
            non_precip_wx = 'HZ'

    # Join precip and non-precip back up into single string
    wx_codes = [non_precip_wx]
    if precip_wx:
        wx_codes.insert(0, precip_wx)
    new_wx_str = ' '.join(wx_codes)

    return new_wx_str


def check_rate(precip_rate, vis, temp, wx_str):
    """
    Checks that appropriate precip rate is used in sig wx code.

    Args:
        precip_rate (float): Precipitation rate
        vis (int): Visibility
        temp (float): Air temperature
        wx_str (str): Sig wx code
    Returns:
        wx_str (str): Updated sig wx code
    """
    # Empty list to add updated wx components to
    new_wxs = []

    # Split wx str into separate components
    for wx_code in wx_str.split():

        # Ignore if not a precip code
        if wx_code not in co.PRECIP_CODES:
            new_wxs.append(wx_code)
            continue

        # Remove rate symbols ('-' and '+') temporarily for processing
        wx_code_clean = wx_code.replace('-', '').replace('+', '')

        # Assume rate insignificant if <= 0.1mm/hr (except for drizzle)
        if precip_rate <= 0.1 and 'DZ' not in wx_code_clean:
            continue

        # Define rules for rates...
        # For convective precip
        if any(wx in wx_code_clean for wx in ['SH', 'TS']):
            limits, assigned_rates, drizzle = [2, 10], ['-', '', '+'], False

        # For dynamic precip
        elif any(wx in wx_code_clean for wx in ['RA', 'SN', 'GS']):
            limits, assigned_rates, drizzle = [0.5, 4], ['-', '', '+'], False

        # For drizzle - no defined rate for drizzle to be light/mod/heavy,
        # so base on visibility (leaving out heavy drizzle)
        else:
            limits, assigned_rates, drizzle = [5000], ['-', ''], True

        # Determine rate based on rules
        for limit, assigned_rate in zip(limits, assigned_rates):
            if drizzle:
                if vis >= limit:
                    rate = '-'
                    break
                rate = ''
            elif precip_rate < limit:
                rate = assigned_rate
                break
            rate = assigned_rates[-1]

        # If rain or drizzle is predicted and air temp is < 0, should be
        # freezing rain/drizzle
        if temp < 0 and wx_code_clean in ['RA', 'DZ']:
            wx_code_clean = f'FZ{wx_code_clean}'

        # Add new rate to wx code
        wx_code = f'{rate}{wx_code_clean}'

        # Add updated code to wx str
        if wx_code:
            new_wxs.append(wx_code)

    # Convert new wx codes back to string
    new_wx_str = ' '.join(new_wxs)

    return new_wx_str


def check_vis_cld_wind(vis, cld_5, wind_mean):
    """
    Adjusts visibility value based on cloud value and wind mean. Low
    visibility correlates with low cloud, especially with light wind.
    Visibility is adjusted rather than cloud because cloud tends to
    verify better than visibility.

    Args:
        vis (int): Visbility
        cld_5 (int): >= 5 okta cloud base
        wind_mean (float): wind mean speed
    Return:
        vis (int): Updated visibility value
    """
    # Define rules depending on cloud base height
    if cld_5 < 100:
        rules =  {'wind': [3, 6, 10, 15], 'vis': [300, 800, 1400, 3000, 5000]}
    elif cld_5 < 200:
        rules =  {'wind': [5, 10, 15], 'vis': [1200, 3000, 5000, 9000]}
    elif cld_5 < 400:
        rules = {'wind': [5, 10], 'vis': [5000, 9000, 9999]}
    else:
        return vis

    # Set limits on visibility based on cloud rules
    for wind_limit, vis_limit in zip(rules['wind'], rules['vis']):
        if wind_mean <= wind_limit:
            vis = min(vis, vis_limit)
            break
        vis = min(vis, rules['vis'][-1])

    return vis


def check_vis_sig_wx(sig_wx, vis):
    """
    Checks visibility against sig wx code, ensuring sensible values.
    Visibility is adjusted downwards if necessary.
    Follows these guidelines:
    https://metoffice.sharepoint.com/:w:/r/sites/FSDCivilCommsSite/
    _layouts/15/Doc.aspx?sourcedoc=%7B15ED6B87-9468-4F62-A341-
    858A88062D82%7D

    Args:
        sig_wx (str): Significant weather code
        vis (int): Visibility value
    Return:
        vis (int): Updated visibility value
    """
    # Leave unadjusted for non-precip codes
    if sig_wx in co.NON_PRECIP_CODES:
        return vis

    # Determine precip rate
    rate = ca.get_rate(sig_wx)

    # Define rates used
    test_rates = ['light', 'moderate', 'heavy']

    # Covers all codes including rain
    if 'RA' in sig_wx:

        # Covers RASN and SHRASN
        if 'SN' in sig_wx:
            threshs, new_viss = [9000, 4500, 2500], [8000, 4000, 2000]

        # Covers all other rain codes
        else:
            threshs, new_viss = [9999, 8000, 4000], [9000, 7000, 3000]

    # Covers SN and SHSN
    elif 'SN' in sig_wx:
        threshs, new_viss = [6000, 2000, 800], [4000, 1500, 800]


    # Covers SHGS (not sure of good max value here) and DZ
    elif any(ele in sig_wx for ele in ['GS', 'DZ']):
        threshs, new_viss = [8000, 5000, 1500], [7000, 4000, 1500]

    # Update visibility if necessary
    for test_rate, thresh, new_vis in zip(test_rates, threshs, new_viss):
        if rate == test_rate and vis >= thresh:
            vis = new_vis
            break

    return vis


def cld_change(old_cat, new_cat, rules):
    """
    Checks if differences in cloud conditions constitutes a significant
    change. Due to the CAVOK/NSC rule, changes between cats 5 and 6 and
    between cats 6 and 7 are not significant in civil/offshore TAFs.
    Otherwise, just an increase or decrease in category is considered.

    Args:
        old_cat (int or float): Old cloud category
        new_cat (int or float): New cloud category
        rules (str): TAF rules for airport (civil, offshore or defence)
    Returns:
        change (int): Indication of significant change (if any)
    """
    # For all defence changes and civil/offshore changes to and from
    # lower categories, simply look at cloud category numbers
    if rules == 'defence' or any(cat <= 4. for cat in [old_cat, new_cat]):
        change = int(new_cat - old_cat)

    # For other civil/offshore changes, need to account for CAVOK rules
    elif old_cat == 5 and new_cat == 7:
        change = 1
    elif old_cat == 7 and new_cat == 5:
        change = -1
    else:
        change = 0

    return change


def dir_change(wind_dirs, wind_means, rules):
    """
    Determines if an difference in wind direction is a significant
    change (depends on wind means).

    Args:
        wind_dirs (tuple): Wind directions
        wind_means (tuple): Wind means
        rules (str): TAF rules for airport
    Returns:
        sig_diff (bool): Indication for whether difference significant
    """
    # Get difference in wind directions in degrees
    diff = ca.diff_calc(wind_dirs)

    # If difference is less than 30 degrees, it is never significant
    if diff < 30:
        sig_diff = False

    # For defence, diffs >= 30 degrees and means >= 15kt are significant
    elif rules == 'defence':
        sig_diff = any(wind_mean >= 15 for wind_mean in wind_means)

    # For civil/offshore, diffs between 30 and 60 degrees only
    # significant if means >= 20kt
    elif diff < 60:
        sig_diff = any(wind_mean >= 20 for wind_mean in wind_means)

    # For civil/offshore, diffs >= 60 degrees only significant if means
    # >= 10kt
    else:
        sig_diff = any(wind_mean >= 10 for wind_mean in wind_means)

    return sig_diff


def gust_change(old_gust, new_gust, old_mean, new_mean, becmg=False):
    """
    Checks for significant gust changes (increase or decrease).

    Args:
        old_gust (int or float): Old gust
        new_gust (int or float): New gust
        old_mean (int or float): Old mean speed
        new_mean (int or float): New wind speed
        becmg (bool): Indication of whether being used for BECMG group
    Returns:
        change (int): Type of significant change (+ve for increase, -ve
                      for decrease, 0 for no change)
    """
    # Conditions for an increase in gust strength - the condition that
    # the new gust is at least 13kt higher than the old mean is due to
    # best practice guidelines, whic dictate that a change group with
    # increased gusts should also include mean wind that is at least 3kt
    # higher
    if all([new_gust >= 25, new_gust >= old_gust + 10,
            new_gust >= old_mean + 13]):
        change = int((new_gust - max(old_gust, 15)) / 10)

        # Extra condition for BECMG changes that mean should be >=15kt
        if becmg and new_mean < 15:
            change = 0

    # Conditions for a decrease in gusts - first condition is that gusts
    # are reported in old wind group (normally base conditions)
    elif all([old_gust >= 25, old_gust >= old_mean + 10]):

        # If gust is not at least 10kt more than the mean, it will not
        # be reported, which would be a significant decrease
        if new_gust < new_mean + 10:
            change = min(-1, int((new_mean - old_gust) / 10))

        # Otherwise, if new gust is at least 10kt less than the old gust
        # this would also be a significant decrease
        elif new_gust <= old_gust - 10:
            change = int((new_gust - old_gust) / 10)

        # Otherwise change is not significant
        else:
            change = 0

        # Overiding the above is if the new mean below 10kt - in theory,
        # any gusts 10kt more than the mean should be reported, but this
        # is unlikely if mean is less than 10kt, so this extra condition
        # creates a safer TAF even though it's not technically necessary
        if new_mean < 10:
            change = min(-1, int((new_mean - old_gust) / 10))

    # Otherwise, change is not significant
    else:
        change = 0

    return change


def mean_change(mean_1, mean_2, rules):
    """
    Determines if mean_2 is a significant change from mean_1.

    Args:
        mean_1 (int or float): First mean wind value
        mean_2 (int or float): Second mean wind value
        rules (str): Airport TAF rules (civil, offshore or defence)
    Returns:
        change (bool): Indication for whether mean speeds are
                       significantly different from each other
    """
    # Civil rules just requires a difference of more than 10kt
    change = abs(mean_2 - mean_1) >= 10

    # Extra condition for defence that mean should be >=15kts before or
    # after
    if rules == 'defence':
        change = change and max(mean_1, mean_2) >= 15

    return change


def vis_change(old_cat, new_cat):
    """
    Checks for significant difference in two visibilities, and whether
    difference is and increase or decrease in TAF category.

    Args:
        old_cat (int or float): Old visibility category
        new_cat (int or float): New visibility category
    Returns:
        change (int): Indication of significant change (if any)
    """
    # Change is simply different between categories (make int to avoid
    # 0.5 categories)
    change = int(new_cat) - int(old_cat)

    return change
