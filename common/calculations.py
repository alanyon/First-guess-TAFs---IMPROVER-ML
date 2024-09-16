"""
Calculating functions used in TAF generation.

Functions:
    assign_wx: Assigns appropriate weather codes.
    diff_calc: Calculates smallest modular difference between wind dirs.
    get_cld_cat: Determines cloud base TAF category.
    get_new_wxs: Determines wx code and implied wx code for BECMG group.
    get_non_precip_change: Determines if non-precip change significant.
    get_period_dts: Finds all dts covered in change group period.
    get_precip_change: Determines if precip change significant.
    get_rate: Determines precip rate from precip wx code.
    get_vis_cat: Determines visibility TAF category.
    mp_queue: Wrapper function for multiprocessing.
    round_cld: Rounds cloud appropriately.
    round_vis: Rounds visibility appropriately.
    use_cavok: Determines whether CAVOK/NSC/NSW is needed.
"""
import math
import sys
from datetime import datetime, timedelta

from dateutil.rrule import HOURLY, rrule

import common.configs as co


def assign_wx(change, new_code, old_code):
    """
    Assigns appropriate weather codes to new and reported implied
    weather.

    Args:
        change (bool): Indicator for whether change significant
        new_code (str): New weather code
        old_code (str): Old weather code
    Returns:
        new_wx (str): New reported weather code
        implied_wx (str): New implied weather code
    """
    # If change significant, update reported and implied weather with
    # new code
    if change:
        new_wx, implied_wx = new_code, new_code

    # If change not significant, implied weather is unchanged and there
    # is no new weather
    else:
        new_wx = ''
        implied_wx = old_code

    return new_wx, implied_wx


def diff_calc(comb):
    """
    Calculates smallest modular difference (wraps around at 360) between
    2 wind directions (could be clockwise or anti-clockwise).

    Args:
        comb (tuple): Two wind directions to compare
    Returns:
        diff (int): Smallest modular difference between the 2 values
    """
    # Get absolute difference
    abs_diff = max(comb) - min(comb)

    # Just take absolute difference if not more than 180 degrees
    if abs_diff <= 180:
        diff = abs_diff

    # Otherwise, take smaller difference going round the other way
    else:
        diff = (min(comb) - max(comb)) % 360

    return diff


def get_cld(cld_cat, rules):
    """
    Returns suitable cloud value based on cloud category.

    Args:
        cld_cat (float): Cloud base category
        rules (str): TAF rules for airport (defence, civil or offshore)
    Return:
        cld_3 (int): >= 3 okta cloud base
        cld_5 (int): >= 5 okta cloud base
    """
    all_cld_dicts = {
        'defence': {7.: [3000, 5000], 6.: [2000, 2000], 5.: [1200, 1200], 
                    4.: [600, 600], 3.: [400, 400], 2.: 1200, 1.: 500},
        'offshore': {7.: [5000, 5000], 6.: [4000, 5000], 5.: [3000, 3000], 
                     4.: [1400, 1400], 3.: [900, 900], 2.: [600, 600], 
                     1.: [300, 300], 0.: [100, 100]},
        'civil': {7.: [5000, 5000], 6.: [4000, 5000], 5.: [3000, 3000], 
                  4.: [1400, 1400], 3.: [800, 800], 2.: [400, 400], 
                  1.: [100, 100]}
    }

    # Get cloud dictionary for rules
    cld_dict = all_cld_dicts.get(rules, {})

    # Get cloud value, ensuring it is within range of dictionary
    if cld_cat > max(cld_dict):
        cld_3, cld_5 = cld_dict.get(max(cld_dict))
    elif cld_cat < min(cld_dict):
        cld_3, cld_5 = cld_dict.get(min(cld_dict))
    else:
        cld_3, cld_5 = cld_dict.get(cld_cat)
    
    return cld_3, cld_5


def get_cld_cat(cld_3, cld_5, rules):
    """
    Determines cloud base TAF category based on cloud values.

    Args:
        cld_3 (int): >= 3 okta cloud base
        cld_5 (int): >= 5 okta cloud base
        rules (str): TAF rules for airport (defence, civil or offshore)
    Return:
        cld_cat (float): Cloud base category
    """
    # Define thresholds, categories and cloud to consider
    # For defence TAFs
    if rules == 'defence':

        # For defence TAFs, both cloud amounts are always considered
        clds = [cld_3, cld_5]

        # Thresholds/categories for defence
        thresholds = [2500, 1500, 700, 500, 300, 200, 0]
        categories = [7., 6., 5., 4., 3., 2., 1.]

    # For civil TAFs
    else:

        # For civil TAFs, mainly only 5 okta cloud considered
        clds = [cld_5]

        # But 3 okta cloud considered for CAVOK conditions
        if cld_3 >= 5000:
            high_cat = 7.
        else:
            high_cat = 6.

        # Offshore thresholds/categories
        if rules == 'offshore':
            thresholds = [5000, 1500, 1000, 700, 500, 200, 0]
            categories = [high_cat, 5., 4., 3, 2., 1., 0.]

        # Main civil thresholds/categories
        else:
            thresholds = [5000, 1500, 1000, 500, 200, 0]
            categories = [high_cat, 5., 4., 3., 2., 1.]

    # Determine category based on thresholds and categories
    for threshold, category in zip(thresholds, categories):
        if all(cld >= threshold for cld in clds):
            cld_cat = category
            break

    return cld_cat


def get_new_wxs(wxs_old, wxs_new, vis_new, rules):
    """
    Determines if two sets of wx codes are significantly different from
    each other, separating precip from non-precip codes, then assigns
    appropriate wx codes to new wx code and implied wx code.

    Args:
        wxs_old (list): Old weather codes
        wxs_new (list): New weather codes
        vis_new (int): New visibility
        rules (str): Airport TAF rules (defence, civil or offshore)
    Return:
        new_wx (str): New wx that will be reported in TAF
        implied_wx (str): Wx that is implied even if it will not be
                          reported in TAF
    """
    # Change empty strings to 'NSW' so it is recognised as wx
    if not wxs_old:
        wxs_old = ['NSW']
    if not wxs_new:
        wxs_new = ['NSW']

    # Split into precip and non-precip codes (assumes max of 1 precip
    # and 1 non-precip code for each)
    try:
        old_precip = str(*[wx for wx in wxs_old if wx in co.PRECIP_CODES])
        new_precip = str(*[wx for wx in wxs_new if wx in co.PRECIP_CODES])
        old_non_precip = str(*[wx for wx in wxs_old
                               if wx in co.NON_PRECIP_CODES])
        new_non_precip = str(*[wx for wx in wxs_new
                               if wx in co.NON_PRECIP_CODES])
    except TypeError:
        print('Error in get_new_wxs: Probably more than 1 wx in one of '
              'the lists:')
        sys.exit()

    # Determine if changes are significant for precip and non-precip,
    # and assign weather codes as necessary
    new_wx_precip, implied_wx_precip = get_precip_change(new_precip,
                                                         old_precip)
    new_wx_non_precip, implied_wx_non_precip = get_non_precip_change(
        new_non_precip, old_non_precip, old_precip, new_wx_precip, vis_new,
        rules
    )

    # Convert lists to strings
    new_wx = ' '.join([new_wx_precip, new_wx_non_precip]).strip()
    implied_wx = ' '.join([implied_wx_precip, implied_wx_non_precip]).strip()

    return new_wx, implied_wx


def get_non_precip_change(new_non_precip, old_non_precip, old_precip,
                          new_wx_precip, vis_new, rules):
    """
    Selects suitable non-precip weather codes for what is reported in
    the TAF and what is implied.

    Args:
        new_non_precip (str): New non-precip code
        old_non_precip (str): Old non-precip code
        old_precip (str): Old precip code
        new_wx_precip (str): Chosen new precip code
        vis_new (int): New visibility value
        rules (str): TAF rules for airport (civil, offshore or defence)
    Return:
        new_wx (str): Weather code to report
        implied_wx (str): Implied weather
    """
    # For cases with non-precip code in new wx and old wx, codes just
    # need to be different to be significant
    if new_non_precip and old_non_precip:
        change = new_non_precip != old_non_precip

    # Change from precip to non-precip always significant
    elif new_non_precip and old_precip:
        change = True

    # Otherwise, change is not valid
    else:
        change = False

    # Determine suitable codes to use for non-precip wx
    new_wx, implied_wx = assign_wx(change, new_non_precip, old_non_precip)

    # Need to ensure mist or fog is turned off if visibility has
    # increased - sometimes an issue if no significant precip change -
    # e.g. 800 -RA FG to 6000 -RA would become 6000 - in this case, make
    # it 6000 NSW (i.e. turn off FG and -RA)
    if 'FG' in old_non_precip:
        if 1000 <= vis_new <= 5000 and not new_wx:
            new_wx = 'BR'
            implied_wx = 'BR'
        elif vis_new > 5000 and not new_wx:
            if rules == 'defence':
                new_wx = 'HZ'
                implied_wx = 'HZ'
            else:
                new_wx = 'NSW'
                implied_wx = 'NSW'
    elif 'BR' in old_non_precip:
        if vis_new > 5000 and not new_wx:
            if rules == 'defence':
                new_wx = 'HZ'
                implied_wx = 'HZ'
            else:
                new_wx = 'NSW'
                implied_wx = 'NSW'

    # Ensure NSW not used when precip code already chosen
    if new_wx_precip and new_wx == 'NSW':
        new_wx = ''

    return new_wx, implied_wx


def get_period_dts(period):
    """
    Finds all dts covered in change group period.

    Args:
        period (list): Change group period
    Return:
        group_dts (dateutil.rrule): dts in period
    """
    # Get start and end of period, converting to datetime objects
    g_start, g_end = [datetime(cdt.year, cdt.month, cdt.day, cdt.hour)
                       for cdt in period]

    # Get all dts between start and end
    group_dts = rrule(HOURLY, dtstart=g_start, until=g_end-timedelta(hours=1))

    return group_dts


def get_precip_change(new_precip, old_precip):
    """
    Selects suitable precip weather codes for what is reported in the
    TAF and what is implied.

    Args:
        new_precip (str): New precip code
        old_precip (str): Old precip code
    Return:
        new_wx (str): Weather code to report
        implied_wx (str): Implied weather
    """
    # For cases with precip code in new wx and old wx, rates need to be
    # considered
    if new_precip and old_precip:

        # Get integers describing rate (assumes only 1 precip code)
        rate_old = get_rate(old_precip, number=True)
        rate_new = get_rate(new_precip, number=True)

        # Determine if significant based on rates
        if rate_old != rate_new:
            change = True

        # Exception is change to freezing precip, which is always
        # significant even if rate is the same
        elif ('FZ' in new_precip) ^ ('FZ' in old_precip):
            change = True

        # Otherwise, change is not significant
        else:
            change = False

    # Change from non precip to precip code is always significant (Light
    # precip not always significant but, as it is always associated
    # with vis change for first guess TAFs, it is)
    elif new_precip and not old_precip:
        change = True

    # If significant change not found, change is no significant
    else:
        change = False

    # Determine suitable precip codes for new and implied wx
    new_wx, implied_wx = assign_wx(change, new_precip, old_precip)

    return new_wx, implied_wx


def get_rate(sig_wx, number=False):
    """
    Determines precip rate from precip wx code.

    Args:
        sig_wx (str): Precip weather code
        number (bool): Indicator for whether to return as integer
                       (defaults to string)
    Return:
        rate (str): Precip rate
    """
    # Determine rates in string and integer form
    if '-' in sig_wx:
        rate_str_int = {False: 'light', True: 1}
    elif '+' in sig_wx:
        rate_str_int = {False: 'heavy', True: 3}
    else:
        rate_str_int = {False: 'moderate', True: 2}

    # Choose type of rate (string or integer) based on number argument
    rate = rate_str_int[number]

    return rate


def get_vis(vis_cat, rules):
    """
    Returns suitable visibility value based on visibility category.

    Args:
        vis_cat (float): Visibility category
        rules (str): TAF rules for airport (defence, civil or offshore)
    Return:
        vis (int): Visibility value
    """
    # Define dictionary for all TAF rules
    all_vis_dicts = {
        'defence': {7.0: 9999, 6.0: 7000, 5.0: 4000, 4.0: 3000, 3.0: 2000, 
                    2.0: 1200, 1.0: 500},
        'offshore': {8.0: 9999, 7.0: 8000, 6.0: 6000, 5.0: 4000, 4.0: 2000, 
                     3.0: 1200, 2.0: 500, 1.0: 300},
        'civil': {6.0: 9999, 5.0: 7000, 4.0: 3000, 3.0: 1200, 2.0: 500, 
                  1.0: 300}
    }

    # Get visibility dictionary for rules
    vis_dict = all_vis_dicts.get(rules, {})

    # Get visibility value, ensuring it is within range of dictionary
    if vis_cat > max(vis_dict):
        vis = vis_dict.get(max(vis_dict))
    elif vis_cat < min(vis_dict):
        vis = vis_dict.get(min(vis_dict))
    else:
        vis = vis_dict.get(vis_cat)

    return vis


def get_vis_cat(vis, sig_wx, rules):
    """
    Determines visibility TAF category based on visibility value.

    Args:
        vis (int): Visibility value
        sig_wx (str): Significant weather code
        rules (str): TAF rules for airport (civil, offshore or defence)
    Return:
        vis_cat (float): Visibility category
    """
    # Determine category in which fog may or may not be present
    if rules == 'defence':
        fg_cat = 2.
    else:
        fg_cat = 3.

    # Add 0.5 to this category if fog not present to distinguish between
    # fog and no fog in same category
    if 'FG' in sig_wx:
        fg_cat += 0.5

    # Define thresholds, categories and cloud to consider
    # For defence TAFs
    if rules == 'defence':
        thresholds = [8000, 5000, 3700, 2500, 1600, 800, 0]
        categories = [7., 6., 5., 4., 3., fg_cat, 1.]
    elif rules == 'offshore':
        thresholds = [9999, 7000, 5000, 3000, 1500, 800, 350, 0]
        categories = [8., 7., 6., 5., 4, fg_cat, 2., 1.]
    else:
        thresholds = [9999, 5000, 1500, 800, 350, 0]
        categories = [6., 5., 4., fg_cat, 2., 1.]

    # Determine category based on thresholds and categories
    for threshold, category in zip(thresholds, categories):
        if vis >= threshold:
            vis_cat = category
            break

    return vis_cat


def mp_queue(target_func, args, queue):
    """
    Wrapper function for allowing multiprocessing of a function and
    ensuring that the output is appended to a queue, to be picked up
    later.

    Args:
        target_func (function): Function used in multiprocessing
        args (list): List of arguments for function
        queue (multiprocessing.Queue): Multiprocessing queue object
    Returns:
        None
    """
    try:
        result = target_func(*args)
        queue.put(result)
    except Exception as e:
        # Put the sentinel value in the queue
        queue.put(co.ERROR_SENTINEL)
        print(f'Error in process: {e}', file=sys.stderr)


def round_cld(cld):
    """
    Rounds cloud base appropriately - rounds down to avoid rounding into
    category above.

    Args:
        cld (int or float): Raw cloud base value
    Returns:
        rounded_cld (int): Rounded cloud value
    """
    # Force to 5000ft for high cloud
    if cld >= 5000:
        rounded_cld = 5000

    # For cloud between 1500 and 5000ft, round down to nearest 500ft
    elif cld >= 1500:
        rounded_cld = int(math.floor(cld / 500.0) * 500.0)

    # For anything lower, round down to nearest 100ft
    else:
        rounded_cld = int(math.floor(cld / 100.0) * 100)

    return rounded_cld


def round_vis(vis, rules):
    """
    Rounds visibility appropriately. ICAO rules allow vis to be forecast
    in steps of 50m up to 800m and 100m steps from 800m to 5km. However,
    visibilities are rounded to nearest 500m above 1500 as these are
    commonly used values in TAFs.

    Args:
        vis (float or int): Raw visibility value
        rules (str): TAF rules (defence, civil or offshore)
    Return:
        rounded_vis (int): Rounded visibility value
    """
    # Force 9999 if visibility in top category
    if vis >= 9999 or vis >= 8000  and rules == 'defence':
        rounded_vis = 9999

    # If between 5000m and top category, round down to nearest 1000m
    elif vis >= 5000:
        rounded_vis = int(math.floor(vis / 1000.0) * 1000.0)

    # If between 1500m and 5000m, round down to nearest 500m
    elif vis >= 1500:
        rounded_vis = int(math.floor(vis / 500.0) * 500.0)

    # Make 400m if between 350m and 400m (avoids category complications)
    elif 400 > vis >= 350:
        rounded_vis = 400

    # If below 350m, round down to nearest 100m (with minimium of 100m)
    else:
        rounded_vis = max(100, int(math.floor(vis / 100.0) * 100.0))

    return rounded_vis


def use_cavok(vis, clds, sig_wx, wx_changes, prev_wx=None):
    """
    Determines whether CAVOK/NSC/NSW is needed.

    Args:
        vis (int): Visibility value
        clds (dict): Cloud values
        sig_wx (str): Sig wx code
        wx_changes (list): Changes being made in base/BECMG
        prev_wx (str): Sig wx in previous base conditions (if any)
    Return:
        cavok_messages (list): Required codes to use - CAVOK, NSC or NSW
    """
    # For non-vis/cld changes, don't need to worry about CAVOK
    if not any(wx in wx_changes for wx in ['vis', 'cld']):
        return []

    # Conditions for NSC
    nsc = all(cld == 5000 for cld in clds.values())

    # CAVOK conditions
    if all([vis == 9999, nsc, sig_wx in ['', 'NSW']]):
        return ['CAVOK']

    # NSC/NSW conditions (can be both)
    cavok_messages = []
    if nsc and 'cld' in wx_changes:
        cavok_messages.append('NSC')
    if all([sig_wx in ['', 'NSW'], prev_wx not in ['', 'NSW'],
            prev_wx is not None]):
        cavok_messages.append('NSW')

    return cavok_messages
