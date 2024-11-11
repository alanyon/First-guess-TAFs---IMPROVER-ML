"""
Constants used throughout TAF generation code.
"""
# TEST_DIR = '/data/users/alanyon/tafs/improver/test_data_2' # For testing
# DATA_DIR = '/scratch/alanyon/tafs/improver_test' # For testing
TEST_DIR = '/data/users/alanyon/tafs/improver/test_data'
DATA_DIR = '/scratch/alanyon/tafs/improver'
MASS_DIR = 'moose:/adhoc/users/ppdev/OS45.2'
# Define a sentinel value for errors in multiprocessing
ERROR_SENTINEL = "ERROR_OCCURRED"
IMPROVER_PARAMETERS = {
    'height_AGL_at_cloud_base_where_cloud_cover_2p5_oktas':
        {'data_type': 'percentiles',
         'short_name': 'cld_3',
         'units': 'ft',
         'fname_start': 'spotperc_extract_',
         'fname_start_alt': 'spotperc_extract_'},
    'height_AGL_at_cloud_base_where_cloud_cover_4p5_oktas':
        {'data_type': 'percentiles',
         'short_name': 'cld_5',
         'units': 'ft',
         'fname_start': 'spotperc_extract_',
         'fname_start_alt': 'spotperc_extract_'},
    'precip_rate_in_vicinity':
        {'data_type': 'percentiles',
         'short_name': 'precip_rate',
         'units': 'mm hr-1',
         'fname_start': 'spotperc_extract_',
         'fname_start_alt': 'spotperc_extract_'},
    'visibility_at_screen_level_in_vicinity':
        {'data_type': 'percentiles',
         'short_name': 'vis',
         'units': 'm',
         'fname_start': 'spotperc_extract_',
         'fname_start_alt': 'spotperc_extract_'},
    'weather_symbols-PT01H':
        {'data_type': 'deterministic',
         'short_name': 'sig_wx',
         'units': '1',
         'fname_start': 'weather_symbols_spot_',
         'fname_start_alt': 'latestspotperc_'},
    'wind_gust_at_10m_max-PT01H':
        {'data_type': 'percentiles',
         'short_name': 'wind_gust',
         'units': 'knots',
         'fname_start': 'spotperc_extract_',
         'fname_start_alt': 'spotperc_extract_'},
    'wind_direction_at_10m':
        {'data_type': 'deterministic',
         'short_name': 'wind_dir',
         'units': 'degrees',
         'fname_start': 'spotperc_extract_',
         'fname_start_alt': 'spotperc_extract_'},
    'wind_speed_at_10m':
        {'data_type': 'percentiles',
         'short_name': 'wind_mean',
         'units': 'knots',
         'fname_start': 'spotperc_extract_',
         'fname_start_alt': 'spotperc_extract_'},
    'temperature_at_screen_level':
        {'data_type': 'percentiles',
         'short_name': 'temp',
         'units': 'celsius',
         'fname_start': 'spotperc_extract_',
         'fname_start_alt': 'spotperc_extract_'},
    'lightning_flash_accumulation_in_vicinity-PT01H':
        {'data_type': 'probabilities',
         'short_name': 'lightning',
         'units': None,
         'fname_start': 'spot_extract_',
         'fname_start_alt': 'spot_extract_'}
}
PERCENTILES = [30, 40, 50, 60, 70]
PARAM_NAMES = {'wind': ['wind_dir', 'wind_mean', 'wind_gust'],
               'vis': ['vis', 'vis_cat', 'sig_wx'],
               'cld': ['cld_3', 'cld_5', 'cld_cat']}
WX_KEYS = {'wind': ['wind_dir', 'wind_mean', 'wind_gust'],
           'vis': ['vis', 'vis_cat', 'implied_sig_wx', 'sig_wx'],
           'cld': ['clds', 'cld_cat']}

# Dictionary giving significant weather TAF codes from Best Data codes
SIG_WX_DICT = {30: 'TSRA', 29: 'TSRA', 28: 'TSRA', 27: 'SN', 26: 'SHSN',
               25: 'SHSN', 24: '-SN', 23: '-SHSN', 22: '-SHSN', 21: 'SHGS',
               20: 'SHGS', 19: 'SHGS', 18: 'RASN', 17: 'SHRASN', 16: 'SHRASN',
               15: 'RA', 14: 'SHRA', 13: 'SHRA', 12: '-RA', 11: 'DZ',
               10: '-SHRA', 9: '-SHRA', 8: '', 7: '', 6: 'FG', 5: 'BR',
               4: '', 3: '', 2: '', 1: '', 0: ''}
NON_PRECIP_CODES = ['FZFG', 'FG', 'BR', 'HZ', '', 'NSW']
PRECIP_CODES = ['+SHSN', '+SN', '+SHRASN', '+RASN', '+TSRA', '+SHGS', '+SHRA',
                '+RA', '+DZ', 'SHSN', 'SN', 'SHRASN', 'RASN', 'TSRA', 'SHGS',
                'SHRA', 'RA', 'DZ', '-SHSN', '-SN', '-SHRASN', '-RASN',
                '-TSRA', '-SHGS', '-SHRA', '-RA', '-DZ']
TS_CODES = ['+TSRA', 'TSRA', '-TSRA']
HVY_CODES = ['+SHSN', '+SN', '+SHRASN', '+RASN', '+SHGS', '+SHRA', '+RA']
AIRPORT_INFO_FILE = ('/home/h04/alanyon/cylc-run/improver_tafs/bin/'
                     'first_guess_TAFs/improver/data_extraction/taf_info.csv')

# Ordering priority of change groups
PRIORITY_DICT = {'base': 0, 'BECMG': 1, 'TEMPO': 2, 'PROB40': 3,
                 'PROB40 TEMPO': 4, 'PROB30': 5, 'PROB30 TEMPO': 6}

PROB_DICT = {'TEMPO': 100, 'PROB40': 40, 'PROB40 TEMPO': 40, 'PROB30': 30,
             'PROB30 TEMPO': 30}
BUST_COLS = ['wind_bust_label', 'vis_bust_label', 'cld_bust_label']

# ML constants
PARAM_COLS = [
    'precip_rate_30.0', 'precip_rate_50.0', 'precip_rate_70.0', 
    'wind_dir_30.0', 'wind_dir_50.0', 'wind_dir_70.0', 'temp_30.0', 
    'temp_50.0', 'temp_70.0', 'wind_mean_30.0', 'wind_mean_50.0', 
    'wind_mean_70.0', 'wind_gust_30.0', 'wind_gust_50.0', 'wind_gust_70.0', 
    'month', 'day', 'hour', 'lead', 'vis_cat_30.0', 'vis_cat_50.0', 
    'vis_cat_70.0', 'cld_cat_30.0', 'cld_cat_50.0', 'cld_cat_70.0'
]
# PARAM_COLS = [
#     'precip_rate_70.0', 'wind_dir_30.0', 'wind_dir_50.0', 'wind_dir_70.0', 
#     'temp_30.0', 'temp_50.0', 'temp_70.0', 'wind_mean_30.0', 'wind_mean_50.0', 
#     'wind_mean_70.0', 'wind_gust_30.0', 'wind_gust_50.0', 'wind_gust_70.0', 
#     'month', 'day', 'hour', 'lead', 'vis_cat_30.0', 'vis_cat_50.0', 
#     'vis_cat_70.0', 'cld_cat_30.0', 'cld_cat_50.0', 'cld_cat_70.0'
# ]
ML_ICAOS = [
    'EGAA', 'EGAC', 'EGAE', 'EGBB', 'EGBJ', 'EGCC', 'EGCK', 'EGEC', 'EGEO', 
    'EGFF', 'EGGD', 'EGGP', 'EGGW', 'EGHC', 'EGHE', 'EGHH', 'EGHI', 'EGHQ', 
    'EGKA', 'EGKB', 'EGKK', 'EGLC', 'EGLF', 'EGLL', 'EGMC', 'EGMD', 'EGNH', 
    'EGNJ', 'EGNM', 'EGNO', 'EGNR', 'EGNT', 'EGNV', 'EGNX', 'EGPA', 'EGPB', 
    'EGPC', 'EGPD', 'EGPE', 'EGPF', 'EGPH', 'EGPI', 'EGPK', 'EGPL', 'EGPN', 
    'EGPO', 'EGPU', 'EGSC', 'EGSH', 'EGSS', 'EGSY', 'EGTC', 'EGTE', 'EGTK',
]
NICE_LABELS = {
    'vis_decrease': 'Obs Vis Lower', 'vis_increase': 'Obs Vis Higher',
    'cld_decrease': 'Obs Cld Lower', 'cld_increase': 'Obs Cld Higher',
    'wind_decrease': 'Obs Wind Lower', 'wind_increase': 'Obs Wind Higher',
    'wind_dir': 'Wind Dir Bust', 'no_bust': 'No Bust', 'bust': 'Bust'
}
DATE_ICAOS = {
    '20230805': 'EGAA', '20230806': 'EGAC', '20230807': 'EGAE', 
    '20230808': 'EGBB', '20230809': 'EGBJ', '20230810': 'EGCC',
    '20230811': 'EGCK', '20230812': 'EGEC', '20230813': 'EGEO',
    '20230814': 'EGFF', '20230815': 'EGGD', '20230816': 'EGGP',
    '20230817': 'EGGW', '20230818': 'EGHC', '20230819': 'EGHE',
    '20230820': 'EGHH', '20230821': 'EGHI', '20230822': 'EGHQ',
    '20230823': 'EGKA', '20230824': 'EGKB', '20230825': 'EGKK',
    '20230826': 'EGLC', '20230827': 'EGLF', '20230828': 'EGLL',
    '20230829': 'EGMC', '20230830': 'EGMD', '20230831': 'EGNH',
    '20230901': 'EGNJ', '20230902': 'EGNM', '20230903': 'EGNO',
    '20230904': 'EGNR', '20230905': 'EGNT', '20230906': 'EGNV',
    '20230907': 'EGNX', '20230908': 'EGPA', '20230909': 'EGPB',
    '20230910': 'EGPC', '20230911': 'EGPD', '20230912': 'EGPE',
    '20230913': 'EGPF', '20230914': 'EGPH', '20230915': 'EGPI',
    '20230916': 'EGPK', '20230917': 'EGPL', '20230918': 'EGPN',
    '20230919': 'EGPO', '20230920': 'EGPU', '20230921': 'EGSC',
    '20230922': 'EGSH', '20230923': 'EGSS', '20230924': 'EGSY',
    '20230925': 'EGTC', '20230926': 'EGTE', '20230927': 'EGTK'
}
