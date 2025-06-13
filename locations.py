"""
Locations
"""

# All locations
locations = [
    'bangladesh',       # 0
    'burkina faso',     # 1
    'cambodia',         # 2
    'cameroon',         # 3
    'cote divoire',     # 4
    'ethiopia',         # 5
    'gambia',           # 6
    'laos',             # 7
    'mali',             # 8
    'mozambique',       # 9
    'myanmar',          # 10
    'nepal',            # 11
    'nigeria',          # 12
    'sierra leone',     # 13
    'tanzania',         # 14
    'togo',             # 15
    'zambia',           # 16
]

nosbdata_locations = ["cote d'ivoire", "cote divoire", "laos"]

cancer_type_locs = ['ethiopia', 'mozambique', 'nigeria', 'tanzania', 'uganda']

vx_coverage_2023_routine = {
    # Country
    'bangladesh':   1.00,
    'burkina faso': 1.00,
    'cambodia':     1.00,
    'cameroon':     0.57,
    'cote divoire': 0.73,
    'ethiopia':     0.32,
    'gambia':       0.02,
    'laos':         0.55,
    'mali':         0.00,
    'mozambique':   0.21,
    'myanmar':      1.00,
    'nepal':        0.00,
    'nigeria':      1.00,
    'sierra leone': 0.41,
    'tanzania':     0.37,
    'togo':         1.00,
    'zambia':       1.00
}

vx_coverage_2023_macs = {
    # Country       11-14, 15-16
    'bangladesh':   [0.28, 0.00],
    'burkina faso': [0.33, 0.00],
    'cambodia':     [0.14, 0.00],
    'cameroon':     [0.00, 0.00],
    'cote divoire': [0.00, 0.00],
    'ethiopia':     [0.00, 0.00],
    'gambia':       [0.00, 0.00],
    'laos':         [0.00, 0.00],
    'mali':         [0.00, 0.00],
    'mozambique':   [0.00, 0.00],
    'myanmar':      [0.08, 0.00],
    'nepal':        [0.00, 0.00],
    'nigeria':      [0.29, 0.00],
    'sierra leone': [0.00, 0.00],
    'tanzania':     [0.00, 0.00],
    'togo':         [1.00, 0.28],
    'zambia':       [1.00, 0.36],
}

vx_coverage_2023_cf_routine = {
    # Country
    'bangladesh':   1.00,
    'burkina faso': 1.00,
    'cambodia':     0.77,
    'cameroon':     0.29,
    'cote divoire': 0.36,
    'ethiopia':     0.32,
    'gambia':       0.02,
    'laos':         0.55,
    'mali':         0.00,
    'mozambique':   0.21,
    'myanmar':      0.67,
    'nepal':        0.00,
    'nigeria':      1.00,
    'sierra leone': 0.41,
    'tanzania':     0.37,
    'togo':         1.00,
    'zambia':       1.00
}

vx_coverage_2023_cf_macs = {
    # Country       11-14, 15-16
    'bangladesh':   [0.02, 0.00],
    'burkina faso': [0.03, 0.00],
    'cambodia':     [0.00, 0.00],
    'cameroon':     [0.00, 0.00],
    'cote divoire': [0.00, 0.00],
    'ethiopia':     [0.00, 0.00],
    'gambia':       [0.00, 0.00],
    'laos':         [0.00, 0.00],
    'mali':         [0.00, 0.00],
    'mozambique':   [0.00, 0.00],
    'myanmar':      [0.00, 0.00],
    'nepal':        [0.00, 0.00],
    'nigeria':      [0.01, 0.00],
    'sierra leone': [0.00, 0.00],
    'tanzania':     [0.00, 0.00],
    'togo':         [0.43, 0.00],
    'zambia':       [0.45, 0.00],
}

######################
vx_coverage_2024_routine = {
    # Country
    'bangladesh':   1.00,
    'burkina faso': 0.90,
    'cambodia':     1.00,
    'cameroon':     0.51,
    'cote divoire': 0.84,
    'ethiopia':     1.00,
    'gambia':       1.00,
    'laos':         1.00,
    'mali':         0.80,
    'mozambique':   1.00,
    'myanmar':      1.00,
    'nepal':        1.00,
    'nigeria':      1.00,
    'sierra leone': 0.76,
    'tanzania':     1.00,
    'togo':         1.00,
    'zambia':       1.00
}

vx_coverage_2024_macs = {
    # Country       11-14, 15-16
    'bangladesh':   [0.63, 0.00],
    'burkina faso': [0.00, 0.00],
    'cambodia':     [0.02, 0.00],
    'cameroon':     [0.00, 0.00],
    'cote divoire': [0.00, 0.00],
    'ethiopia':     [1.00, 0.95],
    'gambia':       [0.60, 0.00],
    'laos':         [1.00, 1.00],
    'mali':         [0.00, 0.00],
    'mozambique':   [0.40, 0.00],
    'myanmar':      [0.13, 0.00],
    'nepal':        [1.00, 0.44],
    'nigeria':      [0.51, 0.00],
    'sierra leone': [0.00, 0.00],
    'tanzania':     [1.00, 0.03],
    'togo':         [0.01, 0.00],
    'zambia':       [0.08, 0.00],
}

vx_coverage_2024_cf_routine = {
    # Country
    'bangladesh':   1.00,
    'burkina faso': 0.45,
    'cambodia':     0.55,
    'cameroon':     0.26,
    'cote divoire': 0.42,
    'ethiopia':     1.00,
    'gambia':       1.00,
    'laos':         1.00,
    'mali':         0.40,
    'mozambique':   1.00,
    'myanmar':      0.76,
    'nepal':        1.00,
    'nigeria':      1.00,
    'sierra leone': 0.38,
    'tanzania':     1.00,
    'togo':         0.52,
    'zambia':       0.66
}

vx_coverage_2024_cf_macs = {
    # Country       11-14, 15-16
    'bangladesh':   [0.19, 0.00],
    'burkina faso': [0.00, 0.00],
    'cambodia':     [0.00, 0.00],
    'cameroon':     [0.00, 0.00],
    'cote divoire': [0.00, 0.00],
    'ethiopia':     [0.61, 0.00],
    'gambia':       [0.16, 0.00],
    'laos':         [0.67, 0.00],
    'mali':         [0.00, 0.00],
    'mozambique':   [0.06, 0.00],
    'myanmar':      [0.00, 0.00],
    'nepal':        [0.49, 0.00],
    'nigeria':      [0.12, 0.00],
    'sierra leone': [0.00, 0.00],
    'tanzania':     [0.38, 0.00],
    'togo':         [0.00, 0.00],
    'zambia':       [0.00, 0.00],
}

######################

vx_coverage_both_macs = {
    # Country       11-14, 15-16
    'bangladesh':   [0.90, 0.00],
    'burkina faso': [0.33, 0.00],
    'cambodia':     [0.16, 0.00],
    'cameroon':     [0.00, 0.00],
    'cote divoire': [0.00, 0.00],
    'ethiopia':     [1.00, 0.91],
    'gambia':       [0.60, 0.00],
    'laos':         [1.00, 1.00],
    'mali':         [0.00, 0.00],
    'mozambique':   [0.40, 0.00],
    'myanmar':      [0.21, 0.00],
    'nepal':        [1.00, 0.44],
    'nigeria':      [0.80, 0.00],
    'sierra leone': [0.00, 0.00],
    'tanzania':     [1.00, 0.03],
    'togo':         [1.00, 0.29],
    'zambia':       [1.00, 0.54],
}

vx_coverage_both_cf_macs = {
    # Country       11-14, 15-16
    'bangladesh':   [0.21, 0.00],
    'burkina faso': [0.03, 0.00],
    'cambodia':     [0.00, 0.00],
    'cameroon':     [0.00, 0.00],
    'cote divoire': [0.00, 0.00],
    'ethiopia':     [0.61, 0.00],
    'gambia':       [0.16, 0.00],
    'laos':         [0.67, 0.00],
    'mali':         [0.00, 0.00],
    'mozambique':   [0.06, 0.00],
    'myanmar':      [0.00, 0.00],
    'nepal':        [0.49, 0.00],
    'nigeria':      [0.13, 0.00],
    'sierra leone': [0.00, 0.00],
    'tanzania':     [0.38, 0.00],
    'togo':         [0.43, 0.00],
    'zambia':       [0.45, 0.00],
}
