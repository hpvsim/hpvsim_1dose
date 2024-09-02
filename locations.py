"""
Locations
"""

# All locations
locations = [
    'bangladesh',       # 0
    'burkina faso',     # 1
    'cameroon',         # 2
    'cote divoire',     # 3
    'ethiopia',         # 4
    'mali',             # 6
    'mozambique',       # 7
    'myanmar',          # 8
    'nigeria',          # 9
    'sierra leone',     # 10
    'tanzania',         # 11
    'togo',             # 12
    'zambia',           # 14
]

nosbdata_locations = ["cote d'ivoire", "cote divoire"]

cancer_type_locs = ['ethiopia', 'mozambique', 'nigeria', 'tanzania', 'uganda']

vx_coverage_2023_routine = {
    # Country
    'bangladesh':   1.00,
    'burkina faso': 1.00,
    'cameroon':     0.56,
    'cote divoire': 0.75,
    'ethiopia':     0.30,
    'mali':         0.00,
    'mozambique':   0.22,
    'myanmar':      1.00,
    'nigeria':      1.00,
    'sierra leone': 0.38,
    'tanzania':     0.35,
    'togo':         1.00,
    'zambia':       1.00
}

vx_coverage_2023_macs = {
    # Country       11-14, 15-16
    'bangladesh':   [0.31, 0.00],
    'burkina faso': [0.30, 0.00],
    'cameroon':     [0.00, 0.00],
    'cote divoire': [0.00, 0.00],
    'ethiopia':     [0.00, 0.00],
    'mali':         [0.00, 0.00],
    'mozambique':   [0.00, 0.00],
    'myanmar':      [0.07, 0.00],
    'nigeria':      [0.28, 0.00],
    'sierra leone': [0.00, 0.00],
    'tanzania':     [0.00, 0.00],
    'togo':         [1.00, 0.36],
    'zambia':       [1.00, 0.28],
}

vx_coverage_2023_cf_routine = {
    # Country
    'bangladesh':   1.00,
    'burkina faso': 1.00,
    'cameroon':     0.28,
    'cote divoire': 0.37,
    'ethiopia':     0.30,
    'mali':         0.00,
    'mozambique':   0.22,
    'myanmar':      0.68,
    'nigeria':      1.00,
    'sierra leone': 0.38,
    'tanzania':     0.35,
    'togo':         1.00,
    'zambia':       1.00
}

vx_coverage_2023_cf_macs = {
    # Country       11-14, 15-16
    'bangladesh':   [0.03, 0.00],
    'burkina faso': [0.01, 0.00],
    'cameroon':     [0.00, 0.00],
    'cote divoire': [0.00, 0.00],
    'ethiopia':     [0.00, 0.00],
    'mali':         [0.00, 0.00],
    'mozambique':   [0.00, 0.00],
    'myanmar':      [0.00, 0.00],
    'nigeria':      [0.01, 0.00],
    'sierra leone': [0.00, 0.00],
    'tanzania':     [0.00, 0.00],
    'togo':         [0.46, 0.00],
    'zambia':       [0.44, 0.00],
}

######################
vx_coverage_2024_routine = {
    # Country
    'bangladesh':   1.00,
    'burkina faso': 0.83,
    'cameroon':     0.48,
    'cote divoire': 0.82,
    'ethiopia':     1.00,
    'mali':         0.78,
    'mozambique':   1.00,
    'myanmar':      1.00,
    'nigeria':      1.00,
    'sierra leone': 0.69,
    'tanzania':     1.00,
    'togo':         1.00,
    'zambia':       1.00
}

vx_coverage_2024_macs = {
    # Country       11-14, 15-16
    'bangladesh':   [0.68, 0.00],
    'burkina faso': [0.00, 0.00],
    'cameroon':     [0.00, 0.00],
    'cote divoire': [0.00, 0.00],
    'ethiopia':     [1.00, 0.91],
    'mali':         [0.00, 0.00],
    'mozambique':   [0.38, 0.00],
    'myanmar':      [0.06, 0.00],
    'nigeria':      [0.51, 0.00],
    'sierra leone': [0.00, 0.00],
    'tanzania':     [0.93, 0.00],
    'togo':         [0.02, 0.00],
    'zambia':       [0.07, 0.00],
}

vx_coverage_2024_cf_routine = {
    # Country
    'bangladesh':   1.00,
    'burkina faso': 0.41,
    'cameroon':     0.24,
    'cote divoire': 0.41,
    'ethiopia':     1.00,
    'mali':         0.39,
    'mozambique':   1.00,
    'myanmar':      0.63,
    'nigeria':      1.00,
    'sierra leone': 0.35,
    'tanzania':     1.00,
    'togo':         0.54,
    'zambia':       0.64
}

vx_coverage_2024_cf_macs = {
    # Country       11-14, 15-16
    'bangladesh':   [0.21, 0.00],
    'burkina faso': [0.00, 0.00],
    'cameroon':     [0.00, 0.00],
    'cote divoire': [0.00, 0.00],
    'ethiopia':     [0.59, 0.00],
    'mali':         [0.00, 0.00],
    'mozambique':   [0.06, 0.00],
    'myanmar':      [0.00, 0.00],
    'nigeria':      [0.13, 0.00],
    'sierra leone': [0.00, 0.00],
    'tanzania':     [0.32, 0.00],
    'togo':         [0.00, 0.00],
    'zambia':       [0.00, 0.00],
}

######################

vx_coverage_both_macs = {
    # Country       11-14, 15-16
    'bangladesh':   [0.99, 0.00],
    'burkina faso': [0.30, 0.00],
    'cameroon':     [0.00, 0.00],
    'cote divoire': [0.00, 0.00],
    'ethiopia':     [1.00, 0.91],
    'mali':         [0.00, 0.00],
    'mozambique':   [0.38, 0.00],
    'myanmar':      [0.13, 0.00],
    'nigeria':      [0.79, 0.00],
    'sierra leone': [0.00, 0.00],
    'tanzania':     [0.93, 0.00],
    'togo':         [1.00, 0.40],
    'zambia':       [1.00, 0.44],
}

vx_coverage_both_cf_macs = {
    # Country       11-14, 15-16
    'bangladesh':   [0.24, 0.00],
    'burkina faso': [0.01, 0.00],
    'cameroon':     [0.00, 0.00],
    'cote divoire': [0.00, 0.00],
    'ethiopia':     [0.59, 0.00],
    'mali':         [0.00, 0.00],
    'mozambique':   [0.06, 0.00],
    'myanmar':      [0.00, 0.00],
    'nigeria':      [0.13, 0.00],
    'sierra leone': [0.00, 0.00],
    'tanzania':     [0.32, 0.00],
    'togo':         [0.46, 0.00],
    'zambia':       [0.44, 0.00],
}
