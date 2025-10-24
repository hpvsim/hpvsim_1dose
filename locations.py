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

location_labels = {
    'bangladesh':   'Bangladesh',
    'burkina faso': 'Burkina Faso',
    'cambodia':     'Cambodia',
    'cameroon':     'Cameroon',
    'cote divoire': 'Côte d\'Ivoire',
    'ethiopia':     'Ethiopia',
    'gambia':       'Gambia',
    'laos':         'Lao PDR',
    'mali':         'Mali',
    'mozambique':   'Mozambique',
    'myanmar':      'Myanmar',
    'nepal':        'Nepal',
    'nigeria':      'Nigeria',
    'sierra leone': 'Sierra Leone',
    'tanzania':     'Tanzania',
    'togo':         'Togo',
    'zambia':       'Zambia',
}

nosbdata_locations = ["cote d'ivoire", "cote divoire", "laos"]

cancer_type_locs = ['ethiopia', 'mozambique', 'nigeria', 'tanzania', 'uganda']

vx_coverage_shipped = {
    # Country      # % of girls in largest possible cohort - see "HPVsim results" col W
    'bangladesh':   0.93,   # 90
    'burkina faso': 0.55,   # 99 of 9yos
    'cambodia':     0.44,   # 85
    'cameroon':     0.19,   # 36
    'cote divoire': 0.27,   # 61
    'ethiopia':     0.90,   # 58
    'gambia':       0.56,   # 15
    'laos':         0.98,   # 95
    'mali':         0.14,   # 15
    'mozambique':   0.47,   # 89
    'myanmar':      0.47,   # 83
    'nepal':        0.99,   # NA
    'nigeria':      0.87,   # 60
    'sierra leone': 0.20,   # 61
    'tanzania':     0.56,   # 94
    'togo':         0.94,   # 36
    'zambia':       0.89    # 60
}

vx_coverage_actual = {
    # Country      # % of girls in largest possible cohort - see "HPVsim results" col AB
    'bangladesh':   0.86,
    'burkina faso': 0.37,
    'cambodia':     0.33,
    'cameroon':     0.20,
    'cote divoire': 0.38,
    'ethiopia':     0.54,
    'gambia':       0.53,
    'laos':         0.45,
    'mali':         0.03,
    'mozambique':   0.46,
    'myanmar':      0.55,
    'nepal':        0.00,
    'nigeria':      0.79,
    'sierra leone': 0.31,
    'tanzania':     0.95,
    'togo':         0.50,
    'zambia':       0.54
}

######################
vx_coverage_cf = {
    # Country
    'bangladesh':   0.34,
    'burkina faso': 0.21,
    'cambodia':     0.17,
    'cameroon':     0.07,
    'cote divoire': 0.10,
    'ethiopia':     0.47,
    'gambia':       0.22,
    'laos':         0.52,
    'mali':         0.05,
    'mozambique':   0.19,
    'myanmar':      0.18,
    'nepal':        0.37,
    'nigeria':      0.33,
    'sierra leone': 0.10,
    'tanzania':     0.30,
    'togo':         0.42,
    'zambia':       0.45
}

######################
vx_coverage_optim = {
    # Country
    'bangladesh':   0.34,
    'burkina faso': 0.21,
    'cambodia':     0.17,
    'cameroon':     0.07,
    'cote divoire': 0.10,
    'ethiopia':     0.47,
    'gambia':       0.22,
    'laos':         0.52,
    'mali':         0.05,
    'mozambique':   0.19,
    'myanmar':      0.18,
    'nepal':        0.37,
    'nigeria':      0.33,
    'sierra leone': 0.10,
    'tanzania':     0.37,
    'togo':         0.42,
    'zambia':       0.45
}

######################
vx_coverage_denom = {
    # Country       # Number of girls aged 9-16 in 2024
    'bangladesh':   9435363,
    'burkina faso': 1809440,
    'cambodia':     986465,
    'cameroon':     2091554,
    'cote divoire': 2248973,
    'ethiopia':     11850432,
    'gambia':       205466,
    'laos':         599529,
    'mali':         1904512,
    'mozambique':   2641889,
    'myanmar':      2566301,
    'nepal':        1693828,
    'nigeria':      17312580,
    'sierra leone': 610929,
    'tanzania':     4876388,
    'togo':         776290,
    'zambia':       2094613
}

vx_coverage_extra = {
    # Country       # Number of girls aged 9-16 in 2024
    'bangladesh':   4408290,
    'burkina faso': 496574.5,
    'cambodia':     218430,
    'cameroon':     196008.75,
    'cote divoire': 304361,
    'ethiopia':     5104041.25,
    'gambia':       57674.5,
    'laos':         272697.5,
    'mali':         133636.5,
    'mozambique':   569026.25,
    'myanmar':      607905,
    'nepal':        840759.5,
    'nigeria':      7522955,
    'sierra leone': 39448.75,
    'tanzania':     2025827.5,
    'togo':         366480,
    'zambia':       934548.25
}

vx_coverage_extra_actual = {
    # Country       # Number of girls aged 9-16 in 2024
    'bangladesh':   4408290,
    'burkina faso': 496574.5,
    'cambodia':     218430,
    'cameroon':     196008.75,
    'cote divoire': 304361,
    'ethiopia':     5104041.25,
    'gambia':       57674.5,
    'laos':         272697.5,
    'mali':         133636.5,
    'mozambique':   569026.25,
    'myanmar':      607905,
    'nepal':        840759.5,
    'nigeria':      7522955,
    'sierra leone': 39448.75,
    'tanzania':     2025827.5,
    'togo':         366480,
    'zambia':       934548.25
}