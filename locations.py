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

vx_coverage_actual = {
    # Country
    'bangladesh':   0.93,
    'burkina faso': 0.55,
    'cambodia':     0.44,
    'cameroon':     0.19,
    'cote divoire': 0.27,
    'ethiopia':     0.90,
    'gambia':       0.56,
    'laos':         0.98,
    'mali':         0.14,
    'mozambique':   0.47,
    'myanmar':      0.47,
    'nepal':        0.99,
    'nigeria':      0.87,
    'sierra leone': 0.20,
    'tanzania':     0.90,
    'togo':         0.94,
    'zambia':       0.89
}

######################
vx_coverage_cf = {
    # Country
    'bangladesh':   0.47,
    'burkina faso': 0.27,
    'cambodia':     0.22,
    'cameroon':     0.09,
    'cote divoire': 0.14,
    'ethiopia':     0.47,
    'gambia':       0.28,
    'laos':         0.52,
    'mali':         0.07,
    'mozambique':   0.25,
    'myanmar':      0.24,
    'nepal':        0.50,
    'nigeria':      0.43,
    'sierra leone': 0.13,
    'tanzania':     0.48,
    'togo':         0.47,
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