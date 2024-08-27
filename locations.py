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
    'malawi',           # 5
    'mali',             # 6
    'mozambique',       # 7
    'myanmar',          # 8
    'nigeria',          # 9
    'sierra leone',     # 10
    'tanzania',         # 11
    'togo',             # 12
    'uganda',           # 13
    'zambia',           # 14
]

nosbdata_locations = ["cote d'ivoire", "cote divoire"]

cancer_type_locs = ['ethiopia', 'mozambique', 'nigeria', 'tanzania', 'uganda']

vx_intro = {
    'bangladesh': 2023,
    'burkina faso': 2022,
    'cameroon': 2020,
    'cote divoire': 2019,
    'ethiopia': 2018,
    'malawi': 2019,
    'mozambique': 2021,
    'myanmar': 2020,
    'nigeria': 2023,
    'sierra leone': 2022,
    'tanzania': 2018,
    'togo': 2023,
    'uganda': 2015,
    'zambia': 2019,
}

vx_doses = {
    'bangladesh': [3639200, 6157000],
    'burkina faso': [747910, 297510],
    'cameroon': [219120, 193530],
    'cote divoire': [304800, 335960],
    'ethiopia': [1012100,  10745350],  # [2 doses, 1 dose]
    # 'malawi': [2019],
    'mozambique': [210140,  1197950],  # [2 doses, 1 dose]
    'myanmar': [600000, 679800],
    'nigeria': [6597400, 9240400],
    'sierra leone': [89310, 83050],  # [2 doses, 1 dose]
    'tanzania': [679750, 4264900],  # [2 doses, 1 dose]
    'togo': [682800, 131600],
    # 'uganda': 2015,
    'zambia': [1585940, 381530],
}