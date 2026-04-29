"""
One-off: dump the PUBLIC reference tables from `HPVsim results 20251030.xlsx`
into `data/` as CSVs. These are safe to commit.

Run from the repo root:
    python scripts/export_xlsx_to_data.py [--xlsx PATH]

Outputs:
    data/population_by_age.csv
        country, age_9, age_10, ..., age_19  (girls-only, 2024 projection)
    data/target_ages.csv
        country, iso3, target_lower, target_upper, schedule_2023, schedule_2024,
        wastage_factor, cohort_upper_exclusive
"""
import argparse
import csv
import os

import openpyxl

# The 17 analysis countries. Key = locations.py lowercase name.
# Each entry lists the spelling variants that appear across Excel sheets.
LOC_ALIASES = {
    'bangladesh':   ['Bangladesh'],
    'burkina faso': ['Burkina Faso'],
    'cambodia':     ['Cambodia'],
    'cameroon':     ['Cameroon'],
    'cote divoire': ["Cote d'Ivoire", "Côte d'Ivoire"],
    'ethiopia':     ['Ethiopia'],
    'gambia':       ['Gambia'],
    'laos':         ['Lao', 'Lao PDR', "Lao People's Democratic Republic"],
    'mali':         ['Mali'],
    'mozambique':   ['Mozambique'],
    'myanmar':      ['Myanmar'],
    'nigeria':      ['Nigeria'],
    'sierra leone': ['Sierra Leone'],
    'tanzania':     ['Tanzania'],
    'togo':         ['Togo'],
    'zambia':       ['Zambia'],
}
LOCATIONS_ORDER = list(LOC_ALIASES.keys())


def _resolve(row_name, aliases):
    for loc, alts in aliases.items():
        if row_name in alts:
            return loc
    return None


def _load_ws(xlsx, name):
    wb = openpyxl.load_workbook(xlsx, data_only=True, read_only=True)
    return wb[name]


def export_population_by_age(xlsx, outpath):
    """Coverage calcs sheet rows 3-20, cols C..M = ages 9..19."""
    ws = _load_ws(xlsx, 'Coverage calcs')
    ages = list(range(9, 20))
    rows_by_loc = {}
    for row_num in range(3, 21):
        name = ws.cell(row=row_num, column=2).value
        loc = _resolve(name, LOC_ALIASES) if name else None
        if loc is None:
            continue
        pops = [ws.cell(row=row_num, column=c).value for c in range(3, 14)]
        rows_by_loc[loc] = pops
    with open(outpath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['location'] + [f'age_{a}' for a in ages])
        for loc in LOCATIONS_ORDER:
            if loc in rows_by_loc:
                w.writerow([loc] + rows_by_loc[loc])
            else:
                print(f'WARNING: no population row for {loc!r} in Coverage calcs')


def export_target_ages(xlsx, outpath):
    """WUENIC sheet (rows 4-20) + HPVsim results col X for cohort upper bound."""
    wuenic = _load_ws(xlsx, 'WUENIC')
    hpv = _load_ws(xlsx, 'HPVsim results')

    cohort_upper_by_loc = {}
    for row_num in range(3, 20):
        name = hpv.cell(row=row_num, column=2).value
        upper = hpv.cell(row=row_num, column=24).value  # col X
        loc = _resolve(name, LOC_ALIASES) if name else None
        if loc and upper is not None:
            cohort_upper_by_loc[loc] = upper

    rows_by_loc = {}
    for row_num in range(4, 21):
        iso3 = wuenic.cell(row=row_num, column=1).value
        name = wuenic.cell(row=row_num, column=2).value
        loc = _resolve(name, LOC_ALIASES) if name else None
        if loc is None:
            continue
        rows_by_loc[loc] = (
            iso3,
            wuenic.cell(row=row_num, column=3).value,
            wuenic.cell(row=row_num, column=4).value,
            wuenic.cell(row=row_num, column=5).value,
            wuenic.cell(row=row_num, column=6).value,
            wuenic.cell(row=row_num, column=7).value,
            cohort_upper_by_loc.get(loc),
        )

    with open(outpath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['location', 'iso3', 'target_lower', 'target_upper',
                    'schedule_2023', 'schedule_2024',
                    'wastage_factor', 'cohort_upper_exclusive',
                    'cf_upper_exclusive'])
        for loc in LOCATIONS_ORDER:
            if loc in rows_by_loc:
                row = list(rows_by_loc[loc])
                cohort_upper = row[-1]
                # Counterfactual 2-dose scenario assumes catch-up reaches at
                # least age 16 (cohort 9..16, i.e. upper-exclusive=17). Countries
                # whose program already extends past 16 keep their wider cohort.
                cf_upper = max(cohort_upper, 17) if cohort_upper else 17
                w.writerow([loc] + row + [cf_upper])
            else:
                print(f'WARNING: no WUENIC row for {loc!r}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlsx', default='HPVsim results 20251030.xlsx')
    parser.add_argument('--outdir', default='data')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    export_population_by_age(args.xlsx, os.path.join(args.outdir, 'population_by_age.csv'))
    export_target_ages(args.xlsx, os.path.join(args.outdir, 'target_ages.csv'))
    print('Wrote public CSVs to', args.outdir)


if __name__ == '__main__':
    main()
