"""
Reproduce the three vaccine-coverage dicts that live in locations.py:

    vx_coverage_shipped  -- from Gavi shipments (2023 + 2024), wastage-adjusted,
                            divided by cohort (girls aged 9..cohort_upper-1).
    vx_coverage_actual   -- from Gavi actuals nvax (girls actually vaccinated),
                            reshaped as doses (× schedule), then / cohort.
    vx_coverage_cf       -- counterfactual: same shipments but under a 2-dose
                            schedule ⇒ half the doses reach girls.

Inputs:
    private/shipments_2023.csv   (country, doses, schedule)       -- gitignored
    private/shipments_2024.csv   (country, doses, schedule, ...)   -- gitignored
    private/gavi_actuals.csv     (iso3, year, nvax, ...)           -- gitignored
    data/population_by_age.csv   (location, age_9..age_19)          -- public
    data/target_ages.csv         (location, iso3, schedule, wastage_factor,
                                  cohort_upper_exclusive, ...)      -- public

Usage:
    python scripts/compute_coverage.py
        [--private-dir private] [--data-dir data]
        [--out locations_coverage.py]

Writes `locations_coverage.py` (Python module with the three dicts).
Run a diff against the current locations.py dicts to verify.
"""
import argparse
import os

import numpy as np
import pandas as pd


# Alias map used to reconcile country-name spellings back to locations.py keys.
# Shipments CSVs use Excel spellings; we normalize to lowercase locations keys.
SHIPMENT_ALIASES = {
    'Bangladesh': 'bangladesh',
    'Burkina Faso': 'burkina faso',
    'Cambodia': 'cambodia',
    'Cameroon': 'cameroon',
    "Cote d'Ivoire": 'cote divoire',
    "Côte d'Ivoire": 'cote divoire',
    'Ethiopia': 'ethiopia',
    'Gambia': 'gambia',
    'Lao': 'laos',
    'Lao PDR': 'laos',
    "Lao People's Democratic Republic": 'laos',
    'Mali': 'mali',
    'Mozambique': 'mozambique',
    'Myanmar': 'myanmar',
    'Nigeria': 'nigeria',
    'Sierra Leone': 'sierra leone',
    'Tanzania': 'tanzania',
    'Togo': 'togo',
    'Zambia': 'zambia',
}


def _normalize_shipments(df):
    df = df.copy()
    df['location'] = df['country'].map(SHIPMENT_ALIASES)
    df = df.dropna(subset=['location'])
    # Schedule 'NITAG recommneded' etc. → treat as default 2 (per Excel behavior
    # where non-numeric schedule falls through to the sheet's 2-dose bucket).
    df['schedule'] = pd.to_numeric(df['schedule'], errors='coerce')
    df = df.dropna(subset=['schedule'])
    df['schedule'] = df['schedule'].astype(int)
    return df


def _cohort_size(pop_row, upper_exclusive):
    """Sum girls aged 9..upper_exclusive-1 (matches HPVsim results col Y formula)."""
    if np.isnan(upper_exclusive):
        return np.nan
    ages = range(9, int(upper_exclusive))
    return sum(pop_row[f'age_{a}'] for a in ages)


def compute(private_dir, data_dir):
    ship23 = _normalize_shipments(pd.read_csv(f'{private_dir}/shipments_2023.csv'))
    ship24 = _normalize_shipments(pd.read_csv(f'{private_dir}/shipments_2024.csv'))
    gavi = pd.read_csv(f'{private_dir}/gavi_actuals.csv')
    pop = pd.read_csv(f'{data_dir}/population_by_age.csv').set_index('location')
    targets = pd.read_csv(f'{data_dir}/target_ages.csv').set_index('location')

    shipped = {}
    actual = {}
    cf = {}

    for loc, trow in targets.iterrows():
        if loc not in pop.index:
            print(f'SKIP {loc}: no population data')
            continue

        cohort = _cohort_size(pop.loc[loc], trow['cohort_upper_exclusive'])
        if np.isnan(cohort) or cohort == 0:
            print(f'SKIP {loc}: cohort unknown')
            continue

        wastage = trow['wastage_factor']
        sch23 = trow['schedule_2023']
        sch24 = trow['schedule_2024']

        d23 = ship23.loc[ship23['location'] == loc, 'doses'].sum()
        d24 = ship24.loc[ship24['location'] == loc, 'doses'].sum()

        # --- shipped: doses × wastage / schedule, divided by cohort.
        vax23 = (d23 * wastage / sch23) if (pd.notna(sch23) and sch23) else 0.0
        vax24 = (d24 * wastage / sch24) if (pd.notna(sch24) and sch24) else 0.0
        shipped[loc] = (vax23 + vax24) / cohort

        # --- cf: treat every dose as a 2-dose regimen ⇒ doses × wastage / 2.
        cf[loc] = (d23 + d24) * wastage / 2.0 / cohort

        # --- actual: from Gavi actuals nvax (girls reached), reshaped as
        #     doses = schedule × nvax. Excel: WUENIC!T = R + S
        #                   where R = sched_2023 × nvax_2023 / wastage
        #                         S = sched_2024 × nvax_2024 / wastage
        #     so AE = T and AF = AE / Y. Note wastage divides, not multiplies --
        #     T is in dose-space relative to cohort denominator already.
        iso3 = trow['iso3']
        g = gavi[(gavi['iso3'] == iso3) & (gavi['vaccine'] == 'hpv')]
        n23 = g.loc[g['year'] == 2023, 'nvax'].sum()
        n24 = g.loc[g['year'] == 2024, 'nvax'].sum()
        doses23 = (sch23 * n23 / wastage) if (pd.notna(sch23) and sch23) else 0.0
        doses24 = (sch24 * n24 / wastage) if (pd.notna(sch24) and sch24) else 0.0
        actual[loc] = (doses23 + doses24) / cohort

    return shipped, actual, cf


def _format_dict(name, d, locations_order):
    lines = [f'{name} = {{']
    for loc in locations_order:
        if loc in d:
            lines.append(f'    {loc!r:<16}: {d[loc]:.2f},')
        else:
            lines.append(f'    {loc!r:<16}: None,  # no data')
    lines.append('}')
    return '\n'.join(lines)


def write_module(shipped, actual, cf, outpath):
    locations_order = [
        'bangladesh', 'burkina faso', 'cambodia', 'cameroon', 'cote divoire',
        'ethiopia', 'gambia', 'laos', 'mali', 'mozambique', 'myanmar',
        'nigeria', 'sierra leone', 'tanzania', 'togo', 'zambia',
    ]
    header = (
        '"""\n'
        'Coverage dicts derived from Gavi shipments + actuals + WUENIC target ages.\n'
        'Auto-generated by scripts/compute_coverage.py -- do not hand-edit.\n'
        '"""\n\n'
    )
    body = '\n\n'.join([
        _format_dict('vx_coverage_shipped', shipped, locations_order),
        _format_dict('vx_coverage_actual', actual, locations_order),
        _format_dict('vx_coverage_cf', cf, locations_order),
    ])
    with open(outpath, 'w') as f:
        f.write(header + body + '\n')


def _diff_vs_locations(shipped, actual, cf):
    try:
        import locations as loc
    except ImportError:
        return
    print('\nDiff vs current locations.py (|delta| > 0.01 shown):')
    for label, computed, current in [
        ('shipped', shipped, loc.vx_coverage_shipped),
        ('actual',  actual,  loc.vx_coverage_actual),
        ('cf',      cf,      loc.vx_coverage_cf),
    ]:
        for k, v in computed.items():
            if v is None:
                continue
            cur = current.get(k)
            if cur is None:
                continue
            if abs(v - cur) > 0.01:
                print(f'  {label:<8} {k:<14} computed={v:.2f}  locations.py={cur:.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--private-dir', default='private')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--out', default='locations_coverage.py')
    args = parser.parse_args()

    shipped, actual, cf = compute(args.private_dir, args.data_dir)
    write_module(shipped, actual, cf, args.out)
    print(f'Wrote {args.out}')
    _diff_vs_locations(shipped, actual, cf)


if __name__ == '__main__':
    main()
