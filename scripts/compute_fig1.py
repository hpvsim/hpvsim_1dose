"""
Reproduces the aggregate numbers that drive Fig 1 ("Impact analysis" sheet).

Two outputs:

    data/fig1_shipments_summary.csv
        left block of `Impact analysis` -- total doses by year × regime
        (1-dose vs 2-dose vs switch intention), plus "additional reachable"
        (= doses/2) and "approximate CC impact" (= girls/70).

    data/fig1_allocation.csv
        right block of `Impact analysis` -- for each (year, schedule) bucket,
        doses allocated and girls reached under three scenarios:
            complete:       actual shipments, girls = doses / schedule
            actual:         Gavi actuals nvax, doses = nvax × schedule
            counterfactual: every dose allocated to 2-dose ⇒ all 1-dose +
                            2-dose doses counted in the 2-dose bucket

Inputs: `private/shipments_*.csv`, `private/gavi_actuals.csv` (gitignored).

Usage:
    python scripts/compute_fig1.py

Known discrepancy vs the Excel workbook: Merged sheet row 24 has Rwanda's
iso3 as 'RWN' instead of 'RWA', so the Excel's SUMIFS misses Rwanda's 112,384
girls reached in 2024. This script joins on the correct iso3 and so yields
2024 2-dose actual = 2.7185M (vs Excel 2.6061M).
"""
import argparse
import os
import unicodedata

import pandas as pd


def _fold(name):
    """Normalize a country name for joining: lowercase, strip diacritics."""
    if not isinstance(name, str):
        return None
    return ''.join(
        c for c in unicodedata.normalize('NFKD', name.lower())
        if not unicodedata.combining(c)
    ).strip()



def _load_shipments(private_dir):
    s23 = pd.read_csv(f'{private_dir}/shipments_2023.csv')
    s24 = pd.read_csv(f'{private_dir}/shipments_2024.csv')

    # Excel treats non-integer schedule (e.g. 'Yes', 'NITAG recommneded') as
    # outside the 1-dose/2-dose buckets. Only 'Yes' flips a row into the
    # switch-intention bucket. 'NITAG recommneded' falls back to schedule=2.
    for df in (s23, s24):
        df['schedule'] = pd.to_numeric(df['schedule'], errors='coerce')

    s23['year'] = 2023
    s24['year'] = 2024
    if 'switch_intention' not in s24.columns:
        s24['switch_intention'] = None
    if 'switch_intention' not in s23.columns:
        s23['switch_intention'] = None

    return pd.concat(
        [s23[['year', 'country', 'doses', 'schedule', 'switch_intention']],
         s24[['year', 'country', 'doses', 'schedule', 'switch_intention']]],
        ignore_index=True,
    )


def _bucketize(ship):
    """Add schedule_bucket ('1-dose' vs '2-dose') and is_switch flag.

    `switch` is cross-cutting: switch-intention rows are ALSO counted in the
    2-dose bucket (Excel row 3 sums all sched=2; row 4 is the switch subset).
    """
    ship = ship.copy()
    ship['schedule_bucket'] = ship['schedule'].map({1: '1-dose', 2: '2-dose'})
    ship['is_switch'] = ship['switch_intention'] == 'Yes'
    return ship


def _doses_by_year(ship, mask=None):
    """Return {2023: x, 2024: y} for doses matching `mask`."""
    sub = ship if mask is None else ship[mask]
    out = sub.groupby('year')['doses'].sum().to_dict()
    return {y: out.get(y, 0) for y in (2023, 2024)}


def compute_summary(ship):
    """Rows 2-5 of Impact analysis (left block), in absolute doses."""
    rows = []
    # Row 2: Single dose use -- sched=1 only.
    d1 = _doses_by_year(ship, ship['schedule_bucket'] == '1-dose')
    # Row 3: 2-dose use -- ALL sched=2 (includes switch subset).
    d2 = _doses_by_year(ship, ship['schedule_bucket'] == '2-dose')
    # Row 4: Switch intention -- only the switch subset (cross-cutting).
    dsw = _doses_by_year(ship, ship['is_switch'])

    def _row(regime, d, additional, impact):
        return {
            'regime': regime,
            'doses_2023': d[2023],
            'doses_2024': d[2024],
            'doses_total': d[2023] + d[2024],
            'additional_girls_reachable': additional,
            'approx_cc_impact': impact,
        }

    total_1dose = d1[2023] + d1[2024]
    total_switch = dsw[2023] + dsw[2024]
    rows.append(_row('1-dose', d1, total_1dose / 2, total_1dose / 2 / 70))
    rows.append(_row('2-dose', d2, None, None))
    rows.append(_row('switch', dsw, total_switch / 2, total_switch / 2 / 70))
    return pd.DataFrame(rows)


def _actuals_by_schedule(ship, private_dir):
    """Sum Gavi actuals nvax by (year, schedule_bucket).

    Joins shipments → Gavi actuals on diacritic-folded country name
    ('Cote d\'Ivoire' ↔ 'Côte d\'Ivoire', 'Lao PDR' ↔ 'Lao...').
    """
    gavi = pd.read_csv(f'{private_dir}/gavi_actuals.csv')
    gavi = gavi[gavi['vaccine'] == 'hpv']
    gavi = gavi.copy()
    # Reconcile name variants between shipment sheets and Gavi actuals.
    # Keys are the diacritic-folded Gavi-actuals name; values are the canonical
    # form used in the shipment sheets.
    GAVI_TO_SHIP = {
        "lao people's democratic republic": 'lao pdr',
        'united republic of tanzania':      'tanzania',
        'sao tome and principe':            'sao tome & principe',
        'republic of moldova':              'moldova',
        'democratic republic of the congo': 'dr congo',
    }
    gavi['country_key'] = gavi['country'].map(_fold).replace(GAVI_TO_SHIP)
    country_to_iso = dict(zip(gavi['country_key'], gavi['iso3']))

    ship = ship.copy()
    ship['country_key'] = ship['country'].map(_fold)
    ship['iso3'] = ship['country_key'].map(country_to_iso)

    unresolved = ship[ship['iso3'].isna()]['country'].unique()
    if len(unresolved):
        print(f'WARNING: no iso3 for shipment countries: {list(unresolved)}')

    merged = ship.merge(
        gavi[['iso3', 'year', 'nvax']], on=['iso3', 'year'], how='left'
    )
    merged['nvax'] = merged['nvax'].fillna(0)
    return merged.groupby(['year', 'schedule_bucket'])['nvax'].sum().reset_index()


def compute_allocation(ship, private_dir):
    """Rows 3-6 of Impact analysis (right block), in MILLIONS.

    Split by (year, schedule) only; switch intention is folded into 2-dose.
    Uses ALL shipment countries (Excel's Merged sheet covers 32 Gavi countries,
    not just the 17 modeled for impact).
    """
    by_ys = ship.groupby(['year', 'schedule_bucket'])['doses'].sum().reset_index()
    actuals = _actuals_by_schedule(ship, private_dir)

    rows = []
    for _, row in by_ys.iterrows():
        year = int(row['year'])
        bucket = row['schedule_bucket']
        if bucket is None or (isinstance(bucket, float) and pd.isna(bucket)):
            continue
        schedule = 1 if bucket == '1-dose' else 2
        complete_doses = row['doses'] / 1e6
        complete_girls = complete_doses / schedule

        act = actuals[(actuals['year'] == year) & (actuals['schedule_bucket'] == bucket)]
        actual_girls = (act['nvax'].sum() / 1e6) if not act.empty else 0.0
        actual_doses = actual_girls * schedule

        rows.append({
            'year': year,
            'schedule': schedule,
            'complete_doses_m': complete_doses,
            'complete_girls_m': complete_girls,
            'actual_doses_m': actual_doses,
            'actual_girls_m': actual_girls,
        })

    out = pd.DataFrame(rows)

    # Counterfactual: every dose allocated to 2-dose. So for each year, the 2-dose
    # bucket gets its own doses + the 1-dose bucket's doses; 1-dose bucket is 0.
    cf_doses = {}
    for year in sorted(out['year'].unique()):
        year_total = out.loc[out['year'] == year, 'complete_doses_m'].sum()
        cf_doses[(year, 1)] = 0.0
        cf_doses[(year, 2)] = year_total
    out['cf_doses_m'] = out.apply(
        lambda r: cf_doses.get((r['year'], r['schedule']), 0.0), axis=1
    )
    out['cf_girls_m'] = out['cf_doses_m'] / 2.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--private-dir', default='private')
    parser.add_argument('--data-dir', default='data')
    args = parser.parse_args()

    ship = _bucketize(_load_shipments(args.private_dir))

    summary = compute_summary(ship)
    alloc = compute_allocation(ship, args.private_dir)

    os.makedirs(args.data_dir, exist_ok=True)
    summary.to_csv(f'{args.data_dir}/fig1_shipments_summary.csv', index=False, float_format='%.4f')
    alloc.to_csv(f'{args.data_dir}/fig1_allocation.csv', index=False, float_format='%.4f')
    print(f'Wrote {args.data_dir}/fig1_shipments_summary.csv')
    print(f'Wrote {args.data_dir}/fig1_allocation.csv')
    print()
    print('Summary (doses):')
    print(summary.to_string(index=False))
    print()
    print('Allocation (millions):')
    print(alloc.to_string(index=False))


if __name__ == '__main__':
    main()
