# scripts/

Reproduces the vaccine-coverage inputs that used to live only in
`HPVsim results 20251030.xlsx`.

## One-time: extract the XLSX

The workbook itself is gitignored because the `2023`, `2024`, and `Gavi actuals`
sheets contain confidential Gavi shipment data. Run the two extraction scripts
once against your local copy:

```
python scripts/export_xlsx_to_private.py   # → private/*.csv  (gitignored)
python scripts/export_xlsx_to_data.py      # → data/*.csv      (tracked)
```

`private/` is in `.gitignore`. Only public reference tables go to `data/`.

## Regenerate the coverage dicts

```
python scripts/compute_coverage.py --out locations_coverage.py
```

Prints any discrepancies against the current `locations.py` values.
The three dicts in `locations_coverage.py` correspond to the Excel columns:

| dict                   | Excel col | meaning                                         |
| ---------------------- | --------- | ----------------------------------------------- |
| `vx_coverage_shipped`  | AA        | doses × wastage / schedule, / cohort            |
| `vx_coverage_actual`   | AF        | Gavi actuals nvax × schedule / wastage, / cohort |
| `vx_coverage_cf`       | --        | (doses 2023 + doses 2024) × wastage / 2, / cf_cohort (2-dose counterfactual) |

Cohort denominators are per-country, pulled from `data/target_ages.csv`:

- `cohort_upper_exclusive` -- used for `shipped` and `actual`. Girls aged
  9..upper-1. Sourced from `HPVsim results` col X.
- `cf_upper_exclusive` -- used for `cf`. Under a 2-dose counterfactual, half as
  many girls are reached per dose pair, so they're spread over a broader
  catch-up cohort: ages 9..16 (upper=17) for most countries, or the country's
  own `cohort_upper_exclusive` if it already extends past age 16 (Tanzania
  stays at 19). Formula: `cf_upper_exclusive = max(cohort_upper_exclusive, 17)`.

## Regenerate the Fig 1 aggregates

```
python scripts/compute_fig1.py
```

Writes two CSVs under `data/`:

- `fig1_shipments_summary.csv` -- left block of the Excel `Impact analysis`
  sheet: total doses by regime (1-dose / 2-dose / switch intention) × year,
  plus "additional girls reachable" and "approx CC impact".
- `fig1_allocation.csv` -- right block: for each (year, schedule), doses and
  girls reached under three scenarios: `complete` (all shipments utilized),
  `actual` (Gavi actuals nvax), `counterfactual` (all doses on 2-dose schedule).

## Files

| file                              | status     | source                             |
| --------------------------------- | ---------- | ---------------------------------- |
| `private/shipments_2023.csv`      | gitignored | XLSX sheet `2023`                  |
| `private/shipments_2024.csv`      | gitignored | XLSX sheet `2024`                  |
| `private/gavi_actuals.csv`        | gitignored | XLSX sheet `Gavi actuals`          |
| `data/population_by_age.csv`      | tracked    | XLSX sheet `Coverage calcs` rows 3-20 |
| `data/target_ages.csv`            | tracked    | XLSX sheet `WUENIC` + `HPVsim results` col X |
| `data/fig1_shipments_summary.csv` | tracked    | derived                            |
| `data/fig1_allocation.csv`        | tracked    | derived                            |

## Notes

- **Nepal** is out of scope for the paper and is dropped everywhere in the
  pipeline.
- **Excel bug** discovered during the port: `Merged` sheet row 24 has Rwanda's
  iso3 as `RWN` instead of `RWA`, so the Excel `SUMIFS` misses Rwanda's 112,384
  girls reached in 2024. `compute_fig1.py` joins on the correct iso3 and yields
  2024 2-dose actual = 2.7185M (vs Excel 2.6061M).
