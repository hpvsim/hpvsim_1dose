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
| `vx_coverage_shipped`  | AA        | doses shipped × wastage / schedule, / cohort    |
| `vx_coverage_actual`   | AF        | doses from Gavi actuals × schedule / wastage, / cohort |
| `vx_coverage_cf`       | AC        | (doses 2023 + doses 2024) × wastage / 2, / cohort (counterfactual 2-dose schedule) |

where cohort = sum of girls aged 9..`cohort_upper_exclusive`-1
(per-country, from `data/target_ages.csv`).

## Files

| file                              | status     | source                             |
| --------------------------------- | ---------- | ---------------------------------- |
| `private/shipments_2023.csv`      | gitignored | XLSX sheet `2023`                  |
| `private/shipments_2024.csv`      | gitignored | XLSX sheet `2024`                  |
| `private/gavi_actuals.csv`        | gitignored | XLSX sheet `Gavi actuals`          |
| `data/population_by_age.csv`      | tracked    | XLSX sheet `Coverage calcs` rows 3-20 |
| `data/target_ages.csv`            | tracked    | XLSX sheet `WUENIC` + `HPVsim results` col X |

## Notes

- **Nepal** has no population data in the XLSX (outside the modeled cohort).
  `compute_coverage.py` emits `None` for Nepal; the downstream
  `locations.py` keeps its handwritten value (`vx_coverage_actual[nepal]=0.00`).
- Still to port: the `Impact analysis` sheet aggregates that feed Fig 1
  (total shipped doses 2023/24 by schedule, switch intention).
