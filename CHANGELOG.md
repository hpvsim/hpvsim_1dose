# Changelog

## 2026-04-19 — HPVsim v2.2.6 lift

- Split the workflow so heavy sims run on a VM (`run_calibration.py`, `run_scenarios.py`, `run_sim.py` → saves `.obj` / `.mres` to `raw_results/` and `results/`) and then extract lightweight plot-ready CSVs. Local plot scripts read the CSVs (no pickles).
- Added `utils.extract_all_csvs()` which aggregates across the 17 countries and produces:
  - `fig2_res_stats.csv` + `fig2_diffs.csv` — cumulative cancers per scenario / cancers averted over time (fig 2)
  - `figS4_calib.csv` + `figS4_targets.csv` — per-country calibration trials + Globocan targets (fig S4)
  - `figS5_asr.csv` — per-country ASR time series with 95% bands (fig S5)
  - `figS6_country_ts.csv` — per-country per-scenario cumulative cancers (fig S6)
- Refactored `plot_fig2_ts.py`, `plot_fig3_countries.py`, `plot_figS4_calib.py`, `plot_figS5_asr.py`, `plot_figS6_country_ts.py` to accept `--resfolder` / `--outpath` and default to `results/v2.2.6_baseline/`.
- Froze `results/v2.2.6_baseline/` (~1.3 MB, 7 CSVs) as the committed baseline.
- Added `compare_baselines.py` for cross-version (v2.2.6 → v2.3 → v3.0) comparison.
- Added README with VM/local workflow and migration instructions, LICENSE (MIT), CHANGELOG.
- `.gitignore`: excludes `raw_results/`, `figures/`, `*.zip`, transient `.sim`/`.msim`/`.mres` in `results/`, while keeping the tracked `*_pars.obj`, `*_calib_reduced.obj`, `*_vx_scens.obj` calibration/scenario artifacts.

## Earlier

- Original 17-country calibration + single-dose scenario analysis for the 1-dose HPV vaccine impact paper: `run_calibration.py` (per-country calibrations), `run_scenarios.py` (No vax / Double dose / Single-dose shipments / Single-dose actual), `run_sim.py` (per-country `.mres` for ASR time series).
