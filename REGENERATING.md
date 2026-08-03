# Regenerating the v3 figures (hpvsim_1dose)

Which script produces which figure for the v2→v3 review. Run from this repo dir with the v3
venv (`.venv`, hpvsim 3.0.0); confirm `hpvsim.__version__ == '3.0.0'` first (editable-install trap).
All v3 runs use `ms_agent_ratio=3` (validated ms-invariant, ~30× less memory than 100) — run in a
normal foreground shell session, not a fragile long background job.

| Figure | Driver → output | Then plot |
|---|---|---|
| fig2 (cumulative cancers/averted), fig3 (by country), figS6 (per-country trajectories) | `python _regen_seq.py` — resumable: runs all 17 countries' vaccination scenarios (2 seeds), saves per-country `results/<loc>_vx_scens.obj`, then extracts `results/_v3gap_scens/{fig2_*,figS6_country_ts,results_direct}.csv` | `python plot_fig2_ts.py --resfolder results/_v3gap_scens`; `plot_fig3_countries.py`; `plot_figS6_country_ts.py` |
| figS4 (calibration, cancers by age) | extracted from the existing `results/<loc>_calib_reduced.obj` via `utils.extract_figS4_calib_csvs(loc.locations, calib_dir='results', out_dir=<dir>)` | `python plot_figS4_calib.py --resfolder <dir>` |
| figS5 (per-country ASR vs GLOBOCAN) | `python _gap_figS5.py` — per country, runs the baseline sim with an `hpv.AgeResults(cancer_incidence)` analyzer, age-standardizes (WHO weights), writes `results/_v3gap5/figS5_asr.csv` | `python plot_figS5_asr.py --resfolder results/_v3gap5` |

Notes:
- The committed `results/v3.0_baseline/` cancer CSVs are the **stale pre-recalibration collapse**
  (all zeros) — regenerate with `_regen_seq.py` (it uses the recalibrated per-country network pars).
- v3 absolute cancer counts are ~700× smaller than the v2.2.6 baseline (`total_pop=n_agents` vs
  national scaling), and figS5's ASR runs high (cancer over-prediction) — both expected; compare
  trends/relative effects. See the review repo `hpvsim_v23_migration_review` for the full write-up.
