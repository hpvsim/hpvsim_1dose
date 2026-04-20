# HPVsim 1-dose analysis

Multi-country analysis of the impact of the WHO single-dose HPV vaccination recommendation across 17 LMICs. Compares counterfactual 2-dose allocations, single-dose regimens with complete utilization, and single-dose regimens with actual observed shipment/uptake, quantifying cancers averted in the 2023/24 vaccination cohort.

**Results in this repository were produced with HPVsim v2.2.6.** Plot-ready baselines live in [`results/v2.2.6_baseline/`](results/v2.2.6_baseline/); all plot scripts default to reading from this folder.

## Installation

```bash
pip install hpvsim==2.2.6 seaborn scipy
```

Python 3.9+.

## Workflow: heavy sims on VM, plots locally

Heavy runs produce `.obj` / `.mres` calibration + scenario outputs that live under `results/` (tracked, small ones) and `raw_results/` (not tracked, per-seed outputs). A final aggregation step produces lightweight plot-ready CSVs under `results/` which get frozen into a versioned `results/v<version>_baseline/` dir and committed. Local plot scripts read the CSVs — no pickles in the plotting path.

### Heavy (VM) scripts

| Script | What it produces |
|---|---|
| `run_calibration.py` | Per-country calibration: `results/{loc}_pars.obj`, `results/{loc}_calib_reduced.obj` |
| `run_scenarios.py` | Per-country vaccination scenarios: `results/{loc}_vx_scens.obj`, top-level `results_direct.csv` / `results_indirect.csv` |
| `run_sim.py` | Per-country `.mres` time-series output: `raw_results/{loc}.mres` |
| `run_pop_estimates.py` | Supporting demographic calculations (`age_data.csv`) |

After heavy runs complete, aggregate into CSVs:

```python
import utils as ut
ut.extract_all_csvs()   # writes results/fig2_*.csv, figS4_*.csv, figS5_*.csv, figS6_*.csv
```

### Plot scripts (local)

| Script | Manuscript figure |
|---|---|
| `plot_fig2_ts.py` | Fig 2 — cumulative cancers + cancers averted in the vaccination cohort, aggregated across countries |
| `plot_fig3_countries.py` | Fig 3 — cancers averted + girls vaccinated + NNV by country |
| `plot_figS4_calib.py` | Fig S4 — calibration boxplots (cancers by age) per country |
| `plot_figS5_asr.py` | Fig S5 — per-country ASR cancer incidence vs Globocan 2020 |
| `plot_figS6_country_ts.py` | Fig S6 — per-country cumulative-cancer trajectories by scenario |

Each plot script takes `--resfolder` (default `results/v2.2.6_baseline`) and `--outpath` arguments.

### Other files

- `locations.py` — the 17 study countries + coverage denominators
- `analyzers.py` — custom cohort analyzers
- `utils.py` — CSV extractors (`extract_*_csvs`) + calibration + behaviour helpers
- `compare_baselines.py` — cross-version comparison (see below)
- `pars_data.py`, `read_sbdata.py` — calibration pars + behavioural data ingestion

## Reproducing the figures

```bash
git clone git@github.com:hpvsim/hpvsim_1dose.git
cd hpvsim_1dose
pip install hpvsim==2.2.6 seaborn scipy

# Render every figure from the committed v2.2.6 baseline
python plot_fig2_ts.py
python plot_fig3_countries.py
python plot_figS4_calib.py
python plot_figS5_asr.py
python plot_figS6_country_ts.py
```

## Cross-version comparison (v2.2.6 → v2.3 → v3.0)

When a new HPVsim version ships, regenerate the baseline in a clean env and compare side-by-side:

```bash
# 1. On a VM, in a clean env pinned to the new version
conda create -n hpvsim230 python=3.11 -y && conda activate hpvsim230
pip install hpvsim==2.3.0 seaborn scipy

# 2. Re-run the heavy scripts + CSV aggregation
python run_calibration.py
python run_scenarios.py
python run_sim.py
python -c "import utils as ut; ut.extract_all_csvs()"

# 3. Freeze the fresh CSVs into a versioned baseline dir
mkdir -p results/v2.3.0_baseline
cp results/fig*.csv results/v2.3.0_baseline/
cp results_direct.csv results/v2.3.0_baseline/

# 4. Commit + push, then locally compare
python compare_baselines.py --baselines v2.2.6_baseline v2.3.0_baseline
```

`compare_baselines.py` prints cumulative-cancers-at-2125 tables across baselines and emits `figures/compare_baselines.png` (cumulative-cancer trajectories overlaid per scenario). Extends trivially to v3.0 by appending the new baseline name to `--baselines`.

## Inputs

- `data/` — per-country Globocan cancer incidence, age pyramids, DHS behavioural parameters, ART coverage
- `popsizes.csv`, `popsizes_older.csv` — demographic inputs for coverage denominators

## Further information

See [hpvsim.org](https://hpvsim.org) and [docs.hpvsim.org](https://docs.hpvsim.org).
