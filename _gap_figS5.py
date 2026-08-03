"""
Gap-fill driver: produce the v3 ``figS5_asr.csv`` (per-country age-standardized
cervical-cancer incidence-rate trajectory) that the plotting script needs.

The published pipeline reads ``<resfolder>/figS5_asr.csv`` with columns
``location, year, value, low, high``. The v3 build never produced it (no
``raw_results/<loc>.mres``), so we compute the ASR trajectory directly here.

For each of the 17 study countries we build the baseline sim, append an
``hpv.AgeResults`` analyzer that records ``cancer_incidence`` by WHO 5-year age
bins at each year 2000..2020, run a few seeds in the FOREGROUND at reduced
scale (ms_agent_ratio=3, n_agents=10000 -- multiscale-invariant for cancer),
age-standardize each year, and reduce to mean / min-max across seeds.

Run:  .venv/Scripts/python.exe _gap_figS5.py
"""
import os

os.environ.update(
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1',
    PYTHONIOENCODING='utf-8',
)

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv

import locations as loc
import run_sim as rs

# WHO World Standard Population weights over 5-year bins 0..85+ (18 bins).
# Copied from hpvsim_india/run_sim.py (canonical ASR helper).
ASR_AGE_EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
                          65, 70, 75, 80, 85, 100])
ASR_STD_WEIGHTS = np.array([.12, .10, .09, .09, .08, .08, .06, .06, .06, .06,
                            .05, .04, .04, .03, .02, .01, 0.005, 0.005, 0])[:-1]

YEARS = list(range(2000, 2021))  # 2000..2020 inclusive
N_SEEDS = 3
END = 2025            # >2020 so calendar-year 2020 is fully covered
MS_AGENT_RATIO = 3    # ms-invariant + memory-safe
N_AGENTS = 10000
OUTDIR = 'results/_v3gap5'


def asr_from_incidence(inc_by_age):
    """Age-standardized rate from a length-18 cancer_incidence-by-age array."""
    return float(np.dot(np.asarray(inc_by_age, dtype=float), ASR_STD_WEIGHTS))


def run_location(location):
    """Return (n_years, n_seeds) ASR matrix for one country."""
    dflocation = location.replace(' ', '_')
    calib_pars = sc.loadobj(f'results/{dflocation}_pars.obj')
    calib_pars.pop('hiv_pars', None)  # not running HIV here

    per_seed = []
    for seed in range(1, N_SEEDS + 1):
        az = hpv.AgeResults(name='asr_ageresults', result_args=sc.objdict(
            cancer_incidence=sc.objdict(
                years=[float(y) for y in YEARS],
                edges=ASR_AGE_EDGES.astype(float))))
        sim = rs.make_sim(location=location, calib_pars=calib_pars,
                          analyzers=[az], seed=seed, end=END,
                          ms_agent_ratio=MS_AGENT_RATIO, n_agents=N_AGENTS)
        sim.run(verbose=0.0)
        # The run sim holds a deep-copied analyzer; find the cancer_incidence one.
        ar = next(a for a in sim.analyzers.values()
                  if isinstance(a, hpv.AgeResults) and 'cancer_incidence' in a.outputs)
        out = ar.outputs['cancer_incidence']
        asr_ts = [asr_from_incidence(out[float(y)]) for y in YEARS]
        per_seed.append(asr_ts)
        print(f'    {location} seed {seed}: ASR(2020)={asr_ts[-1]:.2f}')
    return np.array(per_seed).T  # (n_years, n_seeds)


def _loc_path(location):
    return f'{OUTDIR}/_loc_{location.replace(" ", "_")}.csv'


def run_and_save(location):
    """Run one country and save its partial CSV (resumable)."""
    print(f'[{location}] running {N_SEEDS} seeds...')
    mat = run_location(location)  # (n_years, n_seeds)
    value, low, high = mat.mean(axis=1), mat.min(axis=1), mat.max(axis=1)
    rows = [dict(location=location, year=float(y), value=value[i],
                 low=low[i], high=high[i]) for i, y in enumerate(YEARS)]
    df = pd.DataFrame(rows, columns=['location', 'year', 'value', 'low', 'high'])
    df.to_csv(_loc_path(location), index=False)
    print(f'  -> {location} ASR(2020) mean={value[-1]:.2f} '
          f'[{low[-1]:.2f}, {high[-1]:.2f}]  saved {_loc_path(location)}')


def combine():
    """Merge all per-location partials into figS5_asr.csv."""
    parts = [pd.read_csv(_loc_path(l)) for l in loc.locations
             if os.path.exists(_loc_path(l))]
    df = pd.concat(parts, ignore_index=True)
    outpath = f'{OUTDIR}/figS5_asr.csv'
    df.to_csv(outpath, index=False)
    print(f'Wrote {outpath} ({len(df)} rows, {df.location.nunique()} countries)')


def main(locs=None, skip_existing=True):
    os.makedirs(OUTDIR, exist_ok=True)
    T = sc.timer()
    for location in (locs if locs is not None else loc.locations):
        if skip_existing and os.path.exists(_loc_path(location)):
            print(f'[{location}] already done, skipping')
            continue
        run_and_save(location)
    T.toc('batch done')


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if args == ['combine']:
        combine()
    else:
        main(locs=args or None)
