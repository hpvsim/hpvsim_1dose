"""
Custom analyzers (migrated to HPVsim v3 / Starsim).

The published analysis tracks cancers arising in the 2023/24 vaccination cohort
(agents aged 9-16 in 2024) as that birth cohort ages. In v2 this was a custom
``cohort_cancers`` analyzer that summed ``ppl.scale`` for agents whose
``date_cancerous == sim.t`` inside a moving age band.

In v3 the robust, engine-native way to get age-stratified *incident* cancers is
``hpv.AgeResults`` (which correctly applies both the multiscale weight and the
population scale, and — as of the M10 fix — accumulates incidence across all
sub-steps of a calendar year rather than only the final dt tick). We therefore
build an ``AgeResults`` analyzer with 1-year age bins at every year of the
cohort window and extract the moving cohort band in post-processing.
"""

import numpy as np
import sciris as sc
import hpvsim as hpv

# Fine (1-year) age bins spanning the full lifespan.
COHORT_EDGES = np.arange(0, 101, 1.0)


def make_cohort_analyzer(years, edges=None):
    """Return an ``hpv.AgeResults`` analyzer recording incident ``cancers`` by
    1-year age bin at each requested calendar year."""
    if edges is None:
        edges = COHORT_EDGES
    years = [float(y) for y in years]
    return hpv.AgeResults(result_args=sc.objdict(
        cancers=sc.objdict(years=years, edges=np.asarray(edges, dtype=float)),
    ))


def extract_cohort(analyzer, start=2024, cohort_age=(9, 16), edges=None):
    """Extract per-year cancers in the moving vaccination cohort.

    The cohort is aged ``cohort_age`` (default 9-16) in ``start`` (2024) and
    ages one year per calendar year, so in year ``y`` it occupies ages
    ``[cohort_age[0] + (y-start), cohort_age[1] + (y-start)]``. Returns a
    1-D array of scaled incident cancers per year, ordered by the analyzer's
    configured years.

    ``analyzer`` must be the AgeResults instance held by the *run* sim
    (``sim.analyzers['ageresults']``), because ``hpv.Sim`` deep-copies
    analyzers at construction — the object passed in at build time is a
    stale copy that never runs.
    """
    if edges is None:
        edges = COHORT_EDGES
    out = analyzer.outputs['cancers']
    years = sorted(out.keys())
    vals = []
    for y in years:
        el = y - start
        lo = cohort_age[0] + el
        hi = cohort_age[1] + el
        arr = np.asarray(out[y])
        binlo = max(0, int(np.floor(lo)))
        binhi = min(len(arr), int(np.ceil(hi)))
        vals.append(float(arr[binlo:binhi].sum()) if binlo < binhi else 0.0)
    return np.array(vals)


def reduce_cohort(per_seed_arrays, quantiles=(0.1, 0.9)):
    """Stack per-seed cohort arrays into ``raw`` (n_years, n_seeds) and compute
    median/low/high cumulative trajectories (matching the v2 reduce output that
    the CSV extractors expect)."""
    raw = np.stack(per_seed_arrays, axis=-1)  # (n_years, n_seeds)
    lo, hi = quantiles
    med = np.cumsum(np.quantile(raw, 0.5, axis=-1))
    low = np.cumsum(np.quantile(raw, lo, axis=-1))
    high = np.cumsum(np.quantile(raw, hi, axis=-1))
    sums = raw.sum(axis=0)  # per-seed lifetime cohort total
    return sc.objdict(
        raw=raw,
        cum_med=med, cum_low=low, cum_high=high,
        cum_cancers_best=float(np.quantile(sums, 0.5)),
        cum_cancers_low=float(np.quantile(sums, lo)),
        cum_cancers_high=float(np.quantile(sums, hi)),
    )


# ---------------------------------------------------------------------------
# Behaviour-calibration analyzers (AFS, prop_married).
#
# These were used only for the DHS behaviour-fit diagnostics (read_sbdata.py),
# not on the cancer/figure path. They rely on v2 People attributes
# (``people.level0``, ``people.n_rships``, ``people.current_partners``,
# ``hpv.true``) that changed in v3 and have NOT been ported. They are retained
# here for reference; port to the v3 People/Network API before use.
# ---------------------------------------------------------------------------
