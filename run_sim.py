"""
Define the HPVsim v3 simulation objects.

Migrated from HPVsim v2 to v3 (Starsim-based). Key v3 changes vs the v2 build:

* ``hpv.Sim(pars_dict)`` -> ``hpv.Sim(**kwargs)``; the network is now a
  ``hpv.SexualNetwork`` object passed via ``networks=`` and built from
  ``hpvsim.data.country._network_pars(location, overrides=...)``.
* v2 ``end=`` -> v3 ``stop=``.
* Per-genotype natural-history overrides go through ``genotype_pars=``.
  Durations (``dur_precin``/``dur_cin``) MUST be wrapped in ``ss.years(...)``
  time units (the v3 default genotype durations are ``ss.years``-wrapped;
  an unwrapped ``ss.lognorm_ex(mean=3)`` is interpreted in sub-year units and
  clears infection ~1/dt too fast, collapsing the epidemic).
* Global v2 ``beta`` -> per-genotype directional ``beta`` dict
  ``{'sexualnetwork': [beta*rel_beta*transf2m, beta*rel_beta*transm2f]}``.
  (Setting ``rel_beta`` alone does NOT recompute the transmission ``beta`` dict,
  which is built once from the genotype defaults, so beta is applied directly.)
* v2 ``cancer_fn`` with ``method='cin_integral'`` needs the CIN-curve shape
  params (``form``/``k``/``x_infl``/``ttc``) copied in from ``cin_fn`` (v3
  stores them on ``cancer_fn`` too; a partial override would drop them).
* The DHS male sexual-debut data is NaN for most study countries. v2 sampled a
  NaN lognormal as 0 (males always active); v3 would sample NaN and make males
  never active (killing transmission). We reproduce v2 by mapping NaN debut to
  a ~0 debut age.
* Multiscale + population scaling: ``people.scale`` is only the multiscale
  weight; results additionally apply a population scale. We set ``total_pop``
  to the location's start-year population so v3 counts are on the same
  real-population footing as v2 (which auto-scales).
"""
# Additions to handle numpy multithreading
import os

os.environ.update(
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
)

# Standard imports
import numpy as np
import sciris as sc
import starsim as ss
import hpvsim as hpv
import pandas as pd
from hpvsim.data import country as C

# Imports from this repository
import pars_data as dp
import utils as ut
import locations as loc
import analyzers as an

# %% Settings and filepaths
debug = 0  # Run with smaller population sizes and in serial

# Directional per-act transmission multipliers (v3 GenotypePars defaults;
# same for every genotype). Female->male = transf2m, male->female = transm2f.
TRANSF2M = 1.0
TRANSM2F = 3.69
DEFAULT_GENOTYPE_BETA = 0.25  # v3 GenotypePars.beta baseline


# %% v3 build helpers
def _to_annual_prob(p, dt):
    """Convert per-timestep probability to annual (project layer_probs were
    calibrated per-timestep; v3's ss.prob wrapping treats them as annual)."""
    p = np.clip(np.asarray(p, dtype=float), 0, 1 - 1e-10)
    return 1 - (1 - p) ** (1 / dt)


def _layer_probs_to_annual(layer_probs, dt):
    out = {}
    for lkey, lp in layer_probs.items():
        a = np.asarray(lp, dtype=float).copy()
        for row in (1, 2):  # row 0 = age bins; rows 1,2 = female,male participation
            a[row, :] = _to_annual_prob(a[row, :], dt)
        out[lkey] = a
    return out


def _fix_debut(debut):
    """Map NaN debut params (missing DHS male data) to a ~0 debut age,
    reproducing v2's NaN-lognormal-samples-as-0 behaviour."""
    out = {}
    for sex, d in debut.items():
        d = dict(d)
        if not np.isfinite(d.get('par1', np.nan)):
            out[sex] = dict(dist='uniform', par1=0.0, par2=0.001)
        else:
            out[sex] = d
    return out


def build_network(location, dt=0.25):
    """Build a v3 ``hpv.SexualNetwork`` from this project's per-country
    behaviour parameters (layer_probs, mixing, partners, debut)."""
    layer_probs = _layer_probs_to_annual(
        dp.make_layer_probs(location=location), dt
    )
    overrides = dict(
        layer_probs=layer_probs,
        mixing=dp.mixing[location],
        m_partners=dp.m_partners,
        f_partners=dp.f_partners,
        debut=_fix_debut(ut.make_sb_data(location=location)),
    )
    return hpv.SexualNetwork(**C._network_pars(location, overrides=overrides))


def build_genotype_pars(calib_pars):
    """Translate v2 calibrated genotype pars into a v3 ``genotype_pars`` dict.

    * global ``beta`` x per-genotype ``rel_beta`` -> directional beta dict;
    * ``dur_precin``/``dur_cin`` dist dicts -> ``ss.years``-wrapped lognormals;
    * ``cancer_fn`` merged with the CIN-curve shape params from ``cin_fn``.
    """
    if calib_pars is None:
        return None
    gpars_in = calib_pars.get('genotype_pars', {})
    beta = calib_pars.get('beta', DEFAULT_GENOTYPE_BETA)
    out = {}
    for g, d in gpars_in.items():
        o = {}
        cin = d.get('cin_fn', {})
        rel_beta = d.get('rel_beta', 1.0)
        eff = beta * rel_beta
        o['beta'] = {'sexualnetwork': [min(eff * TRANSF2M, 0.99),
                                       min(eff * TRANSM2F, 0.99)]}
        for k in ('dur_precin', 'dur_cin'):
            if k in d and isinstance(d[k], dict) and 'dist' in d[k]:
                o[k] = ss.lognorm_ex(mean=ss.years(d[k]['par1']),
                                     std=ss.years(d[k]['par2']))
        if 'cin_fn' in d:
            o['cin_fn'] = d['cin_fn']
        if 'cancer_fn' in d:
            shape = {kk: cin[kk] for kk in ('form', 'k', 'x_infl', 'ttc')
                     if kk in cin}
            o['cancer_fn'] = {**shape, **d['cancer_fn']}
        if 'sero_prob' in d:
            o['sero_prob'] = d['sero_prob']
        out[g] = o
    return out


def _total_pop(location, year):
    """Location population at ``year`` (so v3 pop-scaling matches v2)."""
    pt = hpv.load_country(location, year=int(year))['pop_total']
    row = pt.loc[pt.year == int(year), 'pop_size']
    if len(row):
        return float(row.iloc[0])
    return float(pt.pop_size.iloc[0])


# %% Simulation creation
def make_sim(location=None, calib=False, calib_pars=None, debug=0,
             interventions=None, analyzers=None, seed=1, end=None,
             cohort_start=2024, cohort_stop=2125,
             ms_agent_ratio=100, n_agents=None):
    """Define the v3 simulation for a location.

    ``analyzers`` (extra) are appended to the cohort-cancer AgeResults analyzer
    unless ``calib=True``. ``ms_agent_ratio``/``n_agents`` are exposed so
    reduced-scale verification runs can trade fidelity for speed (v3 cancer is
    multiscale-invariant, so a smaller ratio is unbiased).
    """
    if end is None:
        end = 2100
    if calib:
        end = 2020

    dt = [0.25, 1.0][debug]
    start = [1960, 1980][debug]
    if n_agents is None:
        n_agents = [20e3, 1e3][debug]

    net = build_network(location, dt=dt)
    genotype_pars = build_genotype_pars(calib_pars)

    # Analyzers: age-stratified cancers (1-year bins) at every year of the
    # cohort window, from which the moving vaccination-cohort band is extracted.
    if calib:
        analyzer_list = []
    else:
        cohort_years = list(range(int(cohort_start),
                                  min(int(end), int(cohort_stop)) + 1))
        analyzer_list = [an.make_cohort_analyzer(cohort_years)] + sc.tolist(analyzers)

    sim = hpv.Sim(
        location=location,
        genotypes=[16, 18, 'hi5', 'ohr'],
        genotype_pars=genotype_pars,
        init_hpv_dist=dp.init_genotype_dist[location],
        init_seeding='exclusive',
        start=start,
        stop=end,
        dt=dt,
        n_agents=n_agents,
        total_pop=_total_pop(location, start),
        ms_agent_ratio=ms_agent_ratio,
        networks=[net],
        interventions=interventions,
        analyzers=analyzer_list,
        rand_seed=seed,
        verbose=0.0,
    )
    return sim


# %% Simulation running
def run_sim(location=None, interventions=None, analyzers=None, debug=0, seed=1,
            verbose=0.2, do_save=True, calib_pars=None, end=2100, do_shrink=True):
    dflocation = location.replace(' ', '_')
    if calib_pars is None:
        calib_pars = sc.loadobj(f'results/{dflocation}_pars.obj')
        calib_pars.pop('hiv_pars', None)  # not running HIV here

    sim = make_sim(location=location, debug=debug, end=end, seed=seed,
                   interventions=interventions, analyzers=analyzers,
                   calib_pars=calib_pars)
    sim.label = f'{location}--{seed}'
    sim.run(verbose=verbose)
    if do_shrink:
        sim.shrink()
    if do_save:
        sim.save(f'results/{dflocation}.sim')
    return sim


def run_sims(locations=None, debug=False, verbose=-1, analyzers=None,
             do_save=False, **kwargs):
    """Run multiple single-country simulations in parallel."""
    kwargs = sc.mergedicts(dict(debug=debug, verbose=verbose,
                                analyzers=analyzers), kwargs)
    simlist = sc.parallelize(run_sim, iterkwargs=dict(location=locations),
                             kwargs=kwargs, serial=debug, die=True)
    sims = sc.objdict({loc_: sim for loc_, sim in zip(locations, simlist)})
    if do_save:
        for loc_, sim in sims.items():
            sim.save(f'results/{loc_}.sim')
    return sims


# %% Run as a script
if __name__ == '__main__':
    T = sc.timer()
    for location in loc.locations[:1]:
        sim = run_sim(location=location, end=2025, do_shrink=False, do_save=False)
    T.toc('Done')
