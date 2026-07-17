"""
Run single-dose vaccination scenarios (migrated to HPVsim v3).

v3 intervention changes:
* ``hpv.default_vx(prod_name=...)`` is gone; build the product with
  ``hpv.vx(name='nonavalent'|'bivalent', sterilizing_p=...)``. The v2
  ``imm_init`` (sterilizing probability) is now ``sterilizing_p``.
* ``hpv.campaign_vx(prob=, years=, product=, sex=, age_range=, name=)``.
  ``annual_prob`` is gone (a campaign applies ``prob`` once at its year).
  ``sex=0`` = females. Downstream interventions read upstream outcomes via
  ``sim.interventions['<name>'].outcomes[...]`` (``sim.get_intervention`` gone).
* No ``hpv.MultiSim``; sims are run directly per seed.

Cohort cancers are read from each sim's ``AgeResults`` analyzer (see
analyzers.py) and reduced across seeds.
"""

import os

os.environ.update(
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
)

import numpy as np
import sciris as sc
import starsim as ss
import hpvsim as hpv
import pandas as pd

import run_sim as rs
import analyzers as an
import locations as loc

# Settings
debug = 0
n_seeds = [20, 1][debug]
serial = False

SCENARIO_NAMES = ['No vaccination', 'Double dose',
                  'Single dose shipments', 'Single dose actual']


# %% Interventions
def make_st(screen_coverage=0.15, treat_coverage=0.7, start_year=2020):
    """Screen-and-treat cascade (v3). Not used by the dose scenarios below;
    migrated for completeness. Uses ``sim.interventions['<name>']`` wiring."""
    age_range = [30, 50]
    len_age_range = (age_range[1] - age_range[0]) / 2
    model_annual_screen_prob = 1 - (1 - screen_coverage) ** (1 / len_age_range)

    screening = hpv.routine_screening(
        prob=model_annual_screen_prob, start_year=start_year,
        product='hpv', age_range=age_range, name='screening')

    screen_positive = lambda sim: sim.interventions['screening'].outcomes['positive']
    assign_treatment = hpv.routine_triage(
        start_year=start_year, prob=1.0, product='tx_assigner',
        eligibility=screen_positive, name='tx_assigner')

    ablation_eligible = lambda sim: sim.interventions['tx_assigner'].outcomes['ablation']
    ablation = hpv.treat_num(prob=treat_coverage, product='ablation',
                             eligibility=ablation_eligible, name='ablation')

    excision_eligible = lambda sim: list(set(
        sim.interventions['tx_assigner'].outcomes['excision'].tolist() +
        sim.interventions['ablation'].outcomes['unsuccessful'].tolist()))
    excision = hpv.treat_num(prob=treat_coverage, product='excision',
                             eligibility=excision_eligible, name='excision')

    radiation_eligible = lambda sim: sim.interventions['tx_assigner'].outcomes['radiation']
    radiation = hpv.treat_num(prob=treat_coverage / 4, product=hpv.radiation(),
                              eligibility=radiation_eligible, name='radiation')

    return [screening, assign_treatment, ablation, excision, radiation]


def make_vx_scenarios(location=None, year=2024):
    if location in ['ethiopia', 'laos', 'zambia']:
        routine_age = (9, 17)
    elif location in ['togo']:
        routine_age = (9, 16)
    elif location in ['tanzania']:
        routine_age = (9, 19)
    else:
        routine_age = (9, 15)

    prod_name = 'bivalent' if location in ['bangladesh', 'cambodia', 'togo', 'zimbabwe'] else 'nonavalent'

    def single_dose():
        # v2 imm_init dict(uniform, 0.98, 0.99) -> mean sterilizing_p.
        return hpv.vx(name=prod_name, sterilizing_p=0.985)

    shipped_cov = loc.vx_coverage_shipped[location]
    actual_cov = loc.vx_coverage_actual[location]
    cf_cov = loc.vx_coverage_cf[location]

    scenarios = dict()
    scenarios['No vaccination'] = []
    scenarios['Single dose shipments'] = [hpv.campaign_vx(
        prob=shipped_cov, years=year, product=single_dose(), sex=0,
        age_range=routine_age, name='vax_campaign')]
    scenarios['Single dose actual'] = [hpv.campaign_vx(
        prob=actual_cov, years=year, product=single_dose(), sex=0,
        age_range=routine_age, name='vax_campaign')]
    scenarios['Double dose'] = [hpv.campaign_vx(
        prob=cf_cov, years=year, product=single_dose(), sex=0,
        age_range=routine_age, name='vax_campaign')]
    return scenarios


# %% Build + run
def run_scenario_seeds(location, calib_pars, scenario_name, end, seeds,
                       ms_agent_ratio=100, n_agents=None):
    """Run one scenario across seeds; return per-seed cohort arrays."""
    per_seed = []
    for seed in seeds:
        # Rebuild interventions per seed (sims deep-copy/consume them).
        intvs = make_vx_scenarios(location)[scenario_name]
        sim = rs.make_sim(location=location, calib_pars=calib_pars, debug=debug,
                          interventions=intvs, end=end, seed=seed,
                          ms_agent_ratio=ms_agent_ratio, n_agents=n_agents)
        sim.run()
        az = sim.analyzers['ageresults']
        per_seed.append(an.extract_cohort(az))
    return per_seed


def run_location(location, end=2125, seeds=None, ms_agent_ratio=100, n_agents=None):
    if seeds is None:
        seeds = list(range(n_seeds))
    dfl = location.replace(' ', '_')
    calib_pars = sc.loadobj(f'results/{dfl}_pars.obj')
    calib_pars.pop('hiv_pars', None)

    msim_dict = sc.objdict()
    for sname in SCENARIO_NAMES:
        per_seed = run_scenario_seeds(location, calib_pars, sname, end, seeds,
                                      ms_agent_ratio=ms_agent_ratio, n_agents=n_agents)
        red = an.reduce_cohort(per_seed)
        msim_dict[sname] = sc.objdict(
            raw_cohort_cancers=red.raw,
            cohort_cancers=red.cum_cancers_best,
            cohort_cancers_low=red.cum_cancers_low,
            cohort_cancers_high=red.cum_cancers_high,
        )
    sc.saveobj(f'results/{dfl}_vx_scens.obj', msim_dict)
    return msim_dict


def compile_direct(locations, out='results_direct.csv'):
    dfs = []
    for location in locations:
        dfl = location.replace(' ', '_')
        md = sc.loadobj(f'results/{dfl}_vx_scens.obj')
        dd = dict(location=location)
        for scen in md.keys():
            dd[scen] = md[scen]['cohort_cancers']
            dd[scen + ' - lb'] = md[scen]['cohort_cancers_low']
            dd[scen + ' - ub'] = md[scen]['cohort_cancers_high']
        dfs.append(pd.DataFrame(dd, index=[0]))
    pd.concat(dfs).to_csv(out)


# %% Run as a script (reduced-scale verification config)
if __name__ == '__main__':
    T = sc.timer()
    run_locations = ['bangladesh', 'nigeria']
    seeds = [0, 1, 2]
    end = 2125
    ms = 10          # reduced-scale (v3 cancer is multiscale-invariant)
    n_agents = 20000
    for location in run_locations:
        print(f'Running {location} ...', flush=True)
        run_location(location, end=end, seeds=seeds,
                     ms_agent_ratio=ms, n_agents=n_agents)
    compile_direct(run_locations)
    print('Done.')
    T.toc()
