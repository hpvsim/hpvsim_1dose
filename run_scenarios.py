"""
Run scenarios with varying numbers of doses
"""


# %% General settings

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
import hpvsim as hpv
import pandas as pd

# Imports from this repository
import run_sim as rs
import locations as loc

# Settings - used here and imported elsewhere
debug = 0
n_seeds = [20, 1][debug]  # How many seeds to run per cluster
serial = False
if serial: n_seeds = 1

# %% Create interventions


def make_st(screen_coverage=0.15, treat_coverage=0.7, start_year=2020):
    """ Make screening & treatment intervention """

    age_range = [30, 50]
    len_age_range = (age_range[1]-age_range[0])/2
    model_annual_screen_prob = 1 - (1 - screen_coverage)**(1/len_age_range)

    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | \
                                  (sim.t > (sim.people.date_screened + 5 / sim['dt']))
    screening = hpv.routine_screening(
        prob=model_annual_screen_prob,
        eligibility=screen_eligible,
        start_year=start_year,
        product='hpv',
        age_range=age_range,
        label='screening'
    )

    # Assign treatment
    screen_positive = lambda sim: sim.get_intervention('screening').outcomes['positive']
    assign_treatment = hpv.routine_triage(
        start_year=start_year,
        prob=1.0,
        annual_prob=False,
        product='tx_assigner',
        eligibility=screen_positive,
        label='tx assigner'
    )

    ablation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['ablation']
    ablation = hpv.treat_num(
        prob=treat_coverage,
        product='ablation',
        eligibility=ablation_eligible,
        label='ablation'
    )

    excision_eligible = lambda sim: list(set(sim.get_intervention('tx assigner').outcomes['excision'].tolist() +
                                             sim.get_intervention('ablation').outcomes['unsuccessful'].tolist()))
    excision = hpv.treat_num(
        prob=treat_coverage,
        product='excision',
        eligibility=excision_eligible,
        label='excision'
    )

    radiation_eligible = lambda sim: sim.get_intervention('tx assigner').outcomes['radiation']
    radiation = hpv.treat_num(
        prob=treat_coverage/4,  # assume an additional dropoff in CaTx coverage
        product=hpv.radiation(),
        eligibility=radiation_eligible,
        label='radiation'
    )

    st_intvs = [screening, assign_treatment, ablation, excision, radiation]

    return st_intvs


def make_vx_scenarios(location=None, year=2024):

    if location in ['ethiopia', 'laos', 'zambia']:
        routine_age = (9, 17)
    elif location in ['togo']:
        routine_age = (9, 16)
    elif location in ['tanzania']:
        routine_age = (9, 19)
    else:
        routine_age = (9, 15)

    vx_scenarios = dict()

    # Baseline
    vx_scenarios['No vaccination'] = []

    # Single dose
    if location in ['bangladesh', 'cambodia', 'zimbabwe']:  # 'togo'
        # For these countries, we use the single dose product
        product = 'bivalent'
    else:
        product = 'nonavalent'
    singledose = hpv.default_vx(prod_name=product)
    singledose.imm_init = dict(dist='uniform', par1=0.98, par2=0.99)
    eligibility = lambda sim: (sim.people.doses == 0)

    # Coverage levels
    shipped_coverage = loc.vx_coverage_shipped[location]
    actual_coverage = loc.vx_coverage_actual[location]
    cf_coverage = loc.vx_coverage_cf[location]

    # Interventions
    shipped = hpv.campaign_vx(
        prob=shipped_coverage,
        years=year,
        product=singledose,
        sex=0,
        age_range=routine_age,
        eligibility=eligibility,
        interpolate=False,
        annual_prob=False,
        label='Single dose shipments'
    )

    actual = hpv.campaign_vx(
        prob=actual_coverage,
        years=year,
        product=singledose,
        sex=0,
        age_range=routine_age,
        eligibility=eligibility,
        interpolate=False,
        annual_prob=False,
        label='Single dose actual'
    )

    cf = hpv.campaign_vx(
        prob=cf_coverage,
        years=year,
        product=singledose,
        sex=0,
        age_range=routine_age,
        eligibility=eligibility,
        interpolate=False,
        annual_prob=False,
        label='Double dose'
    )

    vx_scenarios['Single dose shipments'] = [shipped]
    vx_scenarios['Single dose actual'] = [actual]
    vx_scenarios['Double dose'] = [cf]

    return vx_scenarios


def make_sims(location=None, calib_pars=None, vx_scenarios=None, end=2100):
    """ Set up scenarios """

    # st_intv = make_st()

    all_msims = sc.autolist()
    for name, vx_intv in vx_scenarios.items():
        sims = sc.autolist()
        for seed in range(n_seeds):
            interventions = vx_intv #+ st_intv
            sim = rs.make_sim(location=location, calib_pars=calib_pars, debug=debug, interventions=interventions, end=end, seed=seed)
            sim.label = name
            sims += sim
        all_msims += hpv.MultiSim(sims)

    msim = hpv.MultiSim.merge(all_msims, base=False)

    return msim


def run_sims(location=None, calib_pars=None, vx_scenarios=None, end=2100, verbose=0.2):
    """ Run the simulations """
    msim = make_sims(location=location, calib_pars=calib_pars, vx_scenarios=vx_scenarios, end=end)
    msim.run(verbose=verbose)
    return msim


# %% Run as a script
if __name__ == '__main__':

    T = sc.timer()
    do_run = True
    do_process = True
    do_compile = True
    end = 2125

    # Run scenarios (usually on VMs, runs n_seeds in parallel over M scenarios)
    if do_run:

        for location in ['togo']:  #loc.locations:
            fnlocation = location.replace(' ', '_')
            calib_pars = sc.loadobj(f'results/{fnlocation}_pars.obj')
            if 'hiv_pars' in calib_pars:
                # Remove hiv_pars if it exists, as we are not running HIV simulations here
                calib_pars.pop('hiv_pars', None)
            vx_scenarios = make_vx_scenarios(location=location, year=2024)

            if serial:
                msim = make_sims(location=location, calib_pars=calib_pars, vx_scenarios=vx_scenarios, end=end)
                for sim in msim.sims:
                    sim.run(verbose=0.1)
            else:
                msim = run_sims(calib_pars=calib_pars, location=location, vx_scenarios=vx_scenarios, end=end)

            if do_process:

                metrics = ['year', 'asr_cancer_incidence', 'n_vaccinated', 'n_precin_by_age', 'n_females_alive_by_age', 'cancers', 'cancer_deaths']

                # Process results
                scen_labels = list(vx_scenarios.keys())
                mlist = msim.split(chunks=len(scen_labels))

                msim_dict = sc.objdict()
                for si, scen_label in enumerate(scen_labels):

                    # Deal with analyzer
                    msim = mlist[si]
                    base_analyzer = msim.sims[0].get_analyzer('cohort_cancers')
                    alist = [sim.get_analyzer('cohort_cancers') for sim in msim.sims]
                    reduced_analyzer = base_analyzer.reduce(alist)

                    reduced_sim = mlist[si].reduce(output=True)
                    mres = sc.objdict({metric: reduced_sim.results[metric] for metric in metrics})

                    mres['cohort_cancers'] = reduced_analyzer.cum_cancers_best
                    mres['cohort_cancers_low'] = reduced_analyzer.cum_cancers_low
                    mres['cohort_cancers_high'] = reduced_analyzer.cum_cancers_high
                    mres['raw_cohort_cancers'] = reduced_analyzer.raw
                    msim_dict[scen_label] = mres

                sc.saveobj(f'results/{fnlocation}_vx_scens.obj', msim_dict)

    which = 'direct'
    if do_compile:

        dfs = []
        for location in loc.locations:
            dd = dict()
            fnlocation = location.replace(' ', '_')
            msim_dict = sc.loadobj(f'results/{fnlocation}_vx_scens.obj')

            dd['location'] = location
            for scen in msim_dict.keys():
                if which == 'direct':
                    dd[scen] = msim_dict[scen]['cohort_cancers']
                    dd[scen+' - lb'] = msim_dict[scen]['cohort_cancers_low']
                    dd[scen+' - ub'] = msim_dict[scen]['cohort_cancers_high']
                elif which == 'indirect':
                    dd[scen] = msim_dict[scen]['cancers'].values[65:].sum()
                    dd[scen+' - lb'] = msim_dict[scen]['cancers'].low[65:].sum()
                    dd[scen+' - ub'] = msim_dict[scen]['cancers'].high[65:].sum()

            dfs += [pd.DataFrame(dd, index=[0])]
        ddf = pd.concat(dfs)
        ddf.to_csv(f'results_{which}.csv')

    print('Done.')
