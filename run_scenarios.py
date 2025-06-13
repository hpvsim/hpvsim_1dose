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


def make_vx_scenarios(location=None, product='bivalent', year=2023):

    routine_age = (9, 10)
    mac11_age = (10, 15)
    mac15_age = (15, 17)

    vx_scenarios = dict()

    # Baseline
    vx_scenarios['Baseline'] = []

    # Single dose
    singledose = hpv.default_vx(prod_name=product)
    singledose.imm_init = dict(dist='beta_mean', par1=0.97, par2=0.025)
    eligibility = lambda sim: (sim.people.doses == 0)

    # 2023 routine
    sq_routine_2023 = loc.vx_coverage_2023_routine[location]
    cf_routine_2023 = loc.vx_coverage_2023_cf_routine[location]

    # 2024 routine
    sq_routine_2024 = loc.vx_coverage_2024_routine[location]
    cf_routine_2024 = loc.vx_coverage_2024_cf_routine[location]

    # MACs in both years
    sq_mac11_14_both = loc.vx_coverage_both_macs[location][0]
    sq_mac15_16_both = loc.vx_coverage_both_macs[location][1]
    cf_mac11_14_both = loc.vx_coverage_both_cf_macs[location][0]

    # 2023 routine - status quo
    routine_single_vx_23 = hpv.campaign_vx(
        prob=sq_routine_2023,
        years=year,
        product=singledose,
        age_range=routine_age,
        eligibility=eligibility,
        interpolate=False,
        label='Single dose routine 2023'
    )

    # 2023 routine - counterfactual
    routine_single_vx_cf_23 = hpv.campaign_vx(
        prob=cf_routine_2023,
        years=year,
        product=singledose,
        age_range=routine_age,
        eligibility=eligibility,
        interpolate=False,
        label='Double dose routine 2023'
    )

    # 2024 routine - status quo
    routine_single_vx_24 = hpv.campaign_vx(
        prob=sq_routine_2024,
        years=year+1,
        product=singledose,
        age_range=routine_age,
        eligibility=eligibility,
        interpolate=False,
        label='Single dose routine 2024'
    )

    # 2024 routine - counterfactual
    routine_single_vx_cf_24 = hpv.campaign_vx(
        prob=cf_routine_2024,
        years=year+1,
        product=singledose,
        age_range=routine_age,
        eligibility=eligibility,
        interpolate=False,
        label='Double dose routine 2024'
    )

    # MACs in both years - status quo
    mac11_single_vx = hpv.campaign_vx(
        prob=sq_mac11_14_both,
        years=year,
        product=singledose,
        age_range=mac11_age,
        eligibility=eligibility,
        interpolate=False,
        label='Single dose MAC 11-14'
    )
    mac15_single_vx = hpv.campaign_vx(
        prob=sq_mac15_16_both,
        years=year,
        product=singledose,
        age_range=mac15_age,
        eligibility=eligibility,
        interpolate=False,
        label='Single dose MAC 15-16'
    )

    vx_scenarios['Single dose'] = [routine_single_vx_23, routine_single_vx_24, mac11_single_vx, mac15_single_vx]

    # Both counterfactual
    mac11_single_vx_cf = hpv.campaign_vx(
        prob=cf_mac11_14_both,
        years=year,
        product=singledose,
        age_range=mac11_age,
        eligibility=eligibility,
        interpolate=False,
        label='Double dose MAC 11-14'
    )

    vx_scenarios['Double dose'] = [routine_single_vx_cf_23, routine_single_vx_cf_24, mac11_single_vx_cf]

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
    end = 2100

    # Run scenarios (usually on VMs, runs n_seeds in parallel over M scenarios)
    if do_run:
        for location in loc.locations:
            fnlocation = location.replace(' ', '_')
            calib_pars = sc.loadobj(f'results/{fnlocation}_pars.obj')
            if 'hiv_pars' in calib_pars:
                # Remove hiv_pars if it exists, as we are not running HIV simulations here
                calib_pars.pop('hiv_pars', None)
            vx_scenarios = make_vx_scenarios(location=location, year=2023)

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

                    # for ii, intv in enumerate(reduced_sim['interventions']):
                    #     intv_label = intv.label
                    #     mres[intv_label] = reduced_sim['interventions'][ii].n_products_used
                    #     if scen_label == 'Double dose':
                    #         mres[intv_label] = reduced_sim['interventions'][ii].n_products_used[:] * 2

                    msim_dict[scen_label] = mres

                sc.saveobj(f'results/{fnlocation}_vx_scens.obj', msim_dict)

    if do_compile:
        dfs = []
        for location in loc.locations:
            dd = dict()
            fnlocation = location.replace(' ', '_')
            msim_dict = sc.loadobj(f'results/{fnlocation}_vx_scens.obj')

            # # Process diffs
            # diffs = msim_dict['Double dose']['raw_cohort_cancers'] - msim_dict['Single dose']['raw_cohort_cancers']
            # diffs_med = np.median(diffs, axis=1).sum()
            # diffs_lb = np.quantile(diffs, q=0.05, axis=1).sum()
            # diffs_ub = np.quantile(diffs, q=0.95, axis=1).sum()

            dd['location'] = location
            for scen in msim_dict.keys():
                dd[scen] = msim_dict[scen]['cohort_cancers']
                dd[scen+' - lb'] = msim_dict[scen]['cohort_cancers_low']
                dd[scen+' - ub'] = msim_dict[scen]['cohort_cancers_high']
            dfs += [pd.DataFrame(dd, index=[0])]
        ddf = pd.concat(dfs)
        ddf.to_csv('results.csv')

    print('Done.')
