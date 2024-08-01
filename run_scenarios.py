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

# Imports from this repository
import run_sim as rs
import locations as loc

# Settings - used here and imported elsewhere
debug = 0
n_seeds = [20, 1][debug]  # How many seeds to run per cluster


# %% Create interventions


def make_vx_scenarios(start_year=2025, product='bivalent', end=2100):

    age_range = (9, 14)
    routine_age = (age_range[0], age_range[0]+1)

    vx_scenarios = dict()

    # Baseline
    vx_scenarios['Baseline'] = []

    # Single dose
    singledose = hpv.default_vx(prod_name=product)
    singledose.imm_init = dict(dist='beta_mean', par1=0.95, par2=0.025)
    eligibility = lambda sim: (sim.people.doses == 0)

    years = np.arange(start_year, end+1, 1)
    n_years = len(years)
    d1_coverage = np.concatenate([np.linspace(0.1, 0.9, 5), np.repeat(0.9, n_years-5)])
    routine_vx1 = hpv.routine_vx(
        prob=d1_coverage,
        years=years,
        product=singledose,
        age_range=routine_age,
        eligibility=eligibility,
        label='Routine vx'
    )
    vx_scenarios['Single dose'] = [routine_vx1]

    doubledose = hpv.default_vx(prod_name=product)
    doubledose.imm_init = dict(dist='beta_mean', par1=0.97, par2=0.025)
    d2_coverage = np.concatenate([np.linspace(0.1, 0.9, 9), np.repeat(0.9, n_years-9)])
    routine_vx2 = hpv.routine_vx(
        prob=d2_coverage,
        years=years,
        product=doubledose,
        age_range=routine_age,
        eligibility=eligibility,
        label='Routine vx'
    )

    vx_scenarios['Double dose'] = [routine_vx2]

    return vx_scenarios


def make_sims(location=None, calib_pars=None, vx_scenarios=None, end=2100):
    """ Set up scenarios """

    st_intv = []  # make_st()

    all_msims = sc.autolist()
    for name, vx_intv in vx_scenarios.items():
        sims = sc.autolist()
        for seed in range(n_seeds):
            interventions = vx_intv + st_intv
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
    do_save = False
    do_process = True
    end = 2100

    # Run scenarios (usually on VMs, runs n_seeds in parallel over M scenarios)
    if do_run:
        for location in loc.locations:
            fnlocation = location.replace(' ', '_')
            calib_pars = sc.loadobj(f'results/{location}_pars.obj')
            vx_scenarios = make_vx_scenarios(start_year=loc.vx_intro[location], end=end)
            msim = run_sims(location=location, vx_scenarios=vx_scenarios, end=end)

            if do_process:

                metrics = ['year', 'asr_cancer_incidence', 'n_precin_by_age', 'n_females_alive_by_age', 'cancers', 'cancer_deaths']

                # Process results
                scen_labels = list(['Baseline', 'Single dose', 'Double dose'])
                mlist = msim.split(chunks=len(scen_labels))

                msim_dict = sc.objdict()
                for si, scen_label in enumerate(scen_labels):
                    reduced_sim = mlist[si].reduce(output=True)
                    mres = sc.objdict({metric: reduced_sim.results[metric] for metric in metrics})

                    for ii, intv in enumerate(reduced_sim['interventions']):
                        intv_label = intv.label
                        mres[intv_label] = reduced_sim['interventions'][ii].n_products_used
                        if scen_label == 'Double dose':
                            mres[intv_label] = reduced_sim['interventions'][ii].n_products_used[:] * 2

                    msim_dict[scen_label] = mres

                sc.saveobj(f'results/{fnlocation}_vx_scens.obj', msim_dict)

    print('Done.')
