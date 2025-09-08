"""
Define the HPVsim simulation objects.
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
import hpvsim as hpv
import pandas as pd

# Imports from this repository
import pars_data as dp
import utils as ut
import locations as loc
import analyzers as an

# %% Settings and filepaths
# Debug switch
debug = 0  # Run with smaller population sizes and in serial


# %% Simulation creation functions
def make_sim(location=None, calib=False, calib_pars=None, debug=0, marriage_scale=None, debut_bias=None,
            interventions=None, analyzers=None, seed=1, end=None, datafile=None):
    """"
    Define parameters, analyzers, and interventions for the simulation
    """
    if end is None:
        end = 2100
    if calib:
        end = 2020

    pars = dict(
        n_agents=[10e3, 1e3][debug],
        dt=[0.25, 1.0][debug],
        start=[1960, 1980][debug],
        end=end,
        network='default',
        genotypes=[16, 18, 'hi5', 'ohr'],
        location=location,
        debut=ut.make_sb_data(location=location, debut_bias=debut_bias),
        mixing=dp.mixing[location],
        layer_probs=dp.make_layer_probs(marriage_scale=marriage_scale, location=location),
        f_partners=dp.f_partners,
        m_partners=dp.m_partners,
        init_hpv_dist=dp.init_genotype_dist[location],
        init_hpv_prev={
            'age_brackets': np.array([12, 17, 24, 34, 44, 64, 80, 150]),
            'm': np.array([0.0, 0.25, 0.6, 0.25, 0.05, 0.01, 0.0005, 0]),
            'f': np.array([0.0, 0.35, 0.7, 0.25, 0.05, 0.01, 0.0005, 0]),
        },
        ms_agent_ratio=100,
        verbose=0.0,
    )

    if calib_pars is not None:
        pars = sc.mergedicts(pars, calib_pars)

    # Analyzers
    if calib:
        analyzers = []
    else:
        analyzers = [an.cohort_cancers()] + sc.tolist(analyzers)
    sim = hpv.Sim(pars=pars, interventions=interventions, analyzers=analyzers, datafile=datafile, rand_seed=seed)

    return sim


# %% Simulation running functions
def run_sim(location=None, interventions=None, analyzers=None, debug=0, seed=1, verbose=0.2,
        marriage_scale=None, debut_bias=None, do_save=True, calib_pars=None, end=2100, do_shrink=True):

    dflocation = location.replace(' ', '_')
    if calib_pars is None:
        calib_pars = sc.loadobj(f'results/{dflocation}_pars.obj')
        if 'hiv_pars' in calib_pars:
            # Remove hiv_pars if it exists, as we are not running HIV simulations here
            calib_pars.pop('hiv_pars', None)

    # Make sim
    sim = make_sim(
        location=location,
        debug=debug,
        marriage_scale=marriage_scale,
        debut_bias=debut_bias,
        end=end,
        interventions=interventions,
        analyzers=analyzers,
        calib_pars=calib_pars,
    )
    sim['rand_seed'] = seed
    sim.label = f'{location}--{seed}'

    # Run
    sim['verbose'] = verbose
    sim.run()
    if do_shrink: sim.shrink()

    if do_save:
        sim.save(f'results/{dflocation}.sim')

    return sim


def make_popsims(end=2024):
    """ Set up scenarios """
    sims = sc.autolist()
    for location in loc.locations:
        sim = make_sim(location=location, end=end)
        sims += sim
    msim = hpv.MultiSim(sims)
    return msim


def run_popsims(end=2024, verbose=0.1):
    """ Run the simulations """
    msim = make_popsims(end=end)
    msim.run(verbose=verbose, keep_people=True)
    dfs = []
    for sim in msim.sims:
        dd = dict()
        location = sim.pars['location']
        dd['location'] = location
        ppl = sim.people
        ps = sim.pars['pop_scale']
        for age in range(9, 20):
            dd[age] = np.count_nonzero(ppl.is_female & (ppl.age>age) & (ppl.age<=(age+1)) & ppl.alive)*ps
        dfs += [pd.DataFrame(dd, index=[2020])]
    ddf = pd.concat(dfs)
    sc.saveobj('results/simpops.df', ddf)
    return msim


def run_sims(
        locations=None, debug=False, verbose=-1, analyzers=None,
        marriage_scale=1, debut_bias=[0, 0], do_save=False, *args, **kwargs
):
    """ Run multiple simulations in parallel """

    kwargs = sc.mergedicts(dict(debug=debug, verbose=verbose, analyzers=analyzers,
                                marriage_scale=marriage_scale, debut_bias=debut_bias), kwargs)
    simlist = sc.parallelize(run_sim, iterkwargs=dict(location=locations), kwargs=kwargs, serial=debug, die=True)
    sims = sc.objdict({location: sim for location, sim in zip(locations, simlist)})  # Convert from a list to a dict

    if do_save:
        for loc,sim in sims.items():
            sim.save(f'results/{loc}.sim')

    return sims


def run_parsets(
        location=None, debug=False, verbose=.1, interventions=None, save_results=True, **kwargs):
    ''' Run multiple simulations in parallel '''

    dflocation = location.replace(' ', '_')
    parsets = sc.loadobj(f'results/{dflocation}_pars_all.obj')
    kwargs = sc.mergedicts(dict(location=location, debug=debug, verbose=verbose, interventions=interventions), kwargs)
    simlist = sc.parallelize(run_sim, iterkwargs=dict(calib_pars=parsets), kwargs=kwargs, serial=debug, die=True)
    msim = hpv.MultiSim(simlist)
    msim.reduce()
    if save_results:
        sc.saveobj(f'results/msims/{dflocation}.mres', msim.results)

    return msim


# %% Run as a script
if __name__ == '__main__':
    T = sc.timer()

    # run_popsims(end=2023, verbose=0.1)

    for location in ['mali']:  #loc.locations:
        # sim = make_sim(location=location, end=2025)
        # sim = run_sim(location=location, end=2025, do_shrink=False)
        msim = run_parsets(location=location)


    T.toc('Done')

