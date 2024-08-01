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

# Imports from this repository
import pars_data as dp
import utils as ut
import locations as loc

# %% Settings and filepaths
# Debug switch
debug = 0  # Run with smaller population sizes and in serial


# %% Simulation creation functions
def make_sim(location=None, calib=False, calib_pars=None, debug=0, interventions=None, seed=1, end=None, datafile=None):
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
        debut=ut.make_sb_data(location=location),
        mixing=dp.mixing[location],
        layer_probs=dp.make_layer_probs(location=location),
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

    sim = hpv.Sim(pars=pars, interventions=interventions, datafile=datafile, rand_seed=seed)

    return sim


# %% Simulation running functions
def run_sim(location=None, interventions=None, debug=0, seed=1, verbose=0.2,
        do_save=True, calib_par_stem=None, calib_pars=None, end=2020):

    dflocation = location.replace(' ', '_')
    if calib_pars is None and calib_par_stem is not None:
        calib_pars = sc.loadobj(f'results/{dflocation + calib_par_stem}.obj')

    # Make sim
    sim = make_sim(
        location=location,
        debug=debug,
        end=end,
        interventions=interventions,
        calib_pars=calib_pars
    )
    sim['rand_seed'] = seed
    sim.label = f'{location}--{seed}'

    # Run
    sim['verbose'] = verbose
    sim.run()
    sim.shrink()

    if do_save:
        sim.save(f'results/{dflocation}.sim')

    return sim


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

    for location in [loc.locations[0]]:
        # sim = run_sim(location=location)
        msim = run_parsets(location=location)

    T.toc('Done')

