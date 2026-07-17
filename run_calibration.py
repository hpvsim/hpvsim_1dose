"""
Calibrate (HPVsim v3).

IMPORTANT — API CHANGE. HPVsim v3 delegates calibration to Starsim's
``ss.Calibration`` (re-exported as ``hpv.Calibration``), whose signature is:

    hpv.Calibration(sim, calib_pars, *, data=None, weights=None,
                    gof_kwargs=None, build_fn=None, eval_fn=None, eval_kw=None,
                    **kwargs)

This is quite different from the v2 ``Calibration(sim, calib_pars=,
genotype_pars=, datafiles=, extra_sim_result_keys=, total_trials=, n_workers=,
storage=)`` used previously:

* ``calib_pars`` is an Optuna-style search space applied via a ``build_fn``
  that mutates the sim in place. For this project the tuned parameters are the
  network cross-layer probabilities and casual ``partners`` (network-level in
  v3, see run_sim.build_network) plus per-genotype ``cin_fn`` shape ``k`` — so
  ``build_fn`` must rebuild the ``hpv.SexualNetwork`` / ``genotype_pars`` from
  the sampled values (the v2 flat ``calib_pars`` dict path no longer applies).
* Age-stratified cancer targets are supplied via ``data=`` and scored with an
  ``eval_fn`` over an ``hpv.AgeResults`` analyzer (see the v3 calibration
  tutorial), replacing v2's ``datafiles`` + ``extra_sim_result_keys``.

Per the migration mandate this calibration is NOT run (it is a long Optuna
search); the committed per-country ``results/<loc>_pars.obj`` are reused by
run_sim/run_scenarios. Porting the search itself to ``build_fn``/``eval_fn`` and
re-fitting is a follow-up (and, per the verification findings, IS required for
v3 — the v2-fit transmission parameters do not sustain the epidemic on the v3
engine). The scaffold below shows the intended v3 structure.
"""

import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')

import sciris as sc
import numpy as np
import hpvsim as hpv

import run_sim as rs
import utils as ut
import locations as loc

# CONFIGURATIONS
to_run = [
    # 'run_calibration',  # long Optuna search — do NOT enable casually
    'load_pars',          # reuse committed calibrated pars
]
debug = False
locations = loc.locations
n_trials = [1000, 1][debug]
n_workers = [40, 1][debug]
storage = None


def make_priors():
    """Search-space priors (unchanged in intent): per-genotype CIN-curve slope
    ``k`` for hi5/ohr. In v3 these are applied through ``genotype_pars`` in the
    Calibration ``build_fn``."""
    return dict(
        hi5=dict(cin_fn=dict(k=[.2, .15, .4, 0.01])),
        ohr=dict(cin_fn=dict(k=[.2, .15, .4, 0.01])),
    )


def run_calib(location=None, n_trials=None, n_workers=None,
              do_plot=False, do_save=True, filestem=''):
    """v3 calibration scaffold. See module docstring: requires a ``build_fn``
    that rebuilds the network / genotype_pars from sampled values and an
    ``eval_fn`` scoring age-stratified cancers against ``data``. Not executed in
    this migration."""
    raise NotImplementedError(
        'v3 calibration must be ported to the ss.Calibration build_fn/eval_fn '
        'API (see module docstring); not run per the migration mandate.'
    )


def load_calib(location=None, do_plot=True, which_pars=0, save_pars=True,
               filestem=''):
    """Load a pre-run calibration object.

    NOTE: the committed ``raw_results/<loc>_calib.obj`` are v2.x calibration
    objects; their ``trial_pars_to_sim_pars`` / plotting API differs under v3's
    ss.Calibration. Reuse of the already-extracted ``results/<loc>_pars.obj``
    (a plain dict of calibrated parameters) is what run_sim/run_scenarios rely
    on and is version-independent.
    """
    dfl = location.replace(' ', '_')
    return sc.load(f'results/{dfl}_pars.obj')


if __name__ == '__main__':
    T = sc.timer()
    if 'run_calibration' in to_run:
        for location in locations:
            run_calib(location=location, n_trials=n_trials, n_workers=n_workers)
    if 'load_pars' in to_run:
        for location in locations[:1]:
            pars = load_calib(location=location)
            print(location, 'calibrated beta =', pars.get('beta'))
    T.toc('Done')
