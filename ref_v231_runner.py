"""
v2.3.1 REFERENCE runner (run under .venv-v2, hpvsim==2.3.1).

Reproduces the SAME reduced-scale config as the v3 run_scenarios.py
(bangladesh + nigeria, 3 seeds, 4 dose scenarios, n_agents=20k,
ms_agent_ratio=10, dt=0.25, 1960-2125) using the v2.3 API, and writes
results/<loc>_vx_scens.obj with the same 'raw_cohort_cancers' layout the
CSV extractors expect. This is a throwaway reference generator kept out of the
v3 code path (the repo's analyzers.py/run_sim.py are now v3).
"""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np
import sciris as sc
import hpvsim as hpv
import hpvsim.utils as hpu
assert hpv.__version__ == '2.3.1', hpv.__version__

import pars_data as dp
import utils as ut
import locations as loc

DT = 0.25
N_AGENTS = 20000
MS = 10
START = 1960
END = 2125
SEEDS = [0, 1, 2]
SCENARIO_NAMES = ['No vaccination', 'Double dose',
                  'Single dose shipments', 'Single dose actual']


# --- v2 cohort_cancers analyzer (from the original repo analyzers.py) ---
class cohort_cancers(hpv.Analyzer):
    def __init__(self, cohort_age=None, start=None, **kwargs):
        super().__init__(**kwargs)
        self.start = start or 2024
        self.cohort_age = cohort_age or [9, 16]
        self.years = None
        self.results = None

    def initialize(self, sim):
        super().initialize()
        self.si = sc.findfirst(sim.res_yearvec, self.start)
        self.npts = len(sim.res_yearvec[self.si:])
        self.years = sim.res_yearvec[self.si:]
        self.results = np.zeros(self.npts)

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start:
            li = np.floor(sim.yearvec[sim.t])
            idx = sc.findfirst(self.years, li)
            ppl = sim.people
            te = sim.yearvec[sim.t] - self.start
            car = [self.cohort_age[0] + te, self.cohort_age[1] + te]
            cic = (ppl.date_cancerous == sim.t) & (ppl.age >= car[0]) & (ppl.age <= car[1])
            if cic.any():
                self.results[idx] += sum(ppl.scale[hpu.true(cic)])


# --- v2.3 sim build (from run_sim.py on v2.3-prob-conversion) ---
def _to_annual_prob(p, dt):
    p = np.clip(p, 0, 1 - 1e-10)
    return 1 - (1 - p) ** (1 / dt)


def _layer_probs_to_annual(layer_probs, dt):
    out = {}
    for lkey, lp in layer_probs.items():
        a = np.asarray(lp).copy().astype(float)
        for row in [1, 2]:
            a[row, :] = _to_annual_prob(a[row, :], dt)
        out[lkey] = a
    return out


def _convert_calib_pars_to_annual(calib_pars, dt):
    if calib_pars is None:
        return calib_pars
    out = dict(calib_pars)
    for key in ('m_cross_layer', 'f_cross_layer'):
        if key in out and out[key] is not None:
            out[key] = float(_to_annual_prob(out[key], dt))
    if 'layer_probs' in out and out['layer_probs'] is not None:
        out['layer_probs'] = _layer_probs_to_annual(out['layer_probs'], dt)
    return out


def make_sim(location, calib_pars, interventions, seed, end=END):
    layer_probs = _layer_probs_to_annual(dp.make_layer_probs(location=location), DT)
    pars = dict(
        n_agents=N_AGENTS, dt=DT, start=START, end=end, network='default',
        genotypes=[16, 18, 'hi5', 'ohr'], location=location,
        debut=ut.make_sb_data(location=location),
        mixing=dp.mixing[location], layer_probs=layer_probs,
        f_partners=dp.f_partners, m_partners=dp.m_partners,
        init_hpv_dist=dp.init_genotype_dist[location],
        init_hpv_prev={
            'age_brackets': np.array([12, 17, 24, 34, 44, 64, 80, 150]),
            'm': np.array([0.0, 0.25, 0.6, 0.25, 0.05, 0.01, 0.0005, 0]),
            'f': np.array([0.0, 0.35, 0.7, 0.25, 0.05, 0.01, 0.0005, 0]),
        },
        ms_agent_ratio=MS, verbose=0.0,
    )
    if calib_pars is not None:
        pars = sc.mergedicts(pars, _convert_calib_pars_to_annual(calib_pars, DT))
    sim = hpv.Sim(pars=pars, interventions=interventions,
                  analyzers=[cohort_cancers()], rand_seed=seed)
    return sim


def make_vx_scenarios(location, year=2024):
    if location in ['ethiopia', 'laos', 'zambia']:
        routine_age = (9, 17)
    elif location in ['togo']:
        routine_age = (9, 16)
    elif location in ['tanzania']:
        routine_age = (9, 19)
    else:
        routine_age = (9, 15)
    product = 'bivalent' if location in ['bangladesh', 'cambodia', 'togo', 'zimbabwe'] else 'nonavalent'

    def sd():
        v = hpv.default_vx(prod_name=product)
        v.imm_init = 0.985
        return v
    elig = lambda sim: (sim.people.doses == 0)
    scen = dict()
    scen['No vaccination'] = []
    scen['Single dose shipments'] = [hpv.campaign_vx(prob=loc.vx_coverage_shipped[location], years=year, product=sd(), sex=0, age_range=routine_age, eligibility=elig, interpolate=False, annual_prob=False, label='Single dose shipments')]
    scen['Single dose actual'] = [hpv.campaign_vx(prob=loc.vx_coverage_actual[location], years=year, product=sd(), sex=0, age_range=routine_age, eligibility=elig, interpolate=False, annual_prob=False, label='Single dose actual')]
    scen['Double dose'] = [hpv.campaign_vx(prob=loc.vx_coverage_cf[location], years=year, product=sd(), sex=0, age_range=routine_age, eligibility=elig, interpolate=False, annual_prob=False, label='Double dose')]
    return scen


def run_location(location):
    dfl = location.replace(' ', '_')
    calib_pars = sc.loadobj(f'results/{dfl}_pars.obj')
    calib_pars.pop('hiv_pars', None)
    scen = make_vx_scenarios(location)
    md = sc.objdict()
    for sname in SCENARIO_NAMES:
        per_seed = []
        for seed in SEEDS:
            intvs = make_vx_scenarios(location)[sname]
            sim = make_sim(location, calib_pars, intvs, seed)
            sim.run(verbose=0)
            a = sim.get_analyzer('cohort_cancers')
            per_seed.append(np.asarray(a.results))
        raw = np.stack(per_seed, axis=-1)  # (n_years, n_seeds)
        sums = raw.sum(axis=0)
        md[sname] = sc.objdict(
            raw_cohort_cancers=raw,
            cohort_cancers=float(np.quantile(sums, 0.5)),
            cohort_cancers_low=float(np.quantile(sums, 0.1)),
            cohort_cancers_high=float(np.quantile(sums, 0.9)),
        )
        print(f'  {location} / {sname}: cohort median={md[sname]["cohort_cancers"]:.0f}', flush=True)
    sc.saveobj(f'results/{dfl}_vx_scens.obj', md)
    return md


if __name__ == '__main__':
    T = sc.timer()
    for location in ['bangladesh', 'nigeria']:
        print(f'Running {location} ...', flush=True)
        run_location(location)
    print('Done.')
    T.toc()
