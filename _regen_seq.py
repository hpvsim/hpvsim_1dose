"""Resumable, single-job 1dose v3 scenario regen (run ALONE, no concurrency).

For each of the 17 countries: if its results/<loc>_vx_scens.obj already has
NON-ZERO cohort cancers (i.e. a good fresh run), skip it; otherwise run + save.
Saving is per-country, so a kill loses at most the in-flight country. After all
17 are good, extract the plot-ready CSVs into results/_v3gap_scens.
"""
import os
import numpy as np
import sciris as sc
import run_scenarios as rsc
import locations as loc

OUT = 'results/_v3gap_scens'
sc.makefilepath(OUT + '/', makedirs=True)
rsc.debug = 0


def is_good(location):
    """True only for a FRESH v3 obj: tiny file (~2.5KB; stale v2 objs are >300KB
    and pickle hpvsim.base.Result which won't unpickle under v3), reduced to a
    scalar `cohort_cancers`, and non-zero."""
    dfl = location.replace(' ', '_')
    p = f'results/{dfl}_vx_scens.obj'
    if not os.path.exists(p) or os.path.getsize(p) > 50_000:
        return False
    try:
        m = sc.loadobj(p)
        for scen in m.values():
            if 'cohort_cancers' not in scen:
                return False
            cc = float(scen['cohort_cancers'])
            if not np.isfinite(cc) or cc <= 0:
                return False
        return True
    except Exception:
        return False


T = sc.timer()
todo = [c for c in loc.locations if not is_good(c)]
print(f'{len(loc.locations)-len(todo)}/{len(loc.locations)} already good; running {len(todo)}: {todo}', flush=True)
for i, location in enumerate(todo):
    print(f'[{i+1}/{len(todo)}] {location} ...', flush=True)
    rsc.run_location(location, end=2125, seeds=[0, 1], ms_agent_ratio=3, n_agents=10000)
    T.toc(f'  {location} done')

if all(is_good(c) for c in loc.locations):
    print('All 17 good -> extracting CSVs', flush=True)
    rsc.compile_direct(loc.locations, out=f'{OUT}/results_direct.csv')
    import utils as ut
    ut.extract_fig2_csvs(loc.locations, vx_scens_dir='results', out_dir=OUT)
    ut.extract_figS6_country_ts_csv(loc.locations, vx_scens_dir='results', out_dir=OUT)
    print('ALL DONE ->', OUT, flush=True)
else:
    bad = [c for c in loc.locations if not is_good(c)]
    print(f'INCOMPLETE - still missing: {bad} (re-run this script to resume)', flush=True)
