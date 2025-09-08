"""
Plot time series of cancers in vaccination cohort
"""


import pylab as pl
import sciris as sc
import pandas as pd
import utils as ut
import numpy as np
import locations as loc


def plot_single(ax, mres, years, color, label=None, smooth=True, normalize=False):

    best = mres['med'][1:-1]
    low = mres['lb'][1:-1]
    high = mres['ub'][1:-1]

    if smooth:
        best = np.convolve(list(best), np.ones(5), "valid")/5
        low = np.convolve(list(low), np.ones(5), "valid")/5
        high = np.convolve(list(high), np.ones(5), "valid")/5
        years = years[4:]

    if normalize:
        cohorts = loc.vx_coverage_denom
        denom = sum(cohorts.values())
        best = best / denom
        low = low / denom
        high = high / denom
    ax.plot(years, best, color=color, label=label)
    ax.fill_between(years, low, high, alpha=0.1, color=color)
    return ax


# %% Run as a script
if __name__ == '__main__':

    # Process data into one large dataframe
    do_process = True
    if do_process:

        scennames = ['No vaccination', 'Double dose', 'Single dose shipments', 'Single dose actual']
        results = {k: np.zeros((102, 20)) for k in scennames}  # 102 years, 20 scenarios
        locations = loc.locations
        for location in locations:
            mres = sc.loadobj(f'results/{location.replace(" ", "_")}_vx_scens.obj')
            for sname in scennames:
                mres_scen = mres[sname]
                results[sname] += mres_scen['raw_cohort_cancers']
        results['year'] = mres_scen['year']

        # Process diffs
        diffs = results['Double dose'] - results['Single dose shipments']
        diffs_med = np.cumsum(np.median(diffs, axis=1))
        diffs_lb = np.cumsum(np.quantile(diffs, q=0.1, axis=1))
        diffs_ub = np.cumsum(np.quantile(diffs, q=0.9, axis=1))
        diffs2 = results['Double dose'] - results['Single dose actual']
        diffs2_med = np.cumsum(np.median(diffs2, axis=1))
        diffs2_lb = np.cumsum(np.quantile(diffs2, q=0.1, axis=1))
        diffs2_ub = np.cumsum(np.quantile(diffs2, q=0.9, axis=1))

        # Save results
        res_stats = sc.objdict()
        for sname in scennames:
            res_stats[sname] = sc.objdict()
            res_stats[sname]['med'] = np.cumsum(np.quantile(results[sname], q=0.5, axis=-1))
            res_stats[sname]['lb'] = np.cumsum(np.quantile(results[sname], q=0.1, axis=-1))
            res_stats[sname]['ub'] = np.cumsum(np.quantile(results[sname], q=0.9, axis=-1))
        res_stats['year'] = results['year']

        # Save
        sc.saveobj('results/res_stats.obj', res_stats)
        sc.saveobj('results/diffs_shipped.obj', {'med': diffs_med, 'lb': diffs_lb, 'ub': diffs_ub, 'year': results['year']})
        sc.saveobj('results/diffs_actual.obj', {'med': diffs2_med, 'lb': diffs2_lb, 'ub': diffs2_ub, 'year': results['year']})

    # Plot the data
    do_plot = True
    if do_plot:
        ut.set_font(24)
        legendfont = 20
        fig, axes = pl.subplots(2, 1, figsize=(15, 12))
        axes = axes.ravel()

        res_stats = sc.loadobj('results/res_stats.obj')
        diffs = sc.loadobj('results/diffs_shipped.obj')
        diffs2 = sc.loadobj('results/diffs_actual.obj')
        colors = sc.gridcolors(6)

        # Top row: cervical cancers over time
        # Hack because the years are stored differently
        ax = axes[0]
        year = res_stats['year']
        start_year = 2025
        end_year = 2125
        si = sc.findinds(year, start_year)[0]
        ei = sc.findinds(year, end_year)[0]
        year = year[si:ei]

        # Plot cumulative cancers
        for sn, scen in enumerate(res_stats.keys()):
            if scen != 'year':
                res = res_stats[scen]
                if scen == 'No vaccination':
                    label = 'No vaccination'
                elif scen == 'Double dose':
                    label = 'Counterfactual double-dose allocation'
                elif scen == 'Single dose shipments':
                    label = 'Single-dose regimen with complete utilization'
                elif scen == 'Single dose actual':
                    label = 'Single-dose regimen with actual utilization'
                ax = plot_single(ax, res, year, colors[sn], label=label)
        ax.set_title('(A) Cumulative cervical cancers in 2023/24 vaccination cohort')
        ax.set_ylim(bottom=0, top=1.8e6)
        sc.SIticks(ax)
        ax.legend(loc='upper left', frameon=False, prop={'size': legendfont})

        # Plot cumulative cervical cancers averted
        ax = axes[1]
        ax = plot_single(ax, diffs, year, colors[4], smooth=False, label='Complete utilization')
        ax = plot_single(ax, diffs2, year, colors[5], smooth=False, label='Actual utilization')
        ax.legend(loc='upper left', frameon=False, prop={'size': legendfont})
        ax.set_title('(B) Cumulative cervical cancers averted by single-dose in 2023/24 vaccination cohort')
        ax.set_ylim(bottom=0, top=500e3)
        sc.SIticks(ax)

        fig.tight_layout()
        fig_name = 'figures/fig2_ts.png'
        sc.savefig(fig_name, dpi=100)
