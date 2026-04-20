"""
Fig 2: time series of cancers in the vaccination cohort across all countries.

Reads plot-ready CSVs produced by `utils.extract_fig2_csvs`:
  - fig2_res_stats.csv — cumulative med/lb/ub per scenario × year
  - fig2_diffs.csv     — cumulative cancers averted by single-dose (shipments / actual)

Run `python -c "import utils as ut; ut.extract_all_csvs()"` on the VM (after
regenerating `results/*_vx_scens.obj`) to refresh the CSVs.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc

import utils as ut


def _plot_single(ax, sub, color, label=None, smooth=True):
    years = sub.year.values
    best = sub.cum_med.values
    low = sub.cum_lb.values
    high = sub.cum_ub.values
    # Match original behaviour: drop first and last points
    years, best, low, high = years[1:-1], best[1:-1], low[1:-1], high[1:-1]
    if smooth and len(best) >= 5:
        best = np.convolve(best, np.ones(5), 'valid') / 5
        low = np.convolve(low, np.ones(5), 'valid') / 5
        high = np.convolve(high, np.ones(5), 'valid') / 5
        years = years[4:]
    ax.plot(years, best, color=color, label=label)
    ax.fill_between(years, low, high, alpha=0.1, color=color)
    return ax


def plot_fig2(resfolder='results/v2.2.6_baseline',
              outpath='figures/fig2_ts.png'):
    res_stats = pd.read_csv(f'{resfolder}/fig2_res_stats.csv')
    diffs = pd.read_csv(f'{resfolder}/fig2_diffs.csv')

    ut.set_font(24)
    legendfont = 20
    fig, axes = pl.subplots(2, 1, figsize=(15, 10))
    axes = axes.ravel()
    colors = sc.gridcolors(6)

    label_map = {
        'No vaccination': 'No vaccination',
        'Double dose': 'Counterfactual 2-dose allocation with complete utilization',
        'Single dose actual': 'Single-dose regimen with actual utilization',
        'Single dose shipments': 'Single-dose regimen with complete utilization',
    }
    scenarios_order = ['No vaccination', 'Double dose',
                       'Single dose actual', 'Single dose shipments']

    ax = axes[0]
    for sn, scen in enumerate(scenarios_order):
        sub = res_stats[res_stats.scenario == scen].sort_values('year')
        # Filter to 2025-2125 window
        sub = sub[(sub.year >= 2025) & (sub.year <= 2125)]
        _plot_single(ax, sub, color=colors[sn], label=label_map[scen])
    ax.set_title('(A) Cumulative cervical cancers in 2023/24 vaccination cohort')
    ax.set_ylim(bottom=0, top=1.8e6)
    sc.SIticks(ax)
    ax.legend(loc='upper left', frameon=False, prop={'size': legendfont})

    ax = axes[1]
    diff_map = {
        'Single dose shipments vs Double dose': ('Complete utilization', colors[4]),
        'Single dose actual vs Double dose': ('Actual utilization', colors[5]),
    }
    for cmp_label, (legend_label, color) in diff_map.items():
        sub = diffs[diffs.comparison == cmp_label].sort_values('year')
        sub = sub[(sub.year >= 2025) & (sub.year <= 2125)]
        _plot_single(ax, sub, color=color, label=legend_label, smooth=False)
    ax.legend(loc='upper left', frameon=False, prop={'size': legendfont})
    ax.set_title('(B) Cumulative cervical cancers averted by single-dose in 2023/24 vaccination cohort')
    ax.set_ylim(bottom=0, top=500e3)
    sc.SIticks(ax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    sc.savefig(outpath, dpi=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resfolder', default='results/v2.2.6_baseline')
    parser.add_argument('--outpath', default='figures/fig2_ts.png')
    args = parser.parse_args()
    plot_fig2(resfolder=args.resfolder, outpath=args.outpath)
    print('Done.')
