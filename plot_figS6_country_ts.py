"""
Fig S6: cumulative cancers in the vaccination cohort, by country and scenario.

Reads `figS6_country_ts.csv` produced by `utils.extract_figS6_country_ts_csv`.
"""
import argparse
import os

import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc

import utils as ut
from locations import locations, location_labels


def _plot_single(ax, sub, color, label=None, smooth=False):
    years = sub.year.values
    best = sub.cum_med.values
    low = sub.cum_lb.values
    high = sub.cum_ub.values
    if smooth and len(best) >= 5:
        best = np.convolve(best, np.ones(5), 'valid') / 5
        low = np.convolve(low, np.ones(5), 'valid') / 5
        high = np.convolve(high, np.ones(5), 'valid') / 5
        years = years[4:]
    ax.plot(years, best, color=color, label=label)
    ax.fill_between(years, low, high, alpha=0.1, color=color)
    return ax


def plot_cohort_ts(resfolder='results/v2.2.6_baseline',
                   outpath='figures/figS6_country_ts.png'):
    df = pd.read_csv(f'{resfolder}/figS6_country_ts.csv')

    ut.set_font(16)
    n_plots = len(locations) + 1  # +1 for the shared legend
    fig, axes = sc.getrowscols(n_plots, make=True, remove_extra=True, figsize=(12, 10))
    axes = axes.flatten()

    scenarios = list(df.scenario.unique())
    colors = sc.gridcolors(len(scenarios))

    for pn, location in enumerate(locations):
        ax = axes[pn]
        for cn, slabel in enumerate(scenarios):
            sub = df[(df.location == location) & (df.scenario == slabel)].sort_values('year')
            if not sub.empty:
                _plot_single(ax, sub, color=colors[cn], label=slabel)
        if pn in [0, 4, 8, 12, 16]:
            ax.set_ylabel('Cumulative cancers', fontsize=12)
        ax.set_ylim(bottom=0)
        ax.set_title(location_labels[location])
        sc.SIticks(ax)

    # Shared legend in the spare axis
    ax = axes[len(locations)]
    for cn, slabel in enumerate(scenarios):
        ax.plot(np.nan, np.nan, color=colors[cn], label=slabel)
    ax.axis('off')
    ax.legend(loc='center', fontsize=16, frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    sc.savefig(outpath, dpi=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resfolder', default='results/v2.2.6_baseline')
    parser.add_argument('--outpath', default='figures/figS6_country_ts.png')
    args = parser.parse_args()
    plot_cohort_ts(resfolder=args.resfolder, outpath=args.outpath)
    print('Done.')
