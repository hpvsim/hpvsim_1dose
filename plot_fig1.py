"""
Fig 1: doses shipped vs girls reached in 2023 + 2024, under three scenarios
(complete utilization, actual utilization, 2-dose counterfactual).

Reads `data/fig1_allocation.csv` produced by `scripts/compute_fig1.py`.
"""
import argparse
import os

import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc

import utils as ut


SCENARIOS = ['complete', 'actual', 'cf']
SCENARIO_LABELS = {
    'complete': 'Complete\nutilization',
    'actual':   'Actual\nutilization',
    'cf':       '2-dose\ncounterfactual',
}


def _stacked_bars(ax, df, metric, colors):
    """One grouped-bar subplot: x = scenario, grouped by year, stacked 1/2-dose."""
    years = sorted(df['year'].unique())
    x = np.arange(len(SCENARIOS))
    width = 0.38
    for i, year in enumerate(years):
        offset = (i - (len(years) - 1) / 2) * width
        bottoms = np.zeros(len(SCENARIOS))
        for sched in [1, 2]:
            vals = []
            for scen in SCENARIOS:
                sub = df[(df['year'] == year) & (df['schedule'] == sched)]
                v = sub[f'{scen}_{metric}_m'].values
                vals.append(v[0] if len(v) else 0)
            vals = np.array(vals)
            ax.bar(
                x + offset, vals, width, bottom=bottoms,
                color=colors[sched - 1],
                label=f'{year} — {sched}-dose' if metric == 'doses' else None,
                edgecolor='white', linewidth=0.5,
            )
            # In-bar totals for visibility
            for xi, bi, vi in zip(x + offset, bottoms, vals):
                if vi > 1.5:
                    ax.text(xi, bi + vi / 2, f'{vi:.1f}', ha='center', va='center',
                            fontsize=9, color='white')
            bottoms = bottoms + vals
        # Total on top of each bar
        for xi, total in zip(x + offset, bottoms):
            if total > 0:
                ax.text(xi, total + 0.8, f'{total:.1f}M', ha='center', va='bottom',
                        fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_fig1(datafolder='data', outpath='figures/fig1.png'):
    df = pd.read_csv(f'{datafolder}/fig1_allocation.csv')

    ut.set_font(14)
    fig, (ax_doses, ax_girls) = pl.subplots(1, 2, figsize=(13, 6), sharey=False)

    colors = sc.gridcolors(2)

    _stacked_bars(ax_doses, df, 'doses', colors)
    _stacked_bars(ax_girls, df, 'girls', colors)

    ax_doses.set_title('Doses allocated (millions)')
    ax_girls.set_title('Girls reached (millions)')
    ax_doses.set_ylabel('Millions')

    # Shared legend: year × schedule swatches + year separators
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=colors[0], label='1-dose schedule'),
        Patch(facecolor=colors[1], label='2-dose schedule'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center', bbox_to_anchor=(0.5, 0.99),
        ncol=2, frameon=False, fontsize=12,
    )
    # Year labels under the x-ticks: 2023 | 2024 pair
    for ax in (ax_doses, ax_girls):
        for i, _ in enumerate(SCENARIOS):
            ax.annotate('2023', xy=(i - 0.2, 0), xytext=(i - 0.2, -0.06),
                        xycoords=('data', 'axes fraction'),
                        ha='center', va='top', fontsize=9, color='0.4')
            ax.annotate('2024', xy=(i + 0.2, 0), xytext=(i + 0.2, -0.06),
                        xycoords=('data', 'axes fraction'),
                        ha='center', va='top', fontsize=9, color='0.4')

    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    sc.savefig(outpath, dpi=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', default='data')
    parser.add_argument('--outpath', default='figures/fig1.png')
    args = parser.parse_args()
    plot_fig1(datafolder=args.datafolder, outpath=args.outpath)
    print(f'Wrote {args.outpath}')
